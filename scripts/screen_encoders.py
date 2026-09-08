"""Screen the four candidate encoders through the whole of stage one.

    python scripts/screen_encoders.py configs/rung1.yaml \\
        configs/rung1-e5-large.yaml configs/rung1-bge-m3.yaml \\
        configs/rung1-gte-base.yaml

    python scripts/screen_encoders.py configs/rung1.yaml ... --cold
    python scripts/screen_encoders.py configs/rung1.yaml ... --limit 500
    python scripts/screen_encoders.py configs/rung2.yaml ... \\
        --json reference/screens/rung2.json

Rungs 1 and 2 of the experiment ladder, and the cheapest decision in the
project: which encoder deserves the GPU budget. All four candidates run off the
shelf at one index size — 8,000 documents at rung 1, the full 32,043 at rung 2 —
entirely on the Mac, and are ranked on dev micro Recall@10, the project's
model-selection metric, with the frozen frequency bands broken out beside it.

The script is one rung at a time. `--json` writes the whole screen out, and
`scripts/compare_rungs.py` reads two of those files to answer the question rung
2 exists for: whether the ranking survived the index growing fourfold.

Screening untrained is deliberate. An off-the-shelf encoder will not beat a
fine-tuned one, but the ranking among encoders is expected to survive
fine-tuning, so screening here costs nothing where training all four at every
rung would cost nine to twelve GPU runs.

**One variable.** The screen refuses configs that differ in anything but their
encoder, because a run whose index size or fusion weights also moved would rank
those instead. It refuses two configs naming the same model for the same
reason. Each config is run through `pipeline.retrieve` and `pipeline.combine` —
the two halves of `pipeline.predict` — so every number here comes from the code
path the pipeline itself uses, and each retriever's own score is free from the
same pass.

**Wall clock.** Recorded per encoder, with the cache state that produced it: a
warm embedding cache reads vectors back from disk, so its seconds are a measure
of I/O and not of the encoder. `--cold` runs each pass against a throwaway
artifact root, which is what makes the seconds a cost — at the price of
discarding the vectors afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms4subjects.artifacts import EMBEDDING_STAGE, ArtifactStore  # noqa: E402
from llms4subjects.config import ExperimentConfig, load_experiment  # noqa: E402
from llms4subjects.corpus import MissingDataset, frequency_bands  # noqa: E402
from llms4subjects.hardware import describe_device, select_device  # noqa: E402
from llms4subjects.models import load_registry  # noqa: E402
from llms4subjects.pipeline import combine, retrieve  # noqa: E402
from llms4subjects.stages import encoders  # noqa: E402
from llms4subjects.stages.indexes import select_documents  # noqa: E402
from llms4subjects.stages.evaluator import (  # noqa: E402
    BANDS,
    EvaluationReport,
    evaluate,
)
from compare_rungs import SCHEMA  # noqa: E402
from run_experiment import FORBIDDEN_SPLIT, load_inputs  # noqa: E402

# Model selection is micro Recall@10 on dev, everywhere in this project.
SELECTION_K = 10

# The k values the screen's tables report. The candidate ceiling is added from
# each config's `fusion.candidates`, which the comparability check holds equal.
TABLE_KS = (5, 10, 50)

# Sections that must be identical across the screened configs. `encoder` is the
# variable; `name` and `notes` are how a config identifies itself to a reader.
HELD_EQUAL = ("index", "label_text", "retrievers", "fusion", "group_prior",
              "reranker", "adjudication")

# What a written screen carries so that a *later* screen can be compared
# against it (ticket 10): everything above except `index`, which is the one
# thing the rungs of the ladder are allowed to differ in. Persisting `index`
# here would make `scripts/compare_rungs.py` refuse every comparison it exists
# to make, and leaving it out is safe because index size is the axis that
# comparison orders its columns by.
PERSISTED_EQUAL = tuple(section for section in HELD_EQUAL if section != "index")

# The one field inside the encoder section that is not the variable. How much of
# a document the model reads is a property of the experiment, not of the model:
# a candidate given 256 tokens against another given 512 would be screened on
# its input length. `batch_size` is deliberately not here — it changes wall
# clock and nothing else, and a large model may need a smaller one to fit.
HELD_EQUAL_IN_ENCODER = ("max_length",)


class NotAScreen(ValueError):
    """Configs that would produce four numbers that are not a comparison."""


@dataclass(frozen=True)
class Screened:
    """One encoder's whole result: what it scored, and what it cost."""

    config: ExperimentConfig
    dimensions: int
    parameters: int | None
    fused: EvaluationReport
    per_retriever: Mapping[str, EvaluationReport]
    load_seconds: float
    retrieve_seconds: float
    # How many matrices of vectors this pass had to compute: the index corpus,
    # the label tower and the split being predicted are three, and a kNN-only
    # run is two. Counted rather than inferred from the cache directory
    # existing, because a pass that found one and computed two spent nearly all
    # of a cold run's time, and calling that "warm" would publish an encoder's
    # real cost as I/O.
    wrote: int

    @property
    def name(self) -> str:
        return self.config.encoder.name

    @property
    def seconds(self) -> float:
        return self.load_seconds + self.retrieve_seconds

    @property
    def cache(self) -> str:
        """How to read this row's seconds: what it computed, or nothing."""
        return f"computed {self.wrote}" if self.wrote else "warm"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="one config per encoder")
    parser.add_argument("--split", default="core_dev", help="the split to predict")
    parser.add_argument("--limit", type=int, help="score only the first N records")
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    parser.add_argument(
        "--cold",
        action="store_true",
        help="time each pass against a throwaway artifact root, discarding vectors",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="also write the screen here, for scripts/compare_rungs.py",
    )
    args = parser.parse_args(argv)

    if args.split == FORBIDDEN_SPLIT:
        print(f"{FORBIDDEN_SPLIT} is the gold test split; see ticket 17.")
        return 1

    configs = [load_experiment(path) for path in args.configs]
    try:
        check(configs)
    except NotAScreen as error:
        print(error)
        return 1

    try:
        inputs = load_inputs(configs[0], args.split, args.limit)
    except MissingDataset as error:
        print(error)
        return 1
    except KeyError as error:
        print(error.args[0] if error.args else error)
        return 1

    device = select_device(args.device)
    print(f"screening {len(configs)} encoders over {len(inputs.records)} "
          f"{args.split} records")
    print(f"data revision: {inputs.revision}")
    print(describe_device(device))
    indexed = len(select_documents(inputs.index_records, configs[0].index))
    print(f"index: {indexed} of {len(inputs.index_records)} documents from "
          f"{', '.join(configs[0].index.corpora)}, "
          f"stratify={configs[0].index.stratify}, seed={configs[0].index.seed}")
    print("timing: cold, against a throwaway artifact root" if args.cold
          else f"timing: against {args.artifacts}, cache state reported per row")
    print()
    print(render_stratification(inputs.index_records, configs[0]))
    print()

    scorer = _Scorer(inputs.records, configs[0].fusion.candidates)
    rows = []
    for config in configs:
        print(f"  {config.encoder.name} ... ", end="", flush=True)
        row = screen(
            config, inputs, scorer, device, args.artifacts, cold_timing=args.cold
        )
        rows.append(row)
        print(
            f"micro R@{SELECTION_K} {row.fused.micro.recall(SELECTION_K):.4f} "
            f"in {row.seconds:.1f}s ({row.cache})"
        )

    ranked = rank(rows, SELECTION_K)
    print()
    print(render_ranking(ranked, SELECTION_K))
    print()
    print(render_bands(ranked, SELECTION_K))
    print()
    print(render_per_retriever(ranked, SELECTION_K))
    print()
    print(render_cost(ranked))
    print()
    print(render_provenance(ranked))

    if args.json_path:
        write_screen(
            args.json_path,
            ranked,
            split=args.split,
            records=len(inputs.records),
            revision=inputs.revision,
            device=device,
            corpus=len(inputs.index_records),
            indexed=len(select_documents(inputs.index_records, configs[0].index)),
        )
        print(f"\nwrote {args.json_path}")
    return 0


def document(
    rows: Sequence[Screened],
    *,
    split: str,
    records: int,
    revision: str,
    device: str,
    corpus: int,
    indexed: int,
) -> dict:
    """The whole screen as plain data, for `scripts/compare_rungs.py`.

    A screen costs hours of Apple Silicon, and the claim it supports — whether
    the encoder ranking survives the index growing — is a statement about two
    screens taken days apart. Written out, the comparison is arithmetic over two
    files that can be re-run in milliseconds and re-read after the fact; held
    only in a terminal, it would be a table someone retyped.
    """
    first = rows[0].config
    return {
        "schema": SCHEMA,
        "split": split,
        "records": records,
        "data_revision": revision,
        "selection_k": SELECTION_K,
        "device": device,
        "index": {
            "documents": indexed,
            "corpus": corpus,
            "corpora": list(first.index.corpora),
            "stratify": first.index.stratify,
            "seed": first.index.seed,
        },
        "rows": [_row_document(row) for row in rows],
    }


def _row_document(row: Screened) -> dict:
    return {
        "encoder": row.name,
        "config": row.config.name,
        "dimensions": row.dimensions,
        "parameters": row.parameters,
        "load_seconds": row.load_seconds,
        "retrieve_seconds": row.retrieve_seconds,
        "wrote": row.wrote,
        "micro": _at_k(row.fused.micro, row.fused.ks),
        "official_macro": _at_k(row.fused.official_macro, row.fused.ks),
        "bands": {
            band: _at_k(metrics, row.fused.ks)
            for band, metrics in row.fused.by_band.items()
        },
        "per_retriever": {
            name: _at_k(report.micro, report.ks)
            for name, report in row.per_retriever.items()
        },
        "held_equal": {
            **{
                section: _without_disabled(row.config.section(section))
                for section in PERSISTED_EQUAL
            },
            **{
                f"encoder.{field}": getattr(row.config.encoder, field)
                for field in HELD_EQUAL_IN_ENCODER
            },
        },
    }


def _without_disabled(value):
    """A stage that is off keeps only its off switch, whatever else it declares.

    A screen document outlives the code that wrote it, and every later ticket
    adds fields to the config sections it works on: ticket 13 gave
    `adjudication` a temperature, a token budget and four more knobs. A rung-1
    screen taken before that and a rung-2 screen taken after it would then
    disagree on `adjudication` and `scripts/compare_rungs.py` would refuse the
    comparison — over the parameters of a stage that ran in neither.

    So a section reporting `enabled: false` is persisted as exactly that. It
    cannot have moved a number it never touched, and what is left is the claim
    worth checking: that the stage was off at both index sizes.
    """
    if isinstance(value, dict):
        if value.get("enabled") is False:
            return {"enabled": False}
        return {key: _without_disabled(item) for key, item in value.items()}
    return value


def _at_k(metrics, ks: Sequence[int]) -> dict[str, float]:
    """Recall at each scored k. JSON object keys are strings, so k is one too."""
    return {str(k): metrics.recall(k) for k in ks}


def write_screen(path, rows: Sequence[Screened], **about) -> None:
    """Write the screen document, creating its directory if it is missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document(rows, **about), indent=2, sort_keys=True, ensure_ascii=False)
    )


def check(configs: Sequence[ExperimentConfig]) -> None:
    """Refuse anything that would make the four numbers incomparable."""
    if len(configs) < 2:
        raise NotAScreen(
            f"a screen compares encoders, so it needs at least two configs; "
            f"got {len(configs)}"
        )

    names = [config.encoder.name for config in configs]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise NotAScreen(
            f"{', '.join(repeated)} named by more than one config; a screen has "
            "one row per encoder"
        )

    differing = differences(configs)
    if differing:
        described = "\n".join(
            f"  {section}: " + " against ".join(sorted(set(map(str, values))))
            for section, values in sorted(differing.items())
        )
        raise NotAScreen(
            "the configs differ in more than their encoder, so the screen would "
            f"rank those differences too:\n{described}"
        )


def differences(
    configs: Sequence[ExperimentConfig],
) -> dict[str, list[object]]:
    """What the configs disagree on, other than which encoder they name."""
    differing = {
        section: [config.section(section) for config in configs]
        for section in HELD_EQUAL
        if len({str(config.section(section)) for config in configs}) > 1
    }
    for field in HELD_EQUAL_IN_ENCODER:
        values = [getattr(config.encoder, field) for config in configs]
        if len(set(values)) > 1:
            differing[f"encoder.{field}"] = list(values)
    return differing


def screen(
    config: ExperimentConfig,
    inputs,
    scorer: "_Scorer",
    device: str,
    artifacts: str,
    cold_timing: bool = False,
) -> Screened:
    """One encoder through the whole of stage one, timed.

    The encoder is loaded here rather than inside `retrieve` so that the load is
    timed apart from the pass and so that its size and dimensions — the other
    half of a cost claim — come from the model that actually ran. Which is also
    why the config is pinned first: `retrieve` keys its vector cache on the
    config it is handed, and exempts one whose encoder was supplied by the
    caller, since that encoder may not be the model the config names. Here it
    is exactly that model, so the pin has to be in the config for this pass to
    share a cache entry with every other harness.
    """
    config = encoders.pinned(config)

    with _artifact_root(artifacts, cold_timing) as root:
        store = ArtifactStore(root, data_revision=inputs.revision)
        # The same key `pipeline.retrieve` will compute, so the count below is
        # of the matrices this pass itself computes.
        cached = store.directory(EMBEDDING_STAGE, config)
        before = _matrices(cached)

        started = time.perf_counter()
        encoder = encoders.load(config.encoder, device)
        loaded = time.perf_counter()

        per_retriever = retrieve(
            inputs.records,
            config,
            inputs.vocabulary,
            inputs.index_records,
            store,
            device=device,
            encoder=encoder,
            name_qualifiers=inputs.name_qualifiers,
        )
        elapsed = time.perf_counter() - loaded
        wrote = _matrices(cached) - before

    return Screened(
        config=config,
        dimensions=encoder.dimensions,
        parameters=encoders.parameter_count(encoder),
        fused=scorer.score(combine(per_retriever, config)),
        per_retriever={
            name: scorer.score(combine({name: lists}, config))
            for name, lists in sorted(per_retriever.items())
        },
        load_seconds=loaded - started,
        retrieve_seconds=elapsed,
        wrote=wrote,
    )


def _matrices(directory) -> int:
    """How many cached vector matrices sit under this key right now."""
    return len(list(directory.glob("*.npy"))) if directory.is_dir() else 0


def _artifact_root(artifacts: str, cold_timing: bool):
    """Where this pass caches its vectors: the real store, or a throwaway one."""
    if not cold_timing:
        return _Kept(artifacts)
    return tempfile.TemporaryDirectory(prefix="screen-encoders-")


class _Kept:
    """The real artifact root, as a context manager, so `screen` has one shape."""

    def __init__(self, root: str):
        self._root = root

    def __enter__(self) -> str:
        return self._root

    def __exit__(self, *_) -> bool:
        return False


class _Scorer:
    """Fixed gold, bands and cells, so a row is one call over one ranking."""

    def __init__(self, records, ceiling: int):
        self._gold = {record.id: record.subjects for record in records}
        self._cells = {record.id: (record.type, record.lang) for record in records}
        self._bands = frequency_bands()
        self.ceiling = ceiling
        self.ks = tuple(sorted({*TABLE_KS, SELECTION_K, ceiling}))

    def score(self, candidates) -> EvaluationReport:
        return evaluate(
            gold=self._gold,
            predictions={result.record_id: result.codes for result in candidates},
            bands=self._bands,
            cells=self._cells,
            ks=self.ks,
        )


def rank(rows: Sequence[Screened], k: int) -> list[Screened]:
    """Best first on the selection metric, ties broken by name.

    Never by argument order: a tie decided by the command line would move with
    the shell history rather than with the measurement.
    """
    return sorted(rows, key=lambda row: (-row.fused.micro.recall(k), row.name))


def render_stratification(index_records, config: ExperimentConfig) -> str:
    """What this rung's index is made of, cell by cell.

    Printed rather than asserted, because the criterion is that the subset is
    stratified by record type and language *and reproducible*: the sample is a
    seeded draw from `index.seed`, so this table is the same table on any
    machine, and a reader can see that no cell was dropped for being small.

    Still printed at rung 2, where the index is the whole corpus and there is no
    draw to describe: the two rungs' tables side by side are what show that the
    larger index differs from the smaller one in size and not in composition.
    """
    selected = select_documents(index_records, config.index)
    corpus = _cells(index_records)
    sample = _cells(selected)

    lines = [
        f"### The index: {len(selected):,} of {len(index_records):,} documents, "
        f"{_how_drawn(config, len(selected), len(index_records))}\n",
        "| record type | language | corpus | corpus share | sampled | sampled share |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell in sorted(corpus, key=lambda cell: (-corpus[cell], cell)):
        lines.append(
            f"| {cell[0]} | {cell[1]} | {corpus[cell]} | "
            f"{corpus[cell] / len(index_records):.4f} | {sample.get(cell, 0)} | "
            f"{sample.get(cell, 0) / max(len(selected), 1):.4f} |"
        )
    return "\n".join(lines)


def _how_drawn(config: ExperimentConfig, selected: int, corpus: int) -> str:
    """How this index was chosen, which is not the same at every rung."""
    if selected >= corpus:
        return "unstratified — every document in the corpus"
    if config.index.stratify:
        return f"stratified (seed {config.index.seed})"
    return f"an unstratified draw (seed {config.index.seed})"


def _cells(records) -> dict[tuple[str, str], int]:
    counted: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record.type, record.lang)
        counted[key] = counted.get(key, 0) + 1
    return counted


def render_ranking(rows: Sequence[Screened], k: int) -> str:
    """The deliverable: four encoders ordered by dev micro Recall@10."""
    available = set(rows[0].fused.ks)
    # The candidate ceiling is `fusion.candidates` — what the pipeline emits —
    # rather than the largest k scored, and it is one column among the others
    # rather than an extra one, so a screen run at a ceiling below 50 reports
    # neither a duplicate column nor a recall past the candidates it has.
    ceiling = rows[0].config.fusion.candidates
    ks = tuple(
        at for at in sorted({*TABLE_KS, k, ceiling})
        if at in available and at <= ceiling
    )
    lines = [
        f"### Encoder screen, rung 1 (dev micro recall, selection is R@{k})\n",
        "| encoder | dim | "
        + " | ".join(f"R@{at}" for at in ks)
        + f" | official R@{k} |",
        "|---" + "|---:" * (len(ks) + 2) + "|",
    ]
    for row in rows:
        cells = " | ".join(f"{row.fused.micro.recall(at):.4f}" for at in ks)
        lines.append(
            f"| {row.name} | {row.dimensions} | {cells} | "
            f"{row.fused.official_macro.recall(k):.4f} |"
        )
    return "\n".join(lines)


def render_bands(rows: Sequence[Screened], k: int) -> str:
    """Where each encoder's score comes from, by frozen frequency band."""
    lines = [
        f"### The screen by frequency band (micro R@{k})\n",
        "| encoder | " + " | ".join(BANDS) + " |",
        "|---" + "|---:" * len(BANDS) + "|",
    ]
    for row in rows:
        cells = " | ".join(
            f"{row.fused.by_band[band].recall(k):.4f}"
            if band in row.fused.by_band
            else "—"
            for band in BANDS
        )
        lines.append(f"| {row.name} | {cells} |")
    return "\n".join(lines)


def render_per_retriever(rows: Sequence[Screened], k: int) -> str:
    """Each retriever alone, from the same pass, so a flip is attributable.

    The lexical column is the check on the rest: BM25 over label strings reads
    no vectors, so it must be identical for every encoder, and a screen where it
    is not has a variable nobody declared.
    """
    names = sorted({name for row in rows for name in row.per_retriever})
    lines = [
        f"### Each retriever alone, per encoder (micro R@{k})\n",
        "| encoder | " + " | ".join(names) + " | fused |",
        "|---" + "|---:" * (len(names) + 1) + "|",
    ]
    for row in rows:
        cells = " | ".join(
            f"{row.per_retriever[name].micro.recall(k):.4f}"
            if name in row.per_retriever
            else "—"
            for name in names
        )
        lines.append(f"| {row.name} | {cells} | {row.fused.micro.recall(k):.4f} |")
    return "\n".join(lines)


def render_cost(rows: Sequence[Screened]) -> str:
    """What each encoder cost, and whether the number is a cost at all."""
    lines = [
        "### Wall clock, per encoder\n",
        "| encoder | parameters | load | stage one | total | cache |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        size = f"{row.parameters / 1e6:.0f}M" if row.parameters else "—"
        lines.append(
            f"| {row.name} | {size} | {row.load_seconds:.1f}s | "
            f"{row.retrieve_seconds:.1f}s | {row.seconds:.1f}s | "
            f"{row.cache} |"
        )
    return "\n".join(lines)


def render_provenance(rows: Sequence[Screened]) -> str:
    """Every screened model's release date, against the declared cutoff."""
    registry = load_registry()
    lines = [
        f"### Model provenance (cutoff {registry.cutoff.isoformat()})\n",
        "| encoder | created | pinned revision | dated |",
        "|---|---|---|---|",
    ]
    for row in rows:
        entry = registry[row.name]
        lines.append(
            f"| {row.name} | {entry.created.isoformat()} | "
            f"`{entry.revision[:12]}` | {entry.revision_date.isoformat()} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
