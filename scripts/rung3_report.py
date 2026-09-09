"""Rung 3: what the larger index bought, and what the GPU bought on top of it.

    python scripts/rung3_report.py \\
        --pair configs/rung3-untrained.yaml configs/rung3.yaml \\
        --pair configs/rung3-gte-base-untrained.yaml configs/rung3-gte-base.yaml \\
        --against reference/screens/rung2.json \\
        --json reference/rung3-report.json

Two things move between rung 2 and rung 3 — the index grows from 32,043
documents to 70,579, and the encoder is fine-tuned — and the whole design of
this report is to keep them apart. Each encoder is run twice at the *same*
all-subjects index, off the shelf and with its adapter, so the difference
between those two rows is the fine-tune and nothing else. Reading them against
that encoder's rung-2 row then splits the total into the part index size bought
and the part the GPU bought, which is the question of how much of the final
score requires a GPU at all.

Each row goes through `screen_encoders.screen`, so every number here comes from
the same `pipeline.retrieve` and `pipeline.combine` the pipeline itself uses.

The second half of the report is band migration. The frozen bands never move
(docs/artifacts.md), so growing the corpus cannot reclassify a label; what it
can do is make a label that no tib-core training record carried reachable by the
neighbour retriever after all. That is reported here as its own quantity, in two
weightings — one vote per label, and one per gold assignment — and the number it
exists for is the one that does *not* move: the labels no corpus of documents
can reach, which bound what document similarity can ever score on this
benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms4subjects.config import ExperimentConfig, load_experiment  # noqa: E402
from llms4subjects.corpus import (  # noqa: E402
    UNSEEN_BAND,
    MissingDataset,
    band_for_count,
    frequency_band_reference,
    label_counts,
)
from llms4subjects.hardware import describe_device, select_device  # noqa: E402
from llms4subjects.finetune import AdapterMismatch, read_adapter  # noqa: E402
from llms4subjects.splits import SplitAlignmentUnverified  # noqa: E402
from llms4subjects.stages import encoders  # noqa: E402
from llms4subjects.stages.evaluator import BANDS  # noqa: E402
from compare_rungs import NotAComparison as LoadingError  # noqa: E402
from compare_rungs import load_screen  # noqa: E402
from run_experiment import FORBIDDEN_SPLIT, load_inputs  # noqa: E402
from screen_encoders import (  # noqa: E402
    HELD_EQUAL,
    _without_disabled,
    SELECTION_K,
    TABLE_KS,
    Screened,
    _row_document,
    _Scorer,
    screen,
)

# Sections that must be identical inside a pair. `encoder` is not one of them —
# the adapter lives there — so it is checked field by field below.
PAIRED_EQUAL = HELD_EQUAL

# Not `encoder-screen/1`. A screen is one row per encoder and is compared to
# another screen by `scripts/compare_rungs.py`; this document is two rows per
# encoder, and the two share a model name. Written under the screen's schema it
# would load, and `Screen.row("BAAI/bge-m3")` would silently return whichever of
# the trained and untrained rows came first. Its own schema makes that comparison
# a refusal instead.
SCHEMA = "rung3-report/1"


class NotAComparison(ValueError):
    """A pair whose two rows differ in more than the fine-tune."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("UNTRAINED", "TRAINED"),
        required=True,
        help="an off-the-shelf config and its fine-tuned counterpart",
    )
    parser.add_argument("--split", default="core_dev", help="the split to predict")
    parser.add_argument("--limit", type=int, help="score only the first N records")
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    parser.add_argument(
        "--against",
        help="a rung-2 screen, to split the gain into index and training",
    )
    parser.add_argument("--json", dest="json_path", help="write the rung-3 screen here")
    args = parser.parse_args(argv)

    if args.split == FORBIDDEN_SPLIT:
        print(f"{FORBIDDEN_SPLIT} is the gold test split; see ticket 17.")
        return 1

    pairs = [
        (load_experiment(untrained), load_experiment(trained))
        for untrained, trained in args.pair
    ]
    try:
        check_pairs(pairs)
    except NotAComparison as error:
        print(error)
        return 1

    try:
        inputs = load_inputs(pairs[0][0], args.split, args.limit)
    except (MissingDataset, SplitAlignmentUnverified) as error:
        print(error)
        return 1
    except KeyError as error:
        print(error.args[0] if error.args else error)
        return 1

    against = None
    if args.against:
        # Before the screening rather than after it: an unreadable or
        # incomparable screen would otherwise surface once four rows had been
        # computed, and `--json` would never be written.
        try:
            against = load_screen(args.against)
            check_against(against, pairs, args.split)
        except (LoadingError, NotAComparison) as error:
            print(error)
            return 1

    device = select_device(args.device)
    print(f"rung 3 over {len(inputs.records)} {args.split} records")
    print(f"data revision: {inputs.revision}")
    print(describe_device(device))
    print(
        f"index: {len(inputs.index_records)} documents from "
        f"{', '.join(pairs[0][0].index.corpora)}"
    )
    if inputs.assembly:
        print(f"index corpus: {inputs.rows} rows read, {inputs.assembly}")
    print()

    scorer = _Scorer(inputs.records, pairs[0][0].fusion.candidates)
    rows: list[tuple[Screened, Screened]] = []
    for untrained, trained in pairs:
        scored = []
        for config in (untrained, trained):
            state = "fine-tuned" if config.encoder.adapter else "off the shelf"
            print(f"  {config.encoder.name} ({state}) ... ", end="", flush=True)
            started = time.perf_counter()
            row = screen(config, inputs, scorer, device, args.artifacts)
            scored.append(row)
            print(
                f"micro R@{SELECTION_K} "
                f"{row.fused.micro.recall(SELECTION_K):.4f} in "
                f"{time.perf_counter() - started:.1f}s ({row.cache})"
            )
        rows.append((scored[0], scored[1]))

    print()
    print(render_training(rows, SELECTION_K))
    print()
    print(render_bands(rows, SELECTION_K))
    print()
    print(render_per_retriever(rows, SELECTION_K))

    if against is not None:
        print()
        print(render_attribution(rows, against, SELECTION_K))

    print()
    print(render_migration(inputs.index_records, inputs.records))
    print()
    print(render_adapters(rows))

    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                document(
                    rows,
                    split=args.split,
                    records=len(inputs.records),
                    revision=inputs.revision,
                    device=device,
                    rows_read=inputs.rows,
                    indexed=len(inputs.index_records),
                    dropped=inputs.dropped,
                ),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0


def document(
    rows: Sequence[tuple[Screened, Screened]],
    *,
    split: str,
    records: int,
    revision: str,
    device: str,
    rows_read: int,
    indexed: int,
    dropped: int,
) -> dict:
    """The whole rung as plain data, for the same reason a screen is written.

    Rung 3 costs a GPU run and several hours of Apple Silicon per row, and what
    it supports — "training added N points, of which the index bought M" — is
    arithmetic over four numbers. Written out, that arithmetic is re-derivable
    from a clean checkout; held in a terminal, it is a table someone retyped.
    """
    first = rows[0][0].config
    return {
        "schema": SCHEMA,
        "split": split,
        "records": records,
        "data_revision": revision,
        "selection_k": SELECTION_K,
        "device": device,
        "index": {
            "documents": indexed,
            "rows_read": rows_read,
            "dropped_held_out": dropped,
            "corpora": list(first.index.corpora),
        },
        "pairs": [
            {
                "encoder": untrained.name,
                "untrained": _row_document(untrained),
                "trained": _row_document(trained),
                "adapter": _adapter_document(trained),
            }
            for untrained, trained in rows
        ],
    }


def _adapter_document(trained: Screened) -> dict:
    """What the fine-tune was, as the adapter itself records it."""
    try:
        manifest = read_adapter(
            trained.config.encoder.adapter,
            trained.config.encoder.name,
            encoders.resolve(trained.config.encoder).revision,
        )
    except AdapterMismatch:
        return {"path": trained.config.encoder.adapter}
    return {"path": trained.config.encoder.adapter, **manifest}


# --- One variable inside a pair ----------------------------------------------


def check_pairs(pairs: Sequence[tuple[ExperimentConfig, ExperimentConfig]]) -> None:
    """Refuse a pair whose difference is anything but the fine-tune.

    The claim this report makes is "training added N points", and it is only
    that claim if the two rows share an index, a label rendering, a fusion and a
    model. A pair that also moved the index would be reporting the sum of two
    changes under one name — which is the mistake rung 2 was designed to avoid
    and this rung inherits.
    """
    for untrained, trained in pairs:
        if untrained.encoder.adapter is not None:
            raise NotAComparison(
                f"{untrained.name} names an adapter, so it is not the "
                "off-the-shelf row of the pair"
            )
        if trained.encoder.adapter is None:
            raise NotAComparison(
                f"{trained.name} names no adapter, so the pair would be one "
                "configuration run twice"
            )
        if untrained.encoder.name != trained.encoder.name:
            raise NotAComparison(
                f"{untrained.encoder.name} against {trained.encoder.name}: a "
                "pair measures a fine-tune, so both rows are one model"
            )
        differing = {
            section: (untrained.section(section), trained.section(section))
            for section in PAIRED_EQUAL
            if untrained.section(section) != trained.section(section)
        }

        if untrained.encoder.max_length != trained.encoder.max_length:
            differing["encoder.max_length"] = (
                untrained.encoder.max_length,
                trained.encoder.max_length,
            )
        if differing:
            described = "\n".join(
                f"  {section}: {before} against {after}"
                for section, (before, after) in sorted(differing.items())
            )
            raise NotAComparison(
                f"{untrained.name} and {trained.name} differ in more than the "
                f"adapter, so the difference between them is not the "
                f"fine-tune:\n{described}"
            )

    # And across the pairs, not only inside them. Every row is scored against
    # the index and the gold of `pairs[0][0]` — one corpus is loaded, once — so
    # a second pair naming a different index would be scored against the first
    # pair's documents while caching its vectors under its own key. The rows
    # would look like a comparison and be two different experiments.
    first = pairs[0][0]
    for untrained, trained in pairs[1:]:
        across = {
            section: (first.section(section), untrained.section(section))
            for section in PAIRED_EQUAL
            if first.section(section) != untrained.section(section)
        }
        if across:
            described = "\n".join(
                f"  {section}: {before} against {after}"
                for section, (before, after) in sorted(across.items())
            )
            raise NotAComparison(
                f"{first.name} and {untrained.name} are in the same report and "
                f"differ in more than their encoder, but every row is scored "
                f"against one index:\n{described}"
            )


def check_against(screen, pairs, split: str) -> None:
    """Refuse a rung-2 screen whose difference from this rung is not the index.

    `scripts/compare_rungs.py` makes this check between two screens and refuses
    a pair that disagrees on anything but `index`; the same claim is being made
    here — "the index adds N" — against a document written weeks earlier, so it
    gets the same guard rather than being trusted for having the right schema.

    Each encoder is checked against its own row, because that is how a screen
    records what it held equal, and an encoder the screen never ran is not an
    error: `render_attribution` prints a dash for it.

    The dataset revision is deliberately not enforced. The rung-2 screen
    predates `all_train.csv`, and building that file changes the dataset
    revision without changing a byte of either split the two runs read, so a
    mismatch here is expected — it is reported beside the table instead.
    """
    if screen.split != split:
        raise NotAComparison(
            f"the screen at {screen.path} scored {screen.split!r} and this "
            f"report scores {split!r}; the two columns would be answers to "
            "different questions"
        )

    for untrained, _ in pairs:
        try:
            row = screen.row(untrained.encoder.name)
        except KeyError:
            continue
        differing = {
            section: (held, _without_disabled(untrained.section(section)))
            for section, held in row.held_equal.items()
            if "." not in section
            and held != _without_disabled(untrained.section(section))
        }
        if differing:
            described = "\n".join(
                f"  {section}: {before} against {after}"
                for section, (before, after) in sorted(differing.items())
            )
            raise NotAComparison(
                f"{untrained.encoder.name} was screened at {screen.path} under "
                f"a different configuration, so the difference between the "
                f"rungs is not the index alone:\n{described}"
            )


# --- Band migration ----------------------------------------------------------


def migration(
    labels: Sequence[str],
    frozen: Mapping[str, int],
    counts: Mapping[str, int],
    boundaries: Sequence[Mapping[str, object]],
    weights: Mapping[str, int] | None = None,
) -> dict[str, dict[str, int]]:
    """Where each label's frozen band would be at the grown corpus's counts.

    Keyed frozen band to would-be band. The frozen band is read from the
    committed counts and never recomputed — that is what "frozen" means, and it
    is why this is a separate table rather than a redrawn one. The would-be band
    is what the same boundaries say about the label's occurrences in the corpus
    now being indexed.

    `weights` counts each label as something other than one — gold assignments,
    when the question is what share of the score a migration touches, rather
    than how many entries of the vocabulary it moves.
    """
    moved: dict[str, dict[str, int]] = {}
    for code in labels:
        before = band_for_count(frozen.get(code, 0), boundaries)
        after = band_for_count(counts.get(code, 0), boundaries)
        weight = weights.get(code, 1) if weights is not None else 1
        row = moved.setdefault(before, {})
        row[after] = row.get(after, 0) + weight
    return moved


def unreachable(moved: Mapping[str, Mapping[str, int]]) -> int:
    """What is still zero-shot after the corpus grew: the finding's own number.

    A label no training document carries is one no neighbour harvest can ever
    propose, at any corpus size, so this is the part of the benchmark that only
    label text can reach.
    """
    return moved.get(UNSEEN_BAND, {}).get(UNSEEN_BAND, 0)


def corpus_counts(records) -> dict[str, int]:
    """How often each code is assigned in the corpus about to be indexed."""
    counted: dict[str, int] = {}
    for record in records:
        for code in dict.fromkeys(record.subjects):
            counted[code] = counted.get(code, 0) + 1
    return counted


# --- Rendering ---------------------------------------------------------------


def _ks(row: Screened, k: int) -> tuple[int, ...]:
    available = set(row.fused.ks)
    ceiling = row.config.fusion.candidates
    return tuple(
        at for at in sorted({*TABLE_KS, k, ceiling})
        if at in available and at <= ceiling
    )


def render_training(rows: Sequence[tuple[Screened, Screened]], k: int) -> str:
    """The deliverable: each encoder off the shelf and fine-tuned, side by side."""
    ks = _ks(rows[0][0], k)
    lines = [
        f"### Fine-tuning at the same index (dev micro recall, selection R@{k})\n",
        "| encoder | weights | "
        + " | ".join(f"R@{at}" for at in ks)
        + f" | official R@{k} |",
        "|---|---" + "|---:" * (len(ks) + 1) + "|",
    ]
    for untrained, trained in rows:
        for row, state in ((untrained, "off the shelf"), (trained, "fine-tuned")):
            cells = " | ".join(f"{row.fused.micro.recall(at):.4f}" for at in ks)
            lines.append(
                f"| {row.name} | {state} | {cells} | "
                f"{row.fused.official_macro.recall(k):.4f} |"
            )
        official = (
            trained.fused.official_macro.recall(k)
            - untrained.fused.official_macro.recall(k)
        )
        lines.append(
            f"| {untrained.name} | **what training added** | "
            + " | ".join(
                f"{trained.fused.micro.recall(at) - untrained.fused.micro.recall(at):+.4f}"
                for at in ks
            )
            + f" | {official:+.4f} |"
        )
    return "\n".join(lines)


def render_bands(rows: Sequence[tuple[Screened, Screened]], k: int) -> str:
    """What training added per frozen band, which is where it is spent."""
    lines = [
        f"### What training added, by frozen frequency band (micro R@{k})\n",
        "| encoder | weights | " + " | ".join(BANDS) + " |",
        "|---|---" + "|---:" * len(BANDS) + "|",
    ]
    for untrained, trained in rows:
        for row, state in ((untrained, "off the shelf"), (trained, "fine-tuned")):
            lines.append(
                f"| {row.name} | {state} | "
                + " | ".join(_band_cell(row, band, k) for band in BANDS)
                + " |"
            )
        lines.append(
            f"| {untrained.name} | **delta** | "
            + " | ".join(
                _band_delta(untrained, trained, band, k) for band in BANDS
            )
            + " |"
        )
    return "\n".join(lines)


def _band_cell(row: Screened, band: str, k: int) -> str:
    if band not in row.fused.by_band:
        return "—"
    return f"{row.fused.by_band[band].recall(k):.4f}"


def _band_delta(untrained: Screened, trained: Screened, band: str, k: int) -> str:
    if band not in untrained.fused.by_band or band not in trained.fused.by_band:
        return "—"
    return (
        f"{trained.fused.by_band[band].recall(k) - untrained.fused.by_band[band].recall(k):+.4f}"
    )


def render_per_retriever(rows: Sequence[tuple[Screened, Screened]], k: int) -> str:
    """Which retriever the training reached, from the same pass.

    The lexical column is the control, and only for an encoder without sparse
    term weights: BM25 over label strings reads no vectors, so fine-tuning
    cannot move it. Where the encoder emits its own term weights the lexical
    retriever is the encoder, and it moves — which is worth seeing rather than
    hiding, since it is a second retriever the one fine-tune bought.
    """
    names = sorted({name for pair in rows for row in pair for name in row.per_retriever})
    lines = [
        f"### Each retriever alone, off the shelf and fine-tuned (micro R@{k})\n",
        "| encoder | weights | " + " | ".join(names) + " | fused |",
        "|---|---" + "|---:" * (len(names) + 1) + "|",
    ]
    for untrained, trained in rows:
        for row, state in ((untrained, "off the shelf"), (trained, "fine-tuned")):
            cells = " | ".join(
                f"{row.per_retriever[name].micro.recall(k):.4f}"
                if name in row.per_retriever
                else "—"
                for name in names
            )
            lines.append(
                f"| {row.name} | {state} | {cells} | "
                f"{row.fused.micro.recall(k):.4f} |"
            )
    return "\n".join(lines)


def render_attribution(
    rows: Sequence[tuple[Screened, Screened]], rung2, k: int
) -> str:
    """The rung's headline: how much of the gain needed a GPU.

    Three numbers per encoder — its rung-2 score, its score at the larger index
    untrained, and its score fine-tuned — and the two differences between them.
    The first difference is index size, which cost Apple Silicon hours; the
    second is the fine-tune, which cost the GPU host.
    """
    lines = [
        f"### Index size against training (dev micro R@{k})\n",
        f"The earlier column is {rung2.path}, taken at dataset revision "
        f"`{rung2.data_revision}`. Building the all-subjects split changes that "
        "revision without changing a byte of the tib-core splits either run "
        "read, so the two are comparable and the difference is the index.\n",
        f"| encoder | rung 2 | rung 3, off the shelf | rung 3, fine-tuned | "
        "index adds | training adds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for untrained, trained in rows:
        try:
            before = rung2.row(untrained.name).micro[k]
        except KeyError:
            lines.append(f"| {untrained.name} | — | — | — | — | — |")
            continue
        middle = untrained.fused.micro.recall(k)
        after = trained.fused.micro.recall(k)
        lines.append(
            f"| {untrained.name} | {before:.4f} | {middle:.4f} | {after:.4f} | "
            f"{middle - before:+.4f} | {after - middle:+.4f} |"
        )
    return "\n".join(lines)


def render_migration(index_records, scored_records) -> str:
    """How many zero-shot labels the larger corpus reaches, and how many it never will."""
    reference = frequency_band_reference()
    boundaries = reference["boundaries"]
    frozen = label_counts()
    counts = corpus_counts(index_records)
    # The same count over the scored split: how many gold assignments each
    # code accounts for, which is what weights a migration by what it is
    # worth rather than by how many vocabulary entries it touches.
    weights = corpus_counts(scored_records)

    # The labels the score is actually made of: every code the scored split's
    # gold carries. The vocabulary-wide table would be dominated by the 64,820
    # codes nothing in this benchmark ever assigns.
    labels = sorted(weights)
    by_label = migration(labels, frozen, counts, boundaries)
    by_assignment = migration(labels, frozen, counts, boundaries, weights=weights)

    zero_labels = sum(by_label.get(UNSEEN_BAND, {}).values())
    moved_labels = zero_labels - unreachable(by_label)
    zero_assignments = sum(by_assignment.get(UNSEEN_BAND, {}).values())
    moved_assignments = zero_assignments - unreachable(by_assignment)
    total_assignments = sum(weights.values())

    lines = [
        "### Band migration: what the extra 38,545 documents reach\n",
        f"Bands stay frozen against tib-core train counts, so nothing below "
        f"changes a table elsewhere. Of the {zero_labels:,} labels in the scored "
        f"split's gold that no tib-core training record carries, the "
        f"all-subjects corpus reaches **{moved_labels:,}** and leaves "
        f"**{unreachable(by_label):,}** unreachable — "
        f"{unreachable(by_assignment) / max(total_assignments, 1):.1%} of gold "
        f"assignments that no document-similarity method can ever propose, at "
        f"any corpus size.\n",
        "| frozen band | labels | reached by the larger corpus | still zero |",
        "|---|---:|---:|---:|",
    ]
    for band in BANDS:
        row = by_label.get(band)
        if not row:
            continue
        total = sum(row.values())
        still = row.get(UNSEEN_BAND, 0)
        lines.append(f"| {band} | {total:,} | {total - still:,} | {still:,} |")

    lines += [
        "",
        "| weighting | zero-shot before | reached | still zero |",
        "|---|---:|---:|---:|",
        f"| one vote per label | {zero_labels:,} | {moved_labels:,} | "
        f"{unreachable(by_label):,} |",
        f"| one vote per gold assignment | {zero_assignments:,} | "
        f"{moved_assignments:,} | {unreachable(by_assignment):,} |",
    ]

    # Where the reached labels landed, since a label reached once is not a label
    # the neighbour retriever will find: it is one document out of 70,579.
    landed = by_label.get(UNSEEN_BAND, {})
    lines += [
        "",
        "| where the reached labels landed | labels |",
        "|---|---:|",
    ]
    for band in BANDS:
        if band == UNSEEN_BAND or band not in landed:
            continue
        lines.append(f"| {band} | {landed[band]:,} |")

    return "\n".join(lines)


def render_adapters(rows: Sequence[tuple[Screened, Screened]]) -> str:
    """What each fine-tune was, read off the adapter rather than off a flag."""
    lines = [
        "### The adapters\n",
        "| encoder | adapter | trainable | pairs | epochs | loss | host | minutes |",
        "|---|---|---:|---:|---:|---|---|---:|",
    ]
    for _, trained in rows:
        directory = trained.config.encoder.adapter
        try:
            manifest = read_adapter(
                directory,
                trained.config.encoder.name,
                encoders.resolve(trained.config.encoder).revision,
            )
        except AdapterMismatch as error:
            lines.append(f"| {trained.name} | {directory} | — | — | — | — | — | — |")
            print(f"warning: {error}", file=sys.stderr)
            continue
        losses = manifest.get("losses") or []
        lines.append(
            f"| {trained.name} | `{directory}` | "
            f"{manifest.get('trainable_parameters', 0) / 1e6:.1f}M | "
            f"{manifest.get('examples', 0):,} | "
            f"{manifest.get('hyperparameters', {}).get('epochs', '—')} | "
            + (" → ".join(f"{loss:.4f}" for loss in losses) or "—")
            + f" | {manifest.get('host', '—')} | "
            f"{manifest.get('seconds', 0) / 60:.1f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
