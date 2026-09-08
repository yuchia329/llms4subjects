"""What an off-the-shelf cross-encoder is worth on top of the fused candidates.

    python scripts/rerank_report.py configs/rung1-rerank.yaml
    python scripts/rerank_report.py configs/rung1-rerank.yaml --sample 300
    python scripts/rerank_report.py configs/rung1-rerank-base.yaml --sample 300

One pass, four deliverables, which is why this is its own harness rather than a
flag on `run_experiment.py`:

- **Before and after.** The retrievers run once; the fused ranking is scored,
  then reranked and scored again. Both rows come from the same candidates, so
  the difference is the cross-encoder and nothing else.
- **The ceiling, next to the precision.** At 2.40 gold labels per record a
  perfect system scores 0.48 at k=5, so a P@5 of 0.15 is 31% of achievable
  rather than 15% of anything. Every precision figure here is printed with the
  achievable maximum for the same records beside it.
- **The confidence measure, calibrated.** The routing signal ticket 13 needs is
  only a signal if it correlates with being right, so it is bucketed against
  measured precision rather than asserted.
- **The evidence for the fine-tune decision.** A fine-tune would be a fourth run
  on the `nlp2` GPU host beyond the three the project has budgeted, so the
  recommendation is a budget decision and wants numbers under it.

Reranked lists are cached under the `reranked` artifact stage, keyed by the same
configuration fingerprint everything else uses, because the pass costs a
cross-encoder forward per candidate and the calibration table should not.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms4subjects.artifacts import ArtifactStore  # noqa: E402
from llms4subjects.config import (  # noqa: E402
    ExperimentConfig,
    IndexConfig,
    load_experiment,
)
from llms4subjects.contracts import (  # noqa: E402
    CODES_PER_RECORD,
    CandidateList,
    Record,
)
from llms4subjects.corpus import MissingDataset, frequency_bands  # noqa: E402
from llms4subjects.hardware import describe_device, select_device  # noqa: E402
from llms4subjects.models import MODEL_CUTOFF  # noqa: E402
from llms4subjects.pipeline import (  # noqa: E402
    combine,
    reranker_label_texts,
    retrieve,
)
from llms4subjects.stages import encoders, indexes, reranker  # noqa: E402
from llms4subjects.stages.evaluator import (  # noqa: E402
    BANDS,
    EvaluationReport,
    evaluate,
)
from run_experiment import FORBIDDEN_SPLIT, load_inputs  # noqa: E402

RERANK_STAGE = "reranked"

# The k values the before/after table reports. 5 and 10 are what the ticket
# asks for, 50 is the submission.
TABLE_KS = (5, 10, 25, 50)

# Model selection is micro Recall@10 on dev, as everywhere else in this project.
SELECTION_K = 10

# Buckets the confidence measure is calibrated over. Ten of them at 5,354 dev
# records is 535 records a bucket, which is enough that a bucket's P@5 is not
# noise and few enough that a monotone trend is visible by eye.
BUCKETS = 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="a config with reranker.enabled true")
    parser.add_argument("--split", default="core_dev", help="the split to predict")
    parser.add_argument(
        "--sample",
        type=int,
        help="score a stratified sample of N records rather than the whole split",
    )
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    parser.add_argument(
        "--query",
        choices=("document", "label"),
        help=(
            "override reranker.query for a screening run; the winning side is "
            "then committed to the config"
        ),
    )
    parser.add_argument(
        "--label-form",
        choices=("rendered", "name"),
        help="override reranker.label_form for a screening run",
    )
    parser.add_argument(
        "--mix",
        choices=("replace", "fuse"),
        help="override reranker.mix for a screening run",
    )
    parser.add_argument(
        "--sweep-mix",
        action="store_true",
        help=(
            "also sweep reranker.mix_weight over the cached scores and print "
            "the tuned block; costs no forward pass"
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="rerank even if a cached pass for this configuration exists",
    )
    args = parser.parse_args(argv)

    if args.split == FORBIDDEN_SPLIT:
        print(f"{FORBIDDEN_SPLIT} is the gold test split; see ticket 17.")
        return 1

    config = load_experiment(args.config)
    overrides = {
        name: value
        for name, value in (
            ("query", args.query),
            ("label_form", args.label_form),
            ("mix", args.mix),
        )
        if value is not None
    }
    if overrides:
        # The artifact key covers the whole reranker section, so an overridden
        # pairing or label form caches beside the config's rather than on top
        # of it.
        config = dataclasses.replace(
            config, reranker=dataclasses.replace(config.reranker, **overrides)
        )
    if not config.reranker.enabled:
        print(
            f"{args.config} has reranker.enabled false; this harness measures "
            "what reranking changes, so it needs the config that turns it on"
        )
        return 1

    if not [name for name, settings in config.retrievers.items() if settings.enabled]:
        print(
            f"{args.config} enables no retriever, so there are no candidates to "
            "rerank; the all-off ablation is a row in the retriever table"
        )
        return 1

    if config.group_prior.enabled:
        # `predict` boosts between fusion and reranking, and this harness fuses
        # and reranks itself. Applying the boost here would be a second
        # implementation of it; not applying it would print every row below
        # under the name of a pipeline that never ran.
        print(
            f"{args.config} enables the group prior, which this harness does "
            "not apply: it fuses and reranks. Use the config with the prior "
            "off, or scripts/ablate_group_prior.py for that stage."
        )
        return 1

    try:
        inputs = load_inputs(config, args.split)
    except MissingDataset as error:
        print(error)
        return 1
    except KeyError as error:
        print(error.args[0] if error.args else error)
        return 1

    records = _sampled(inputs.records, args.sample)
    device = select_device(args.device)
    print(f"experiment:    {config.name}  ({args.config})")
    print(f"data revision: {inputs.revision}")
    resolved = reranker.resolve(config.reranker)
    print(
        f"reranker:      {config.reranker.model} @ {resolved.revision[:12]}, "
        f"released {resolved.release.created}, revision dated "
        f"{resolved.release.revision_date} (cutoff {MODEL_CUTOFF})"
    )
    print(
        f"pairing:       {config.reranker.query} as the query, "
        f"{config.reranker.label_form} label text"
    )
    mix = config.reranker.mix
    print(
        "mixing:        "
        + (
            "the model's order replaces the fused one"
            if mix == "replace"
            else f"fused with the retrieval order at weight "
            f"{config.reranker.mix_weight} and rrf_k {config.reranker.mix_rrf_k}"
        )
    )
    print(describe_device(device))
    print(
        f"reranking the top {config.reranker.input_k} of {len(records)} "
        f"{args.split} records to {config.reranker.output_k}\n"
    )

    store = ArtifactStore(args.artifacts, data_revision=inputs.revision)

    started = time.perf_counter()
    per_retriever = retrieve(
        records,
        config,
        inputs.vocabulary,
        inputs.index_records,
        store,
        device=device,
        name_qualifiers=inputs.name_qualifiers,
    )
    fused = combine(per_retriever, config)
    print(f"retrieved and fused in {time.perf_counter() - started:.1f}s")

    # One name for this record set, used by every cached pass below: a sweep
    # row that spelled it differently would miss the cache it just wrote.
    scored_split = f"{args.split}-sample" if args.sample else args.split

    after, seconds = _reranked(
        records, fused, config, inputs, store, device, scored_split, args.refresh
    )

    scorer = _Scorer(records, config.fusion.candidates)
    before_report = scorer.score(fused)
    after_report = scorer.score(after)

    _headline(records, before_report, after_report, config, seconds)
    _bands(before_report, after_report)
    _languages(before_report, after_report)
    # Both rankings, because the routing signal ticket 13 needs has to come
    # from somewhere whether or not the reranker earns its place: fused scores
    # are a mean reciprocal rank, reranked ones a mean relevance, and which of
    # the two tracks correctness is a measurement rather than a preference.
    _calibration(fused, scorer, "fused")
    _calibration(after, scorer, "reranked")
    if args.sweep_mix:
        _sweep_mix(
            records, fused, config, inputs, store, device, scored_split, scorer
        )
    _movement(fused, after, config)
    return 0


def _sampled(records: Sequence[Record], size: int | None) -> list[Record]:
    """A stratified subsample of the split, or all of it.

    Stratified rather than a prefix, and this is not fastidiousness: the dev CSV
    is grouped, so its first 300 records are 300 English ones. A prefix would
    make the cross-lingual row of every table below a report on one language.

    The sampler is the index selector, because "a seeded sample stratified by
    record type and language" is a decision this project has already made once
    and should not make twice.
    """
    if not size or size >= len(records):
        return list(records)
    return indexes.select_documents(
        records, IndexConfig(size=size, stratify=True, seed=42)
    )


# --- The reranking pass, cached ---------------------------------------------


def _reranked(
    records: Sequence[Record],
    fused: Sequence[CandidateList],
    config: ExperimentConfig,
    inputs,
    store: ArtifactStore,
    device: str,
    split: str,
    refresh: bool,
    quiet: bool = False,
) -> tuple[list[CandidateList], float]:
    """The reranked lists, replaying a cached scoring pass where there is one.

    What is cached is the cross-encoder's scores for the pairs, not the ranking
    it produced. The ranking is arithmetic over those scores — `mix`,
    `mix_weight`, `mix_rrf_k`, `output_k` — and caching the arithmetic's output
    would make a sweep over a knob that reads the scores cost a forward pass per
    value. So the key normalises those knobs away, and the sweep below is free.

    Replay goes back through `reranker.rerank` behind the model interface rather
    than reconstructing a ranking here, so a cached row and a fresh one come
    from the same stage code.
    """
    texts = reranker_label_texts(
        config, inputs.vocabulary, inputs.name_qualifiers, _translations(config)
    )
    path = store.path(RERANK_STAGE, _score_key(config), f"{split}-{len(records)}.npy")

    if path.exists() and not refresh:
        if not quiet:
            print(f"scores replayed from {path}\n")
        model = _Replay(np.load(path))
        return reranker.rerank(records, fused, texts, config.reranker, model), 0.0

    model = _Recording(reranker.load(config.reranker, device))
    started = time.perf_counter()
    reranked = reranker.rerank(records, fused, texts, config.reranker, model)
    seconds = time.perf_counter() - started
    pairs = len(model.scores)
    print(
        f"reranked {pairs} pairs in {seconds:.1f}s "
        f"({pairs / max(seconds, 1e-9):.0f} pairs/s, "
        f"{seconds / max(len(records), 1):.3f}s/record)\n"
    )

    store.prepare(RERANK_STAGE, _score_key(config))
    _write(path, model.scores)
    return reranked, seconds


# What a cached scoring pass is keyed by is the pairs it scored, so every knob
# that is read *after* the scoring is normalised to one value first. `output_k`
# is one of them: it cuts a ranking, it does not change a score.
SCORED_BY_NOTHING = {
    "mix": "replace",
    "mix_weight": 1.0,
    "mix_rrf_k": 60,
    "output_k": CODES_PER_RECORD,
}


def _score_key(config: ExperimentConfig) -> ExperimentConfig:
    """The config the score cache is keyed by.

    `query`, `label_form`, `input_k`, the label-text flags and the two models'
    pinned revisions decide which pairs are scored and by what, so all of them
    are in the key; the ranking knobs are read afterwards and are normalised
    away, which is what makes the `mix_weight` sweep free.

    The revisions come through `encoders.pinned` and `reranker.pinned` rather
    than from the config as loaded, because a config file carries no revision —
    the pin lives in the registry — so a key over the sections as written would
    replay one model's scores after the other had been re-pinned.
    """
    return dataclasses.replace(
        encoders.pinned(reranker.pinned(config)),
        reranker=dataclasses.replace(
            reranker.pinned(config).reranker, **SCORED_BY_NOTHING
        ),
    )


class _Recording:
    """The real model, keeping the scores it returned so they can be cached."""

    def __init__(self, model: reranker.CrossEncoder):
        self._model = model
        self.scores: np.ndarray = np.zeros(0, dtype=np.float32)

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        self.scores = self._model.score(pairs)
        return self.scores


class _Replay:
    """A cached scoring pass behind the model interface.

    It refuses a different number of pairs rather than scoring what it can: the
    pairs are positional, so a mismatch means the cache belongs to another run
    and every score would land on the wrong candidate.
    """

    def __init__(self, scores: np.ndarray):
        self._scores = scores

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if len(pairs) != len(self._scores):
            raise ValueError(
                f"cached pass holds {len(self._scores)} scores for "
                f"{len(pairs)} pairs; rerun with --refresh"
            )
        return self._scores


def _translations(config: ExperimentConfig) -> Mapping[str, str]:
    if not config.label_text.bilingual:
        return {}
    from llms4subjects.corpus import load_label_translations

    return load_label_translations()


def _write(path: Path, scores: np.ndarray) -> None:
    # Written aside and moved into place, as the embedding cache is: a run
    # interrupted mid-write would otherwise leave a truncated file that the
    # next run reads as a hit.
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.save(handle, scores)
    partial.replace(path)


# --- Scoring ----------------------------------------------------------------


class _Scorer:
    """Fixed gold, bands and cells, so a row is one call over one ranking."""

    def __init__(self, records: Sequence[Record], ceiling: int):
        self.records = records
        self.gold = {record.id: record.subjects for record in records}
        self._cells = {record.id: (record.type, record.lang) for record in records}
        self._bands = frequency_bands()
        self.ks = TABLE_KS + (ceiling,)

    def score(self, candidates: Sequence[CandidateList]) -> EvaluationReport:
        return evaluate(
            gold=self.gold,
            predictions={result.record_id: result.codes for result in candidates},
            bands=self._bands,
            cells=self._cells,
            ks=self.ks,
        )


def achievable_precision(records: Sequence[Record], k: int) -> float:
    """What a perfect system scores at this k, on exactly these records.

    A record with 2 gold labels cannot score above 0.4 at k=5, so precision on
    this benchmark is bounded by the gold set sizes rather than by 1.0. Printed
    beside every precision figure because the leaderboard's 0.25 at k=5 reads as
    a quarter of the way there and is in fact roughly half.
    """
    if not records:
        return 0.0
    return float(
        np.mean([min(len(record.subjects), k) / k for record in records])
    )


def _headline(
    records: Sequence[Record],
    before: EvaluationReport,
    after: EvaluationReport,
    config: ExperimentConfig,
    seconds: float,
) -> None:
    gold = float(np.mean([len(record.subjects) for record in records]))
    print(f"### Reranking, before and after ({len(records)} records)\n")
    print(f"{gold:.2f} gold labels per record.\n")

    print("| | " + " | ".join(f"P@{k}" for k in TABLE_KS) + " | " +
          " | ".join(f"R@{k}" for k in TABLE_KS) + " |")
    print("|---" + "|---:" * (2 * len(TABLE_KS)) + "|")
    for name, report in (("fused", before), ("reranked", after)):
        cells = [f"{report.micro.precision(k):.4f}" for k in TABLE_KS]
        cells += [f"{report.micro.recall(k):.4f}" for k in TABLE_KS]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print(
        "| achievable | "
        + " | ".join(f"{achievable_precision(records, k):.4f}" for k in TABLE_KS)
        + " | "
        + " | ".join("1.0000" for _ in TABLE_KS)
        + " |"
    )

    print("\nas a share of achievable precision:")
    for k in (5, 10):
        ceiling = achievable_precision(records, k)
        print(
            f"  P@{k}: fused {before.micro.precision(k) / ceiling:.1%}, "
            f"reranked {after.micro.precision(k) / ceiling:.1%} of {ceiling:.4f}"
        )

    ceiling = config.fusion.candidates
    print(
        f"\nthe candidate ceiling every ranking here works inside: micro "
        f"R@{ceiling} {before.micro.recall(ceiling):.4f} — a perfect reranker "
        f"over these candidates would score that at any k"
    )

    print("\nofficial macro-over-cells, the leaderboard's aggregation:")
    for name, report in (("fused", before), ("reranked", after)):
        print(
            f"  {name}: P@5 {report.official_macro.precision(5):.4f}, "
            f"R@{SELECTION_K} {report.official_macro.recall(SELECTION_K):.4f}"
        )

    if seconds:
        print(
            f"\ncost: {seconds:.1f}s for {len(records)} records on this host, "
            f"{config.reranker.input_k} pairs each"
        )


def _bands(before: EvaluationReport, after: EvaluationReport) -> None:
    """Where the reordering lands, by how often a label was trained on."""
    print(f"\n### By frequency band (micro R@{SELECTION_K} and P@5)\n")
    print(
        "| band | assignments | fused R@10 | reranked R@10 | fused P@5 "
        "| reranked P@5 |"
    )
    print("|---" + "|---:" * 5 + "|")
    for band in BANDS:
        if band not in after.by_band:
            continue
        first, second = before.by_band[band], after.by_band[band]
        print(
            f"| {band} | {second.assignments} | "
            f"{first.recall(SELECTION_K):.4f} | {second.recall(SELECTION_K):.4f} | "
            f"{first.precision(5):.4f} | {second.precision(5):.4f} |"
        )


def _languages(before: EvaluationReport, after: EvaluationReport) -> None:
    """The cross-lingual gap, which a cross-encoder can widen or close.

    It reads the document and the label jointly, so unlike the retrievers it can
    use an English abstract against a German heading directly rather than through
    a shared vector space. Whether it does is the interesting part.
    """
    print(f"\n### By document language (micro R@{SELECTION_K} and P@5)\n")
    print(
        "| language | records | fused R@10 | reranked R@10 | fused P@5 "
        "| reranked P@5 |"
    )
    print("|---" + "|---:" * 5 + "|")
    by_records = sorted(
        after.by_language_micro,
        key=lambda name: -after.by_language_micro[name].records,
    )
    for language in by_records:
        first = before.by_language_micro[language]
        second = after.by_language_micro[language]
        print(
            f"| {language} | {second.records} | "
            f"{first.recall(SELECTION_K):.4f} | {second.recall(SELECTION_K):.4f} | "
            f"{first.precision(5):.4f} | {second.precision(5):.4f} |"
        )


# --- The confidence measure -------------------------------------------------


def _calibration(
    ranked: Sequence[CandidateList], scorer: _Scorer, what: str
) -> None:
    """Confidence against measured correctness, in equal-sized buckets.

    The measure is only a routing signal if the least-confident records are the
    ones that are actually wrong, which is a claim about ordering rather than
    about the numbers themselves — so the table is bucketed by confidence rank
    and each bucket's own precision is measured, and the correlation is reported
    beside it.
    """
    confidences = np.array(
        [reranker.confidence(result) for result in ranked], dtype=np.float64
    )
    hits = np.array(
        [
            len(set(result.codes[:5]) & set(scorer.gold[result.record_id]))
            for result in ranked
        ],
        dtype=np.float64,
    )
    gold_sizes = np.array(
        [len(scorer.gold[result.record_id]) for result in ranked], dtype=np.float64
    )
    precision = hits / 5.0
    recall = np.divide(
        hits, gold_sizes, out=np.zeros_like(hits), where=gold_sizes > 0
    )

    print(
        f"\n### The confidence measure over the {what} ranking, calibrated "
        f"(top-{reranker.CONFIDENCE_K} mean score)\n"
    )
    print("| bucket | records | mean confidence | P@5 | R@5 | records with no hit |")
    print("|---" + "|---:" * 5 + "|")

    order = np.argsort(-confidences, kind="stable")
    for index, bucket in enumerate(np.array_split(order, BUCKETS)):
        if not len(bucket):
            continue
        print(
            f"| {index + 1} | {len(bucket)} | {confidences[bucket].mean():.4f} | "
            f"{precision[bucket].mean():.4f} | {recall[bucket].mean():.4f} | "
            f"{float((hits[bucket] == 0).mean()):.1%} |"
        )

    if confidences.std() and precision.std():
        correlation = f"{float(np.corrcoef(confidences, precision)[0, 1]):.4f}"
    else:
        # One of the two is constant, which happens only on a handful of
        # records; saying so beats printing the nan numpy would produce.
        correlation = "undefined (no variance in one of the two)"
    print(f"\nPearson correlation of confidence with per-record P@5: {correlation}")
    least = order[-len(order) // 5:]
    print(
        f"the least-confident 20% ({len(least)} records) score P@5 "
        f"{precision[least].mean():.4f} against {precision.mean():.4f} overall — "
        f"which is what ticket 13 would route"
    )


# What `--sweep-mix` tries. Coarse, and for the reason the fusion weight grid is:
# the right weight is a property of one reranker against one retrieval stack,
# and a finer grid would be fitting 300 records of dev noise.
MIX_WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _sweep_mix(
    records: Sequence[Record],
    fused: Sequence[CandidateList],
    config: ExperimentConfig,
    inputs,
    store: ArtifactStore,
    device: str,
    split: str,
    scorer: "_Scorer",
) -> None:
    """`mix_weight` against the metrics, over the cached scoring pass.

    Every row is the real stage over the same scores, so the sweep is arithmetic
    and costs seconds. Weight 0 is the fused ranking reproduced through the
    reranking path, which is what makes the other rows attributable.
    """
    print(f"\n### Tuning `mix_weight` (micro, selection on R@{SELECTION_K})\n")
    print(f"| mix_weight | P@5 | P@10 | R@{SELECTION_K} | R@50 |")
    print("|---" + "|---:" * 4 + "|")

    rows = []
    for weight in MIX_WEIGHT_GRID:
        weighted = dataclasses.replace(
            config,
            reranker=dataclasses.replace(
                config.reranker, mix="fuse", mix_weight=weight
            ),
        )
        ranked, _ = _reranked(
            records, fused, weighted, inputs, store, device, split, False, quiet=True
        )
        report = scorer.score(ranked)
        rows.append((weight, report))
        print(
            f"| {weight} | {report.micro.precision(5):.4f} | "
            f"{report.micro.precision(10):.4f} | "
            f"{report.micro.recall(SELECTION_K):.4f} | "
            f"{report.micro.recall(50):.4f} |"
        )

    best, report = max(rows, key=lambda row: row[1].micro.recall(SELECTION_K))
    print(
        f"\nbest on micro R@{SELECTION_K}: mix_weight {best} at "
        f"{report.micro.recall(SELECTION_K):.4f}\n"
    )
    print("the tuned block, for the config:\n")
    print(f"reranker:\n  mix: fuse\n  mix_weight: {best}")


def _movement(
    fused: Sequence[CandidateList],
    reranked: Sequence[CandidateList],
    config: ExperimentConfig,
) -> None:
    """How much the reranker actually moved, so a null result is visible as one.

    A cross-encoder that agrees with fusion everywhere would show up in the
    metrics as no change and in a code review as a bug; these two numbers tell
    those apart.
    """
    promoted = []
    kept = []
    for before, after in zip(fused, reranked):
        positions = {code: rank for rank, code in enumerate(before.codes)}
        top = after.codes[:5]
        promoted.extend(positions[code] for code in top if code in positions)
        kept.append(len(set(top) & set(before.codes[:5])))

    print("\n### What the reranker moved\n")
    if not promoted:
        print("nothing: no record had a candidate to move")
        return
    print(
        f"mean pre-rerank rank of a final top-5 code: {np.mean(promoted):.1f} "
        f"(of {config.reranker.input_k} candidates)"
    )
    print(
        f"of the 5 codes fusion put first, {np.mean(kept):.2f} are still in the "
        "reranked top 5"
    )


if __name__ == "__main__":
    raise SystemExit(main())
