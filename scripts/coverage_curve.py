"""Coverage against precision on dev, for the workflow the metric cannot score.

    python scripts/coverage_curve.py configs/rung1.yaml
    python scripts/coverage_curve.py configs/rung1.yaml --figure artifacts/coverage.png
    python scripts/coverage_curve.py configs/rung1-rerank-base.yaml --sample 300

**Outside the official metric, and it cannot be otherwise.** The submission
format is exactly 50 ranked codes per record and has no representation for
fewer, so a system on this benchmark cannot abstain and the leaderboard cannot
reward abstaining. But the workflow the project exists to support is
suggest-and-confirm — a librarian confirming proposals — and there a system that
declines the records it is worst at is more useful than one that guesses
everywhere. This harness is the one place that trade is measured, every row it
prints is marked, and nothing it prints is comparable to a published number.

The curve is arithmetic over scores already computed: the confidence measure
`stages.reranker.confidence` orders the records, and at each coverage level the
retained subset is scored by the same evaluator every other table in this
project goes through. No model is loaded to produce it. A reranking config is
read back from the `reranked` score cache and **refused** if that pass is not on
disk, rather than quietly spending the hours it would cost — see ticket 12 for
what a dev pass costs and `scripts/rerank_report.py` for the command that
produces one.

Nothing here changes the output contract. Abstention is a reading of the
confidence column, never a shortened ranking: every record still carries its 50
codes, which is what `tests/test_coverage_curve.py` holds.

Implemented by ticket 16.
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms4subjects.artifacts import ArtifactStore  # noqa: E402
from llms4subjects.config import ExperimentConfig, IndexConfig, load_experiment  # noqa: E402
from llms4subjects.contracts import CandidateList, Code, Record  # noqa: E402
from llms4subjects.corpus import MissingDataset, frequency_bands  # noqa: E402
from llms4subjects.hardware import describe_device, select_device  # noqa: E402
from llms4subjects.pipeline import combine, reranker_label_texts, retrieve  # noqa: E402
from llms4subjects.stages import indexes, reranker  # noqa: E402
from llms4subjects.stages.evaluator import evaluate  # noqa: E402
from rerank_report import Replay, score_cache_path  # noqa: E402
from run_experiment import FORBIDDEN_SPLIT, load_inputs  # noqa: E402

# The marker every rendering of this figure carries. It is an acceptance
# criterion of ticket 16 rather than a courtesy: a coverage-restricted precision
# is higher than the same system's official precision by construction, so a
# figure of this shape that travelled without the marker would read as a
# leaderboard-comparable improvement.
OUTSIDE = "outside the official metric"

# Precision is read at 5 and recall at 10, as everywhere else in this project:
# P@5 is what a suggest-and-confirm screen shows, R@10 is the model-selection
# signal. The confidence measure is a mean over the top 5 for the same reason.
PRECISION_K = 5
RECALL_K = 10

# The coverage levels the table reports, full coverage first so every other row
# has a baseline beside it. Ten points at 5,354 dev records puts at least 535
# records under the tightest row, which is enough for its precision not to be
# noise.
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


class NoCachedPass(FileNotFoundError):
    """A reranking config whose scoring pass is not on disk.

    Raised rather than falling back to a forward pass: this harness produces a
    figure from scores already computed, and a silent 3.9-hour reranking run is
    the opposite of that.
    """


@dataclass(frozen=True)
class Point:
    """One coverage level: what was answered, and how well.

    `requested` and `coverage` are both kept because they can differ, and the
    difference is the honest part. Records tied at the threshold are answered or
    declined together — a system cannot tell them apart, so a curve that split
    them to hit a round number would be reporting a threshold it never used.
    """

    requested: float
    coverage: float
    records: int
    threshold: float
    precision: float
    achievable: float
    recall: float
    assignments: int
    assignment_share: float
    no_hit: float

    @property
    def share_of_achievable(self) -> float:
        """Precision against the maximum the retained gold set sizes allow.

        At 2.40 gold labels a record a perfect system scores 0.48 at k=5, and
        the ceiling moves with the subset, so the raw figure alone is a
        misreading waiting to happen.
        """
        return self.precision / self.achievable if self.achievable else 0.0


def curve(
    records: Sequence[Record],
    ranked: Sequence[CandidateList],
    bands: Mapping[Code, str],
    coverages: Sequence[float] = COVERAGES,
    precision_k: int = PRECISION_K,
    recall_k: int = RECALL_K,
    confidence: Callable[[CandidateList], float] = reranker.confidence,
) -> tuple[Point, ...]:
    """One `Point` per requested coverage, most permissive first.

    `records` carry the gold and the scoring cell; `ranked` the rankings, matched
    by record id rather than by position, because a positional match that slipped
    would score one record's ranking against another's gold and look like a
    result. Every id must be present on both sides.

    The metric is the project's evaluator over the retained subset, not a second
    implementation of precision: a curve whose arithmetic drifted from the
    results tables would be comparing two different systems.

    A record with no gold labels is dropped before anything is counted, as the
    evaluator drops it: recall would have no denominator, and keeping it in the
    coverage column while the precision column could not see it would make every
    row's two halves disagree about how many records they describe.
    """
    scorable = [record for record in records if record.subjects]
    if not scorable:
        return ()
    rankings = _aligned(scorable, ranked)
    records = scorable

    confidences = {
        record_id: confidence(result) for record_id, result in rankings.items()
    }
    # Ties are resolved by record id so that the ordering is a property of the
    # data rather than of the order a dict happened to be in; which record of a
    # tie sits at position n never decides who is retained, because the
    # threshold is a score and every record holding it is kept.
    ordered = sorted(confidences, key=lambda record_id: (-confidences[record_id], record_id))

    gold = {record.id: record.subjects for record in records}
    cells = {record.id: (record.type, record.lang) for record in records}
    assignments = sum(len(codes) for codes in gold.values())
    total = len(ordered)

    points = []
    for requested in coverages:
        # Rounded up: a coverage level that asked for half of an odd split
        # answers the extra record rather than declining it, so no row claims a
        # tighter threshold than it took.
        wanted = min(total, max(1, math.ceil(requested * total)))
        threshold = confidences[ordered[wanted - 1]]
        retained = [
            record_id
            for record_id in ordered
            if confidences[record_id] >= threshold
        ]
        points.append(
            _point(
                requested,
                threshold,
                retained,
                rankings,
                gold,
                cells,
                bands,
                assignments,
                total,
                precision_k,
                recall_k,
            )
        )
    return tuple(points)


def _point(
    requested: float,
    threshold: float,
    retained: Sequence[str],
    rankings: Mapping[str, CandidateList],
    gold: Mapping[str, Sequence[Code]],
    cells: Mapping[str, tuple[str, str]],
    bands: Mapping[Code, str],
    assignments: int,
    total: int,
    precision_k: int,
    recall_k: int,
) -> Point:
    """One row: the retained subset scored, with what declining it gave up."""
    subset = {record_id: gold[record_id] for record_id in retained}
    report = evaluate(
        gold=subset,
        predictions={
            record_id: rankings[record_id].codes for record_id in retained
        },
        bands=bands,
        cells={record_id: cells[record_id] for record_id in retained},
        ks=tuple(dict.fromkeys((precision_k, recall_k))),
    )
    # Every retained record is scored: `curve` drops the gold-less ones before
    # any of this, so the coverage column and the precision column count the
    # same records.
    held = sum(len(subset[record_id]) for record_id in retained)
    misses = sum(
        1
        for record_id in retained
        if not set(rankings[record_id].codes[:precision_k]) & set(subset[record_id])
    )
    return Point(
        requested=requested,
        coverage=len(retained) / total if total else 0.0,
        records=len(retained),
        threshold=threshold,
        precision=report.micro.precision(precision_k),
        achievable=achievable(
            [len(subset[record_id]) for record_id in retained], precision_k
        ),
        recall=report.micro.recall(recall_k),
        assignments=held,
        assignment_share=held / assignments if assignments else 0.0,
        no_hit=misses / len(retained) if retained else 0.0,
    )


def achievable(gold_sizes: Sequence[int], k: int) -> float:
    """What a perfect system scores at this k on exactly these records.

    `scripts/rerank_report.py` computes this over `Record` objects; the curve
    holds gold set sizes for a subset it selected by id, and the ceiling has to
    be recomputed per row because it moves with the subset — records the system
    is confident about are not records with an average number of gold labels.
    """
    if not gold_sizes:
        return 0.0
    return float(np.mean([min(size, k) / k for size in gold_sizes]))


def _aligned(
    records: Sequence[Record], ranked: Sequence[CandidateList]
) -> dict[str, CandidateList]:
    """The ranking for each record, refusing anything that cannot be one.

    Duplicate ids are refused rather than collapsed: `all_train` holds 45 ids
    that appear twice — the same work filed twice — so a curve that keyed by id
    would quietly score one filing's ranking against the other's gold and report
    a coverage denominator it never used.
    """
    seen = {record.id for record in records}
    if len(seen) != len(records):
        duplicated = sorted(
            record.id
            for record in records
            if [other.id for other in records].count(record.id) > 1
        )
        raise ValueError(
            f"{len(records) - len(seen)} record ids appear more than once "
            f"(first: {duplicated[0]!r}); the curve is keyed by record id, so a "
            "repeated id would score one record's ranking against another's gold"
        )
    rankings = {result.record_id: result for result in ranked}
    missing = [record.id for record in records if record.id not in rankings]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(records)} records have no ranking "
            f"(first: {missing[0]!r}); the curve is a slice of one pass over "
            "one record set, not a join of two"
        )
    return {record.id: rankings[record.id] for record in records}


# --- Rendering --------------------------------------------------------------


def figure_title(read_from: str) -> str:
    return f"Coverage against precision on dev — {OUTSIDE} ({read_from})"


def render(
    points: Sequence[Point],
    read_from: str,
    precision_k: int = PRECISION_K,
    recall_k: int = RECALL_K,
) -> str:
    """The table, as markdown, with the marker in its heading.

    `read_from` names the ranking the confidence came from, because ticket 12
    measured the three available rankings to calibrate very differently and a
    curve that did not say which one it read would not be reproducible.
    """
    lines = [
        f"### {figure_title(read_from)}",
        "",
        f"| coverage asked | coverage taken | records | confidence >= | "
        f"P@{precision_k} | of achievable | R@{recall_k} on answered | "
        f"gold answered | no hit in {precision_k} |",
        "|---" + "|---:" * 8 + "|",
    ]
    for point in points:
        lines.append(
            f"| {point.requested:.0%} | {point.coverage:.1%} | {point.records} | "
            f"{point.threshold:.4f} | {point.precision:.4f} | "
            f"{point.share_of_achievable:.1%} | {point.recall:.4f} | "
            f"{point.assignment_share:.1%} | {point.no_hit:.1%} |"
        )
    return "\n".join(lines)


def figure(
    points: Sequence[Point],
    path: Path,
    read_from: str,
    precision_k: int = PRECISION_K,
) -> Path:
    """The curve as a PNG: precision against coverage, with its ceiling.

    Written where it was asked for and nowhere by default. `matplotlib` is
    imported here rather than at module scope so that the table — which is the
    deliverable — is produced on a host that has no plotting stack.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coverages = [point.coverage * 100 for point in points]
    figure_, axes = plt.subplots(figsize=(7.5, 4.5))
    axes.plot(
        coverages,
        [point.precision for point in points],
        marker="o",
        label=f"P@{precision_k}",
    )
    axes.plot(
        coverages,
        [point.achievable for point in points],
        linestyle="--",
        color="grey",
        label=f"achievable P@{precision_k}",
    )
    axes.set_xlabel("coverage: share of dev records still answered (%)")
    axes.set_ylabel(f"precision at {precision_k} on the answered records")
    # Wrapped rather than shrunk: the marker is part of the title, and a
    # title that runs off the canvas is a figure that travels without it.
    axes.set_title("\n".join(textwrap.wrap(figure_title(read_from), 74)), fontsize=9)
    axes.invert_xaxis()
    axes.set_ylim(bottom=0.0)
    axes.grid(alpha=0.3)
    # Bottom left: precision rises to the right and its ceiling runs along
    # the top, so every other corner has a line in it.
    axes.legend(loc="lower left", fontsize=8)
    figure_.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure_.savefig(path, dpi=150)
    plt.close(figure_)
    return path


# --- The ranking the curve reads --------------------------------------------


def cached_scores(
    config: ExperimentConfig,
    store: ArtifactStore,
    split: str,
    records: int,
    command: str = "python scripts/rerank_report.py <config>",
) -> np.ndarray:
    """The reranker's scores for this configuration, or a refusal naming them.

    The key is `scripts/rerank_report.py`'s, so the pass this reads is the pass
    that harness wrote — the same scores the calibration table and the band
    breakdown were taken from, rather than a second run of the same model.
    """
    path = score_cache_path(config, store, split, records)
    if not path.exists():
        raise NoCachedPass(
            f"no cached reranking pass at {path}. This harness produces a "
            "figure from scores already computed and will not spend a "
            "cross-encoder pass to make one; run\n\n"
            f"    {command}\n\n"
            "first — the sample and the config have to match, because the cache "
            "is keyed by both — or read the curve off the fused ranking with a "
            "config that has reranker.enabled false."
        )
    return np.load(path)


def _sampled(records: Sequence[Record], size: int | None) -> list[Record]:
    """A stratified subsample of the split, or all of it.

    The sampler is the index selector, as in `scripts/rerank_report.py`, and for
    the same reason: the dev CSV is grouped, so a prefix would draw one language
    and the curve would be a figure about English records.
    """
    if not size or size >= len(records):
        return list(records)
    return indexes.select_documents(
        records, IndexConfig(size=size, stratify=True, seed=42)
    )


def _translations(config: ExperimentConfig) -> Mapping[str, str]:
    if not config.label_text.bilingual:
        return {}
    from llms4subjects.corpus import load_label_translations

    return load_label_translations()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="the experiment whose ranking is read")
    parser.add_argument("--split", default="core_dev", help="the split to curve")
    parser.add_argument(
        "--sample",
        type=int,
        help="curve a stratified sample of N records rather than the whole split",
    )
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    parser.add_argument(
        "--figure",
        help="also write the curve as a PNG here; the table is printed either way",
    )
    args = parser.parse_args(argv)

    if args.split == FORBIDDEN_SPLIT:
        # Dev-only, and this is the harness where that matters most: a curve is
        # a sweep over thresholds, so running it on the gold test split would be
        # ten peeks rather than one. See ticket 17.
        print(
            f"{FORBIDDEN_SPLIT} is the gold test split; the coverage curve is "
            "dev-only (ticket 16) and the test split is opened once (ticket 17)."
        )
        return 1

    config = load_experiment(args.config)

    if config.adjudication.enabled:
        # The adjudicated ranking is a reordering by a language model, and this
        # harness spends nothing: reproducing it would need the API, and
        # skipping it would print a curve under the name of a pipeline that
        # never ran.
        print(
            f"{args.config} enables adjudication, which this harness does not "
            "apply: it fuses, optionally replays a cached reranking pass, and "
            "curves the result. Use the config with adjudication off."
        )
        return 1
    if config.group_prior.enabled:
        print(
            f"{args.config} enables the group prior, which this harness does "
            "not apply. Use the config with the prior off, or "
            "scripts/ablate_group_prior.py for that stage."
        )
        return 1
    if not [name for name, settings in config.retrievers.items() if settings.enabled]:
        print(
            f"{args.config} enables no retriever, so there is nothing to be "
            "confident about; the all-off ablation is a row in the retriever "
            "table."
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
    store = ArtifactStore(args.artifacts, data_revision=inputs.revision)

    print(f"experiment:    {config.name}  ({args.config})")
    print(f"data revision: {inputs.revision}")
    print(describe_device(device))
    print(f"curving {len(records)} {args.split} records\n")

    # The cached scoring pass is looked for before anything is retrieved: the
    # check needs only the config, the split and the record count, and a
    # refusal that arrived after retrieval would have spent minutes of the
    # encoder to say that this harness spends nothing.
    scores = None
    if config.reranker.enabled:
        scored_split = f"{args.split}-sample" if args.sample else args.split
        command = f"python scripts/rerank_report.py {args.config}" + (
            f" --sample {args.sample}" if args.sample else ""
        )
        try:
            scores = cached_scores(
                config, store, scored_split, len(records), command
            )
        except NoCachedPass as error:
            print(error)
            return 1

    fused = combine(
        retrieve(
            records,
            config,
            inputs.vocabulary,
            inputs.index_records,
            store,
            device=device,
            name_qualifiers=inputs.name_qualifiers,
        ),
        config,
    )

    ranking, read_from = fused, "the fused ranking"
    if scores is not None:
        ranking = reranker.rerank(
            records,
            fused,
            reranker_label_texts(
                config, inputs.vocabulary, inputs.name_qualifiers, _translations(config)
            ),
            config.reranker,
            Replay(scores),
        )
        read_from = (
            "the reranked ranking, fused with the retrieval order"
            if config.reranker.mix == "fuse"
            else "the reranker's own relevance, replacing the fused order"
        )
        print("the reranked ranking was replayed from the cached scoring pass\n")

    points = curve(records, ranking, frequency_bands())
    print(render(points, read_from))
    print(
        f"\nThe confidence measure is `reranker.confidence` over {read_from}: "
        f"the mean score of a record's top {reranker.CONFIDENCE_K}. Ticket 12 "
        "measured what each available ranking's confidence is worth as a "
        "signal (docs/results.md); this figure inherits that.\n"
    )
    print(
        f"Every row is {OUTSIDE}. The submission format is 50 codes with no "
        "representation for abstention, so no row here is comparable to a "
        "leaderboard figure, and the system's output contract is unchanged: "
        "50 codes are still always returned."
    )

    if args.figure:
        print(f"\nfigure written to {figure(points, Path(args.figure), read_from)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
