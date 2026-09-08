"""Metrics from gold and predicted label lists. A pure function, by design.

Reports both aggregations the project needs — record-level micro averages for
model selection, and the official macro-over-cells figure for comparability
with the published leaderboard — and slices every metric by frozen frequency
band, document language and record type.

Bands are read from the committed frozen artifact and never recomputed, so
growing the index cannot silently reclassify which labels count as tail.

## The two aggregations, and why both

The organizers' script averages precision and recall over the records of each
`<record type> × language` cell, then averages those cell figures with equal
weight to reach the `Overall` row it publishes. That makes a cell holding one
record worth as much as a cell holding two thousand, which is fine for a
leaderboard and useless for model selection: one record can flip which encoder
appears to win. So the micro figure — every gold assignment weighted once — is
what this project iterates on, and the official figure is what it reports.

Their arithmetic is reproduced here exactly, including two details that a
reimplementation from the metric definitions would get wrong:

- per-record precision and recall are rounded to four decimals *before* being
  averaged, and only the per-cell F1 is rounded again afterwards;
- the `Overall` F1 is the mean of the cells' F1 values, not the F1 of the mean
  precision and recall. The two differ, and the published number is the former.

Everything that mirrors the official path therefore goes through
`_official_average`; the micro path never rounds.

## What the official script cannot read

Its reader seeds its results with `de` and `en` and then indexes into them by
directory name, so a record in any other language raises a `KeyError` inside
their code. The test set contains French, Spanish, Czech, Turkish, Dutch and
Japanese records. Those cells are scored here and are absent from their sheet,
which is one more reason the headline needs the micro figure beside it.

Implemented by ticket 03.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..contracts import Cell, Code, cell_of

BANDS = ("head", "torso", "tail", "zero")

# The band a label falls in when the frozen reference does not name it. Kept in
# step with `corpus.UNSEEN_BAND`, which is where the reference is read; a stage
# may not read the dataset itself, so the constant is repeated rather than
# imported across that boundary.
UNSEEN_BAND = "zero"

OFFICIAL_KS = tuple(range(5, 55, 5))

METRIC_NAMES = ("precision", "recall", "f1")

# The official script rounds every per-record precision and recall to this many
# decimals before averaging. Reproducing the rounding is what makes the local
# evaluator agree with it rather than merely resemble it.
OFFICIAL_DECIMALS = 4


@dataclass(frozen=True)
class Metrics:
    """Precision, recall and F1 at each k, at one aggregation and slice.

    `records` and `assignments` are the slice's support: a band recall of 0.31
    means something different over 8.9% of gold assignments than over 44.5%,
    and the breakdown is the deliverable, so the denominator travels with it.
    """

    at_k: Mapping[int, Mapping[str, float]]
    records: int = 0
    assignments: int = 0

    def precision(self, k: int) -> float:
        return self.at_k[k]["precision"]

    def recall(self, k: int) -> float:
        return self.at_k[k]["recall"]

    def f1(self, k: int) -> float:
        return self.at_k[k]["f1"]


@dataclass(frozen=True)
class CellContribution:
    """How much of the official recall figure one scoring cell accounts for.

    `weight` is what the official aggregation gives the cell — one vote, whatever
    its size — and `record_share` is what a micro average would give it. The
    gap between those two numbers is the whole story of the divergence.

    `recall_share` is deliberately about recall alone rather than all three
    metrics: it is the quantity docs/spec.md commits to reporting ("eleven cells
    holding 0.4% of test records carry 55% of the official `Overall`"), and
    recall is the figure the leaderboard is read on.
    """

    cell: Cell
    records: int
    record_share: float
    weight: float
    recall_share: Mapping[int, float]

    @property
    def record_type(self) -> str:
        return self.cell.record_type

    @property
    def language(self) -> str:
        return self.cell.language

    @property
    def mean_recall_share(self) -> float:
        return sum(self.recall_share.values()) / len(self.recall_share)


@dataclass(frozen=True)
class Divergence:
    """The gap between the micro and official-macro aggregations, as a result.

    Not diagnostics: docs/spec.md commits to reporting this, because eleven
    cells holding 0.4% of test records carry 55% of the official `Overall`, and
    a headline that does not say so is misleading by omission.
    """

    delta: Mapping[int, Mapping[str, float]]
    cells: tuple[CellContribution, ...] = ()

    def dominant(self, share: float = 0.5) -> tuple[CellContribution, ...]:
        """The fewest cells, largest first, accounting for `share` of the figure."""
        taken: list[CellContribution] = []
        total = 0.0
        for cell in self.cells:
            taken.append(cell)
            total += cell.mean_recall_share
            if total >= share:
                break
        return tuple(taken)


@dataclass(frozen=True)
class EvaluationReport:
    micro: Metrics
    official_macro: Metrics
    by_band: Mapping[str, Metrics]
    by_language: Mapping[str, Metrics]
    by_type: Mapping[str, Metrics]
    # The same languages micro-averaged. Both are reported because they answer
    # different questions and can disagree sharply: the official figure gives a
    # three-record cell the same vote as a 1,522-record one, so a retriever
    # whose behaviour genuinely differs by language — the lexical one, whose
    # whole premise is that German headings appear verbatim and English ones do
    # not — needs the assignment-weighted figure for the size of that gap.
    by_language_micro: Mapping[str, Metrics] = field(default_factory=dict)
    by_cell: Mapping[Cell, Metrics] = field(default_factory=dict)
    divergence: Divergence = field(default_factory=lambda: Divergence({}))
    ks: tuple[int, ...] = OFFICIAL_KS
    scored: int = 0
    # Records the official reader would not have scored either: their gold set
    # is empty, so recall has no denominator. Named rather than counted, because
    # a split that suddenly grows some is a dataset bug, not a metric.
    without_gold: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Scored:
    """One record's gold set, ranking and cell, resolved once."""

    record_id: str
    gold: frozenset[Code]
    ranking: tuple[Code, ...]
    cell: Cell

    def hits(self, k: int, gold: frozenset[Code] | None = None) -> int:
        # Sets, matching the official scorer: a repeated code in the ranking
        # neither helps nor is penalised.
        return len((self.gold if gold is None else gold) & set(self.ranking[:k]))


def evaluate(
    gold: Mapping[str, Sequence[Code]],
    predictions: Mapping[str, Sequence[Code]],
    bands: Mapping[Code, str],
    cells: Mapping[str, tuple[str, str]],
    ks: Sequence[int] = OFFICIAL_KS,
) -> EvaluationReport:
    """Score predictions against gold. `cells` maps record id to (type, lang).

    A record with no prediction scores zero rather than being skipped, which is
    what the official script does after filling missing files with empty lists.
    A record with no gold labels is excluded and named in `without_gold`: recall
    would have no denominator, and their reader drops such records too.

    Raises `KeyError` for a scored record missing from `cells`, because guessing
    a cell would silently move a record between the groups being compared.
    """
    ks = tuple(dict.fromkeys(int(k) for k in ks))

    scored: list[_Scored] = []
    without_gold: list[str] = []
    for record_id, codes in gold.items():
        gold_set = frozenset(codes)
        if not gold_set:
            without_gold.append(record_id)
            continue
        scored.append(
            _Scored(
                record_id=record_id,
                gold=gold_set,
                ranking=tuple(predictions.get(record_id, ())),
                cell=cell_of(cells, record_id),
            )
        )

    by_cell = {
        cell: _official_average(group, ks)
        for cell, group in _group(scored, lambda record: record.cell).items()
    }
    micro = _micro(scored, ks)
    official_macro = _macro_over_cells(by_cell, ks)

    return EvaluationReport(
        micro=micro,
        official_macro=official_macro,
        by_band=_by_band(scored, bands, ks),
        by_language={
            language: _official_average(group, ks)
            for language, group in _group(scored, lambda r: r.cell.language).items()
        },
        by_language_micro={
            language: _micro(group, ks)
            for language, group in _group(scored, lambda r: r.cell.language).items()
        },
        by_type={
            record_type: _official_average(group, ks)
            for record_type, group in _group(scored, lambda r: r.cell.record_type).items()
        },
        by_cell=by_cell,
        divergence=_divergence(micro, official_macro, by_cell, len(scored), ks),
        ks=ks,
        scored=len(scored),
        without_gold=tuple(without_gold),
    )


def render(
    report: EvaluationReport,
    ks: Sequence[int] | None = None,
    slice_metric: str = "recall",
) -> str:
    """The report as text: the two aggregations side by side, then the slices.

    The headline block carries all three metrics, because the leaderboard this
    project is measured against is quoted on precision and recall together. The
    slice blocks carry one metric — recall unless asked otherwise — because a
    four-band table times three metrics times ten values of k is a wall of
    digits nobody reads.

    `ks` narrows the columns; the default ten fit a wide terminal and not much
    else. Deliberately plain: it exists so a rung's result can be appended to
    the results document as it completes rather than reconstructed from memory.
    """
    ks = tuple(ks) if ks is not None else report.ks
    if slice_metric not in METRIC_NAMES:
        raise ValueError(f"slice_metric must be one of {METRIC_NAMES}, got {slice_metric!r}")

    excluded = (
        f"  (excluded, no gold: {len(report.without_gold)})"
        if report.without_gold
        else ""
    )
    lines = [f"scored records: {report.scored}{excluded}"]


    delta = report.divergence.delta
    for metric in METRIC_NAMES:
        lines += ["", f"{metric:<{_LABEL}} " + _header(metric, ks)]
        lines.append(_row("  micro", report.micro, metric, ks))
        lines.append(_row("  official-macro", report.official_macro, metric, ks))
        if delta:
            lines.append(
                f"{'  divergence':<{_LABEL}} "
                + " ".join(f"{delta[k][metric]:+8.4f}" for k in ks)
            )

    # Each slice is labelled with its aggregation, because a band row and a
    # language row are not averaged the same way and the difference is not
    # recoverable from the numbers.
    for title, group in (
        ("band [micro]", report.by_band),
        ("language [micro]", report.by_language_micro),
        ("language [off.]", report.by_language),
        ("type [off.]", report.by_type),
    ):
        lines += ["", f"{title:<{_LABEL}} " + _header(slice_metric, ks)]
        lines += [
            _row(f"  {name} ({metrics.assignments})", metrics, slice_metric, ks)
            for name, metrics in group.items()
        ]

    if report.divergence.cells:
        dominant = report.divergence.dominant()
        lines += [
            "",
            f"{len(dominant)} of {len(report.divergence.cells)} cells carry "
            f"{sum(cell.mean_recall_share for cell in dominant):.1%} of the "
            "official recall figure:",
        ]
        lines += [
            f"  {cell.record_type:<12} {cell.language:<3} "
            f"{cell.records:>6} records  {cell.record_share:>7.2%} of records  "
            f"{cell.mean_recall_share:>7.2%} of the figure"
            for cell in dominant
        ]

    return "\n".join(lines)


# Wide enough for the longest slice label ("  Conference (312)") to keep every
# column aligned; a ragged table is read as a different table.
_LABEL = 20


def _header(metric: str, ks: Sequence[int]) -> str:
    initial = metric[0].upper()
    return " ".join(f"{initial + '@' + str(k):>8}" for k in ks)


def _row(label: str, metrics: Metrics, metric: str, ks: Sequence[int]) -> str:
    return f"{label:<{_LABEL}} " + " ".join(
        f"{metrics.at_k[k][metric]:8.4f}" for k in ks
    )


def _group(records: Iterable[_Scored], key) -> dict:
    """Records grouped by `key`, in order of first appearance."""
    groups: dict = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    return groups


def _micro(
    records: Sequence[_Scored],
    ks: Sequence[int],
    gold: Mapping[str, frozenset[Code]] | None = None,
) -> Metrics:
    """Every gold assignment weighted once: the model-selection signal.

    `gold` restricts each record's gold set, which is how a band slice is taken.
    The restriction changes the numerator but never the record set: every scored
    record stays in the precision denominator, so the band precisions sum back
    to the overall precision, and recall is unaffected either way because a
    record with no gold in the band contributes nothing to either side of it.
    """
    restricted = [
        (record, record.gold if gold is None else gold[record.record_id])
        for record in records
    ]
    assignments = sum(len(codes) for _, codes in restricted)

    at_k = {}
    for k in ks:
        hits = sum(record.hits(k, codes) for record, codes in restricted)
        precision = hits / (len(records) * k) if records else 0.0
        recall = hits / assignments if assignments else 0.0
        at_k[k] = _scores(precision, recall)
    return Metrics(
        at_k=at_k,
        # The records this slice actually says something about, which for a band
        # is the records holding at least one of its labels.
        records=sum(1 for _, codes in restricted if codes),
        assignments=assignments,
    )


def _official_average(records: Sequence[_Scored], ks: Sequence[int]) -> Metrics:
    """The official arithmetic for one group: round per record, then average.

    Used for every slice that has an official counterpart — cell, record type
    and language — so those numbers are comparable to the published sheets
    rather than merely similar to them.
    """
    at_k = {}
    for k in ks:
        precision = 0.0
        recall = 0.0
        for record in records:
            hits = record.hits(k)
            precision += round(hits / k, OFFICIAL_DECIMALS)
            recall += round(hits / len(record.gold), OFFICIAL_DECIMALS)
        count = len(records)
        precision = precision / count if count else 0.0
        recall = recall / count if count else 0.0
        at_k[k] = _scores(precision, recall, decimals=OFFICIAL_DECIMALS)
    return Metrics(
        at_k=at_k,
        records=len(records),
        assignments=sum(len(record.gold) for record in records),
    )


def _macro_over_cells(by_cell: Mapping[Cell, Metrics], ks: Sequence[int]) -> Metrics:
    """The published `Overall` row: one vote per cell, F1 averaged as a column.

    Averaging the cells' F1 values is not the F1 of the averaged precision and
    recall, and the official spreadsheet does the former by taking the mean of
    every metric column. This reproduces that rather than correcting it.
    """
    at_k = {}
    for k in ks:
        at_k[k] = {
            name: (
                sum(metrics.at_k[k][name] for metrics in by_cell.values())
                / len(by_cell)
                if by_cell
                else 0.0
            )
            for name in METRIC_NAMES
        }
    return Metrics(
        at_k=at_k,
        records=sum(metrics.records for metrics in by_cell.values()),
        assignments=sum(metrics.assignments for metrics in by_cell.values()),
    )


def _by_band(
    records: Sequence[_Scored], bands: Mapping[Code, str], ks: Sequence[int]
) -> dict[str, Metrics]:
    """One micro slice per band, over that band's gold assignments only.

    Recall is the number to read: of the gold labels in this band, how many did
    the ranking reach. Precision is the share of the k slots across the whole
    split that landed on a band-b gold label, so the four band precisions sum
    to the overall micro precision — a decomposition rather than four separate
    precisions, which is the only reading that stays additive.

    Micro is the right aggregation here and the only one offered. The official
    macro-over-cells figure exists to be comparable to the published table, and
    the organizers publish no band breakdown, so a macro band number would be
    comparable to nothing while inheriting the small-cell distortion this band
    breakdown is meant to see through.

    A band that holds nothing still appears, with a zero support, because a band
    silently missing from a results table reads as "not measured" rather than
    "measured, and empty".
    """
    restricted = {
        band: {
            record.record_id: frozenset(
                code
                for code in record.gold
                if bands.get(code, UNSEEN_BAND) == band
            )
            for record in records
        }
        for band in BANDS
    }
    return {band: _micro(records, ks, restricted[band]) for band in BANDS}


def _divergence(
    micro: Metrics,
    official_macro: Metrics,
    by_cell: Mapping[Cell, Metrics],
    scored: int,
    ks: Sequence[int],
) -> Divergence:
    delta = {
        k: {
            name: official_macro.at_k[k][name] - micro.at_k[k][name]
            for name in METRIC_NAMES
        }
        for k in ks
    }

    # Each cell gets the same weight in the official figure, so its share of
    # that figure is its recall over the sum of every cell's recall.
    totals = {
        k: sum(metrics.at_k[k]["recall"] for metrics in by_cell.values()) for k in ks
    }
    weight = 1 / len(by_cell) if by_cell else 0.0
    cells = [
        CellContribution(
            cell=cell,
            records=metrics.records,
            record_share=metrics.records / scored if scored else 0.0,
            weight=weight,
            recall_share={
                k: (
                    metrics.at_k[k]["recall"] / totals[k]
                    if totals[k]
                    else weight  # no cell scored anything; share it out evenly
                )
                for k in ks
            },
        )
        for cell, metrics in by_cell.items()
    ]
    cells.sort(key=lambda cell: (-cell.mean_recall_share, cell.record_type, cell.language))
    return Divergence(delta=delta, cells=tuple(cells))


def _scores(precision: float, recall: float, decimals: int | None = None) -> dict:
    """One k's three numbers. `decimals` mirrors the official rounding of F1."""
    total = precision + recall
    f1 = 2 * precision * recall / total if total else 0.0
    if decimals is not None:
        f1 = round(f1, decimals)
    return {"precision": precision, "recall": recall, "f1": f1}
