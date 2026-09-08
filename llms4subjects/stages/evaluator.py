"""Metrics from gold and predicted label lists. A pure function, by design.

Reports both aggregations the project needs — record-level micro averages for
model selection, and the official macro-over-cells figure for comparability
with the published leaderboard — and slices every metric by frozen frequency
band, document language and record type.

Bands are read from the committed frozen artifact and never recomputed, so
growing the index cannot silently reclassify which labels count as tail.

Implemented by ticket 03.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..contracts import Code

BANDS = ("head", "torso", "tail", "zero")

OFFICIAL_KS = tuple(range(5, 55, 5))


@dataclass(frozen=True)
class Metrics:
    """Precision, recall and F1 at each k, at one aggregation and slice."""

    at_k: Mapping[int, Mapping[str, float]]


@dataclass(frozen=True)
class EvaluationReport:
    micro: Metrics
    official_macro: Metrics
    by_band: Mapping[str, Metrics]
    by_language: Mapping[str, Metrics]
    by_type: Mapping[str, Metrics]


def evaluate(
    gold: Mapping[str, Sequence[Code]],
    predictions: Mapping[str, Sequence[Code]],
    bands: Mapping[Code, str],
    cells: Mapping[str, tuple[str, str]],
    ks: Sequence[int] = OFFICIAL_KS,
) -> EvaluationReport:
    """Score predictions against gold. `cells` maps record id to (type, lang)."""
    raise NotImplementedError("ticket 03: evaluator")
