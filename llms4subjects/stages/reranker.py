"""Cross-encoder scoring of a document against each candidate's label text.

Reduces the top 100 to the final 50, and reports a per-record confidence so the
adjudicator can be routed only the least-confident records.

Implemented by ticket 12.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..config import RerankerConfig
from ..contracts import CandidateList, Code, Record


def rerank(
    records: Sequence[Record],
    candidates: Sequence[CandidateList],
    label_texts: Mapping[Code, str],
    config: RerankerConfig,
    device: str,
) -> list[CandidateList]:
    raise NotImplementedError("ticket 12: cross-encoder reranking")


def confidence(candidates: CandidateList) -> float:
    """Per-record confidence after reranking, used only for routing."""
    raise NotImplementedError("ticket 12: confidence measure")
