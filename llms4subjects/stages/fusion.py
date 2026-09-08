"""Reciprocal rank fusion over the retrievers, keeping provenance.

Weights are tuned on dev so that no retriever's score scale dominates by
accident, and each surviving candidate remembers which retrievers found it and
at what rank.

Implemented by ticket 07.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..config import FusionConfig
from ..contracts import CandidateList


def fuse(
    per_retriever: Mapping[str, Sequence[CandidateList]],
    weights: Mapping[str, float],
    config: FusionConfig,
) -> list[CandidateList]:
    """Fuse each retriever's ranked lists into one top-`candidates` list.

    With no retrievers enabled the result is one empty candidate list per
    record, not an error: an ablation that disables everything is a legitimate
    row in the table.
    """
    raise NotImplementedError("ticket 07: reciprocal rank fusion")
