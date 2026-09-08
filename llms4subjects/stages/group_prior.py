"""A 66-way classifier over GND classification groups, as a score adjustment.

This is the one place a dense output layer fits: 66 classes with roughly 485
training documents each. The predicted distribution boosts candidates in likely
groups and never filters unlikely ones, because 23.3% of records carry labels
spanning three or more groups.

Implemented by ticket 11.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..config import GroupPriorConfig
from ..contracts import CandidateList, Code, Record


def predict_groups(
    records: Sequence[Record], config: GroupPriorConfig, device: str
) -> list[Mapping[str, float]]:
    """A distribution over the 66 classification groups, per record."""
    raise NotImplementedError("ticket 11: group prior")


def apply_prior(
    candidates: Sequence[CandidateList],
    group_distributions: Sequence[Mapping[str, float]],
    group_of_code: Mapping[Code, str],
    config: GroupPriorConfig,
) -> list[CandidateList]:
    """Boost candidates whose group the prior finds likely."""
    raise NotImplementedError("ticket 11: group prior")
