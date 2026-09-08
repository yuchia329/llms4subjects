"""LLM selection over a candidate list, with hard constraint validation.

The model sees the top 30 candidates for a routed record and may reorder them.
Any identifier it returns that was not offered is a constraint violation: the
response is rejected and logged rather than silently scored, so the model can
never invent a plausible-looking GND code.

Responses are cached by record id and prompt revision, so re-scoring does not
re-bill.

Implemented by ticket 13.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..config import AdjudicationConfig
from ..contracts import CandidateList, Code, Record


class ConstraintViolation(ValueError):
    """The model returned an identifier that was not in the candidate set."""


@dataclass(frozen=True)
class Adjudication:
    record_id: str
    codes: tuple[Code, ...]
    rejected: bool = False
    reason: str = ""


def route(
    candidates: Sequence[CandidateList],
    confidences: Sequence[float],
    config: AdjudicationConfig,
) -> list[str]:
    """Record ids of the least-confident `route_fraction` of records."""
    raise NotImplementedError("ticket 13: routing")


def adjudicate(
    records: Sequence[Record],
    candidates: Sequence[CandidateList],
    label_texts: Mapping[Code, str],
    config: AdjudicationConfig,
    responses: Mapping[str, str] | None = None,
) -> list[Adjudication]:
    """Select from the offered candidates, rejecting out-of-set identifiers.

    `responses` is the cache: a recorded response per record id, so tests and
    re-scores never call the API.
    """
    raise NotImplementedError("ticket 13: LLM adjudication")


def validate(offered: Sequence[Code], returned: Sequence[Code]) -> tuple[Code, ...]:
    """Return `returned` if it is a subset of `offered`, else raise."""
    raise NotImplementedError("ticket 13: constraint validation")
