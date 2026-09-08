"""Reciprocal rank fusion over the retrievers, keeping provenance.

The three retrievers score in three incomparable units — a summed cosine
similarity over neighbouring documents, a single cosine similarity against a
label vector, a BM25 score — so combining the scores themselves would let
whichever unit happens to be largest decide the ranking. Fusion therefore reads
rank and nothing else: a code's fused score is

    sum over the retrievers that found it of  weight / (rrf_k + rank)

with rank counted from 1, and taken as the candidate's position in the list this
stage was handed — after the vocabulary restriction, so the ranks are the
contiguous ones the fusion actually sees rather than a retriever's own positions
with the out-of-vocabulary drops still counted. `rrf_k` is what a rank is worth
relative to agreement: at 60 a code two retrievers put in their top 20 outranks
a code one retriever put first, and the smaller it is the harder a first place
is to overtake. The weights are tuned on dev and committed to the config, so no
retriever's influence is an accident of its score scale.

Each surviving candidate keeps every retriever's rank for it, because the
project's central claim is about attribution and a fused list that forgets where
its entries came from cannot support one. What it does not keep is the
retrievers' own scores: they are the three incomparable units this stage exists
to avoid combining, and a fused candidate carrying one of them would invite
exactly the comparison rank fusion is here to prevent. A stage that needs a
retriever's score reads that retriever's own output, which `pipeline.retrieve`
returns.

Implemented by ticket 07.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..config import FusionConfig
from ..contracts import Candidate, CandidateList, Code

# Positions are 0-based and reciprocal rank fusion is defined over 1-based
# ranks: 1/(rrf_k + 1) for a first place, not 1/rrf_k.
RANK_OFFSET = 1


def fuse(
    per_retriever: Mapping[str, Sequence[CandidateList]],
    weights: Mapping[str, float],
    config: FusionConfig,
) -> list[CandidateList]:
    """Fuse each retriever's ranked lists into one top-`candidates` list.

    Every retriever must have returned a list for the same records in the same
    order — they are all handed the same record sequence — and a mismatch is an
    error rather than a positional guess, because fusing one record's codes into
    another's would be invisible in the output and fatal to every number taken
    from it.

    With no retrievers at all the result is empty rather than an error: the
    ablation that disables everything is a legitimate row in the table, and the
    pipeline supplies the record ids for it, which fusion does not have.
    """
    if not per_retriever:
        return []

    missing = sorted(set(per_retriever) - set(weights))
    if missing:
        # A weight defaulted to 1.0 here would be reported as a tuned one.
        raise KeyError(
            f"no fusion weight for {', '.join(missing)}; every enabled "
            "retriever needs one"
        )

    names = list(per_retriever)
    records = _record_ids(per_retriever, names)

    fused: list[CandidateList] = []
    for position, record_id in enumerate(records):
        scores: dict[Code, float] = {}
        sources: dict[Code, dict[str, int]] = {}
        for name in names:
            weight = weights[name]
            ranked = per_retriever[name][position].candidates
            for rank, candidate in enumerate(ranked):
                # The position in this list is the rank, rather than whatever
                # the candidate remembers: a retriever stamps its own position
                # before the vocabulary restriction drops codes out from under
                # it, and a fused list that recorded those would leave gaps and
                # penalise a candidate for a neighbour that was filtered away.
                scores[candidate.code] = scores.get(candidate.code, 0.0) + (
                    weight / (config.rrf_k + rank + RANK_OFFSET)
                )
                sources.setdefault(candidate.code, {})[name] = rank

        fused.append(
            CandidateList(
                record_id=record_id,
                candidates=tuple(
                    Candidate(code=code, score=score, sources=sources[code])
                    for code, score in _ordered(scores)[: config.candidates]
                ),
            )
        )
    return fused


def _record_ids(
    per_retriever: Mapping[str, Sequence[CandidateList]], names: Sequence[str]
) -> list[str]:
    expected = [result.record_id for result in per_retriever[names[0]]]
    for name in names[1:]:
        found = [result.record_id for result in per_retriever[name]]
        if found != expected:
            differing = sorted(set(found) ^ set(expected)) or ["(reordered)"]
            raise ValueError(
                f"{names[0]} and {name} ranked different records — "
                f"{', '.join(differing[:5])}; fusion is positional and every "
                "retriever is handed the same records"
            )
    return expected


def _ordered(scores: Mapping[Code, float]) -> list[tuple[Code, float]]:
    """Codes by fused score, ties broken by code so no ranking depends on a dict."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
