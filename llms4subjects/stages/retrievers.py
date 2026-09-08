"""Three retrievers behind one interface, each returning scored candidates.

- `knn` — nearest training documents, harvesting their gold subjects. This is
  the mechanism the winning leaderboard system used.
- `dense` — the document scored against all 79,427 label vectors, which is the
  only path to a label that appears in no training record.
- `lexical` — label strings matched against the document text, for German
  compound headings that appear verbatim.

A retriever never sees the gold subjects of the record it is retrieving for;
the index's subjects belong to other records. See docs/spec.md,
"Anti-requirements".

Implemented by tickets 04 (knn), 05 (dense) and 06 (lexical).
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..config import RetrieverConfig
from ..contracts import Candidate, CandidateList, Record
from ..contracts import Code
from .encoders import Encoder
from .indexes import DocumentIndex, LabelIndex

# Query vectors are scored against the whole index at once, so the similarity
# block is held to a few hundred rows: 5,354 dev records against 32,043 indexed
# documents is 686 MB in one piece and 33 MB in these.
QUERY_BATCH = 256

# Keeps a filled candidate strictly below every harvested one, whatever the
# similarities happen to be.
FILL_GAP = 1e-6


class Retriever(Protocol):
    """One candidate source. `name` is the provenance tag on its candidates."""

    name: str

    def retrieve(self, records: Sequence[Record]) -> list[CandidateList]: ...


class NeighbourRetriever:
    """Document-to-document: harvest the subjects of the nearest neighbours.

    A code's score is the summed similarity of the neighbours that carry it, so
    a subject two close documents agree on outranks one a single document
    mentions. Similarities are clipped at zero: a neighbour that is unlike the
    record should contribute nothing, not argue against a code some other
    neighbour supports.

    The record being predicted is dropped from its own neighbourhood by id. It
    is only ever there when the index and the query set overlap, and when it is,
    its own gold subjects would arrive as its nearest neighbour's — which is the
    construction docs/spec.md names as the reason any number from the earlier
    draft is meaningless.

    ## Harvesting, and then filling

    `neighbours` documents rarely carry `top_k` distinct subjects — 20
    neighbours at 2.4 subjects each is roughly 40 codes, and the submission
    format takes 50. So the scan continues past `neighbours` until `top_k`
    distinct codes exist, and everything found in that continuation is ranked
    *below* everything harvested, by shifting its scores under the harvest.

    That keeps `neighbours` meaning exactly what it says. A deeper scan that
    simply kept summing would let a code seen twice at neighbours 30 and 40
    outrank one seen at neighbour 2, which would make the setting a suggestion
    rather than a parameter and quietly change what an ablation over it measures.
    """

    name = "knn"

    def __init__(
        self, index: DocumentIndex, encoder: Encoder, config: RetrieverConfig
    ):
        self._index = index
        self._encoder = encoder
        self._config = config

    def retrieve(self, records: Sequence[Record]) -> list[CandidateList]:
        if not records:
            return []
        if not len(self._index):
            return [CandidateList(record.id, ()) for record in records]

        vectors = self._encoder.encode_documents([record.text for record in records])

        results: list[CandidateList] = []
        for start in range(0, len(records), QUERY_BATCH):
            batch = records[start : start + QUERY_BATCH]
            similarities = _similarities(
                vectors[start : start + len(batch)], self._index.vectors
            )
            results.extend(
                self._rank(record, row) for record, row in zip(batch, similarities)
            )
        return results

    def _rank(self, record: Record, similarities: np.ndarray) -> CandidateList:
        neighbours = [
            position
            for position in np.argsort(-similarities, kind="stable")
            if self._index.record_ids[position] != record.id
        ]
        top_k = self._config.top_k

        harvested = self._harvest(neighbours[: self._config.neighbours], similarities)
        ranked = _ordered(harvested)[:top_k]

        if len(ranked) < top_k:
            ranked += self._fill(
                neighbours[self._config.neighbours :],
                similarities,
                harvested,
                top_k - len(ranked),
            )

        return CandidateList(
            record_id=record.id,
            candidates=tuple(
                Candidate(code=code, score=score, sources={self.name: rank})
                for rank, (code, score) in enumerate(ranked)
            ),
        )

    def _harvest(
        self, neighbours: Sequence[int], similarities: np.ndarray
    ) -> dict[Code, float]:
        scores: dict[Code, float] = {}
        for position in neighbours:
            weight = max(float(similarities[position]), 0.0)
            for code in self._index.subjects[position]:
                scores[code] = scores.get(code, 0.0) + weight
        return scores

    def _fill(
        self,
        deeper: Sequence[int],
        similarities: np.ndarray,
        harvested: dict[Code, float],
        wanted: int,
    ) -> list[tuple[Code, float]]:
        """Codes from beyond `neighbours`, ranked under everything harvested."""
        scores: dict[Code, float] = {}
        for position in deeper:
            weight = max(float(similarities[position]), 0.0)
            for code in self._index.subjects[position]:
                if code not in harvested:
                    scores[code] = scores.get(code, 0.0) + weight
            if len(scores) >= wanted:
                break

        ordered = _ordered(scores)[:wanted]
        if not ordered:
            return []

        floor = min(harvested.values()) if harvested else 0.0
        offset = floor - FILL_GAP - ordered[0][1]
        return [(code, score + offset) for code, score in ordered]


def _similarities(queries: np.ndarray, indexed: np.ndarray) -> np.ndarray:
    """Cosine similarities, both sides having been normalised by the encoder.

    Apple's Accelerate BLAS leaves the floating-point status flags set after a
    perfectly ordinary matrix product, so numpy reports overflow and division by
    zero on results that are finite and correct. The flags are ignored and the
    result is checked instead, which is the assertion the warnings were only
    gesturing at.
    """
    with np.errstate(all="ignore"):
        similarities = queries @ indexed.T
    if not np.isfinite(similarities).all():
        raise ValueError(
            "similarities are not finite; the encoder returned a vector that is "
            "not a unit vector, or the index holds one"
        )
    return similarities


def _ordered(scores: dict[Code, float]) -> list[tuple[Code, float]]:
    """Codes by score, ties broken by code so a ranking never depends on a dict."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class DenseLabelRetriever:
    """Document-to-label: score the document against every vocabulary entry."""

    name = "dense"

    def __init__(
        self, index: LabelIndex, encoder: Encoder, config: RetrieverConfig
    ):
        raise NotImplementedError("ticket 05: dense label retriever")

    def retrieve(self, records: Sequence[Record]) -> list[CandidateList]:
        raise NotImplementedError("ticket 05: dense label retriever")


class LexicalLabelRetriever:
    """Lexical matching over label strings."""

    name = "lexical"

    def __init__(
        self,
        codes: Sequence[Code],
        texts: Sequence[str],
        config: RetrieverConfig,
    ):
        raise NotImplementedError("ticket 06: lexical retriever")

    def retrieve(self, records: Sequence[Record]) -> list[CandidateList]:
        raise NotImplementedError("ticket 06: lexical retriever")
