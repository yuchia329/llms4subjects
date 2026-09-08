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

from ..config import RetrieverConfig
from ..contracts import CandidateList, Record
from ..contracts import Code
from .encoders import Encoder
from .indexes import DocumentIndex, LabelIndex


class Retriever(Protocol):
    """One candidate source. `name` is the provenance tag on its candidates."""

    name: str

    def retrieve(self, records: Sequence[Record]) -> list[CandidateList]: ...


class NeighbourRetriever:
    """Document-to-document: harvest the subjects of the nearest neighbours."""

    name = "knn"

    def __init__(
        self, index: DocumentIndex, encoder: Encoder, config: RetrieverConfig
    ):
        raise NotImplementedError("ticket 04: kNN retriever")

    def retrieve(self, records: Sequence[Record]) -> list[CandidateList]:
        raise NotImplementedError("ticket 04: kNN retriever")


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
