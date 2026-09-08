"""Document and label indexes for one encoder and corpus selection.

Vectors and the index structure are returned to the caller, which decides where
they are cached; the index builder itself holds no paths.

Implemented by ticket 04 (documents) and ticket 05 (labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import IndexConfig
from ..contracts import Code, Record
from .encoders import Encoder


@dataclass(frozen=True)
class DocumentIndex:
    """Indexed documents, with the gold subjects a kNN retriever harvests."""

    record_ids: tuple[str, ...]
    vectors: np.ndarray
    subjects: tuple[tuple[Code, ...], ...]


@dataclass(frozen=True)
class LabelIndex:
    """Every vocabulary entry as a vector, so unseen labels stay reachable."""

    codes: tuple[Code, ...]
    vectors: np.ndarray


def select_documents(records: Sequence[Record], config: IndexConfig) -> list[Record]:
    """Sample the index corpus, stratified by record type and language."""
    raise NotImplementedError("ticket 04: index selection")


def build_document_index(
    records: Sequence[Record], encoder: Encoder
) -> DocumentIndex:
    raise NotImplementedError("ticket 04: document index")


def build_label_index(
    codes: Sequence[Code], texts: Sequence[str], encoder: Encoder
) -> LabelIndex:
    raise NotImplementedError("ticket 05: label index")
