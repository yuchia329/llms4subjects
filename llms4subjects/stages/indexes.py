"""Document and label indexes for one encoder and corpus selection.

Vectors and the index structure are returned to the caller, which decides where
they are cached; the index builder itself holds no paths.

Implemented by ticket 04 (documents) and ticket 05 (labels).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Collection, Sequence

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

    def __len__(self) -> int:
        return len(self.record_ids)


@dataclass(frozen=True)
class LabelIndex:
    """Every vocabulary entry as a vector, so unseen labels stay reachable."""

    codes: tuple[Code, ...]
    vectors: np.ndarray


def select_documents(records: Sequence[Record], config: IndexConfig) -> list[Record]:
    """Sample the index corpus, stratified by record type and language.

    `size` is an absolute number of documents, never a fraction of what it was
    given: the experiment ladder varies index size alone, so "8,000 documents"
    has to mean the same thing whether it is drawn from tib-core train or from
    the larger all-subjects split. A `size` beyond the corpus takes all of it.
    """
    if config.size is None or config.size >= len(records):
        return list(records)

    rng = random.Random(config.seed)
    if not config.stratify:
        return _in_corpus_order(records, rng.sample(range(len(records)), config.size))

    cells: dict[tuple[str, str], list[int]] = {}
    for position, record in enumerate(records):
        cells.setdefault((record.type, record.lang), []).append(position)

    chosen: list[int] = []
    for cell, quota in _quotas(cells, config.size).items():
        chosen.extend(rng.sample(cells[cell], quota))
    return _in_corpus_order(records, chosen)


def _quotas(
    cells: dict[tuple[str, str], list[int]], size: int
) -> dict[tuple[str, str], int]:
    """How many documents each cell contributes: proportional, then rounded up.

    Largest remainder, which keeps every cell within one document of its share.
    A cell whose share rounds to nothing is then given one anyway, taken from
    the largest cell: an index missing a language or a record type outright is a
    different experiment rather than a smaller one.
    """
    total = sum(len(positions) for positions in cells.values())
    exact = {cell: len(positions) * size / total for cell, positions in cells.items()}
    quotas = {cell: int(share) for cell, share in exact.items()}

    remaining = size - sum(quotas.values())
    by_remainder = sorted(
        exact, key=lambda cell: (-(exact[cell] - quotas[cell]), cell)
    )
    for cell in by_remainder[:remaining]:
        quotas[cell] += 1

    if size >= len(cells):
        for cell in sorted(quotas):
            if quotas[cell]:
                continue
            donor = max(quotas, key=lambda other: (quotas[other], other))
            quotas[donor] -= 1
            quotas[cell] += 1

    # A quota can never exceed its cell: it is that cell's share of a sample
    # smaller than the corpus, and the one document a starved cell is given
    # comes from a cell that has more.
    return quotas


def _in_corpus_order(records: Sequence[Record], positions: Sequence[int]) -> list[Record]:
    """The chosen records in the order the corpus holds them, not sample order.

    Two configurations that select the same documents then produce byte-identical
    indexes, so a cached artifact is reusable rather than merely equivalent.
    """
    return [records[position] for position in sorted(positions)]


def build_document_index(
    records: Sequence[Record],
    encoder: Encoder,
    codes: Collection[Code] | None = None,
) -> DocumentIndex:
    """Embed the index corpus and keep the subjects a neighbour harvest reads.

    `codes` restricts the harvestable subjects to the vocabulary being predicted
    over. Restricting here rather than at the end of the pipeline keeps an
    out-of-vocabulary subject from spending a candidate slot on a code that
    would only be discarded — which matters for the all-subjects index, whose
    records carry codes outside tib-core.
    """
    subjects = tuple(
        tuple(
            code
            for code in dict.fromkeys(record.subjects)
            if codes is None or code in codes
        )
        for record in records
    )
    return DocumentIndex(
        record_ids=tuple(record.id for record in records),
        vectors=encoder.encode_documents([record.text for record in records]),
        subjects=subjects,
    )


def build_label_index(
    codes: Sequence[Code], texts: Sequence[str], encoder: Encoder
) -> LabelIndex:
    raise NotImplementedError("ticket 05: label index")
