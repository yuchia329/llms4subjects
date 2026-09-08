"""Document and label indexes for one encoder and corpus selection.

Vectors and the index structure are returned to the caller, which decides where
they are cached; the index builder itself holds no paths.

Implemented by ticket 04 (documents), ticket 05 (labels) and ticket 06 (the
lexical index over label strings).
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Collection, Mapping, Protocol, Sequence

import numpy as np

from ..config import IndexConfig
from ..contracts import Code, Record
from .encoders import Encoder, SparseEncoder


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

    def __len__(self) -> int:
        return len(self.codes)


def _check_paired(codes: Sequence[Code], others: Sequence, noun: str) -> None:
    """Refuse two sequences that would zip short, whatever they hold.

    Every index here is built from codes paired with something rendered from
    them, and a short zip would file one code's row under another code's — which
    no later stage could detect, because the result is a well-formed index of
    the wrong thing.
    """
    if len(codes) != len(others):
        raise ValueError(
            f"{len(codes)} codes against {len(others)} {noun}; zipping them "
            "short would index a code under another code's row"
        )


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
    """Embed the whole vocabulary, one vector per entry, in the order given.

    Every entry is indexed, including the 64,820 tib-core codes that no training
    record carries: a label with no training example is reachable only by being
    scored directly, which is what this index is for.

    The texts go through `encode_labels` rather than `encode_documents`, because
    the label side of a document-to-label pair is the asymmetric one — E5 was
    trained with `passage:` here and `query:` on the document — and the two
    towers must not be encoded under the same convention by accident.
    """
    _check_paired(codes, texts, "label texts")
    return LabelIndex(
        codes=tuple(codes), vectors=encoder.encode_labels(list(texts))
    )


# --- Lexical matching over label strings ------------------------------------

# Okapi BM25's usual constants. They are not exposed as configuration: the
# lexical retriever's weight in the fusion is what an ablation tunes, and two
# more knobs would multiply the grid without changing which labels match.
LEXICAL_K1 = 1.2
LEXICAL_B = 0.75

# Word characters, which under `re.UNICODE` includes the umlauts and the eszett
# that a German heading is written with. No stemming and no stopword list: IDF
# already discounts a term that half the vocabulary shares, and a stemmer would
# have to be language-specific in a corpus that is half English.
TOKEN_PATTERN = re.compile(r"\w+")

# How many of the best-scoring label strings are inspected to find `top_k`
# distinct codes. Strings outnumber codes — a label has a name and its synonyms
# — so the window is widened until enough distinct codes are inside it.
VARIANT_HEADROOM = 4


def lexical_terms(text: str) -> list[str]:
    """The terms a lexical match is made of, from a label string or a record."""
    return TOKEN_PATTERN.findall(text.lower())


class LexicalMatcher(Protocol):
    """A lexical candidate source: record texts in, ranked codes out."""

    def rank(
        self, texts: Sequence[str], top_k: int
    ) -> list[list[tuple[Code, float]]]: ...

    def __len__(self) -> int: ...


class _InvertedIndex:
    """Term-weighted entries under an inverted index, best code per query.

    One entry is one string that can match: a label's preferred name, one of its
    synonyms, or — on the sparse path — the label's whole text. `owners` maps
    each entry back to the code it belongs to, so a label reached through both
    its name and a synonym is one candidate, at its best-matching string's
    score, rather than two candidates or a double-counted one.

    Only the entries a query's terms actually reach are touched, which is what
    makes this affordable at 79,427 labels: `rank_bm25`, the obvious
    off-the-shelf choice, walks every document for every query term, so a dev
    split of 5,354 records against the whole vocabulary is billions of dict
    lookups. The scores buffer is reused across queries and reset over exactly
    the entries that were touched.
    """

    def __init__(
        self,
        entries: Sequence[Mapping[str, float]],
        owners: Sequence[int],
        codes: Sequence[Code],
    ):
        self._codes = tuple(codes)
        self._owners = np.asarray(owners, dtype=np.int64)
        self._scores = np.zeros(len(entries), dtype=np.float64)

        postings: dict[str, tuple[list[int], list[float]]] = {}
        for position, weights in enumerate(entries):
            for term, weight in weights.items():
                positions, values = postings.setdefault(term, ([], []))
                positions.append(position)
                values.append(float(weight))
        self._postings = {
            term: (np.asarray(positions), np.asarray(values))
            for term, (positions, values) in postings.items()
        }

    def __len__(self) -> int:
        return len(self._codes)

    def rank(
        self, query: Mapping[str, float], top_k: int
    ) -> list[tuple[Code, float]]:
        """The best `top_k` codes for one query's weighted terms."""
        touched = []
        for term, weight in query.items():
            posting = self._postings.get(term)
            if posting is None:
                continue
            positions, values = posting
            # A term appears at most once in a posting list, so the indices are
            # unique and this is an ordinary in-place add rather than `add.at`.
            self._scores[positions] += values * weight
            touched.append(positions)

        if not touched or top_k <= 0:
            self._reset(touched)
            return []

        reached = np.unique(np.concatenate(touched))
        ranked = self._best_per_code(reached, top_k)
        self._scores[reached] = 0.0
        return ranked

    def _reset(self, touched: Sequence[np.ndarray]) -> None:
        for positions in touched:
            self._scores[positions] = 0.0

    def _best_per_code(
        self, reached: np.ndarray, top_k: int
    ) -> list[tuple[Code, float]]:
        """Collapse matched strings onto their codes, keeping the best score.

        The window over the matched strings is widened until it holds `top_k`
        distinct codes or all of them: collapsing can only remove entries, so a
        window of the best `n` strings yields at most `n` codes and the ones it
        yields are the right ones.
        """
        values = self._scores[reached]
        window = top_k * VARIANT_HEADROOM

        while True:
            best: dict[Code, float] = {}
            for position, score in self._window(reached, values, window):
                code = self._codes[self._owners[position]]
                # Not `> best.get(code, 0.0)`: a model whose term weights can go
                # negative would lose those codes rather than rank them last.
                current = best.get(code)
                if current is None or score > current:
                    best[code] = score
            if len(best) >= top_k or window >= len(reached):
                break
            window *= 2

        return sorted(best.items(), key=lambda item: (-item[1], item[0]))[:top_k]

    def _window(
        self, reached: np.ndarray, values: np.ndarray, window: int
    ) -> Sequence[tuple[int, float]]:
        """The `window` best-scoring matched strings, ties included.

        Ties are taken whole rather than cut through, so which of two equally
        scored labels survives is decided by the code order `_best_per_code`
        sorts on and never by where a partition happened to fall.
        """
        if window >= len(reached):
            return list(zip(reached.tolist(), values.tolist()))
        threshold = values[np.argpartition(-values, window - 1)[:window]].min()
        keep = values >= threshold
        return list(zip(reached[keep].tolist(), values[keep].tolist()))


class LexicalIndex:
    """Label strings under BM25, the fallback where an encoder has no weights.

    Each of a label's strings is one BM25 document, so a heading with nine
    synonyms is not penalised as one long document, and its score is its
    best-matching string's rather than a blend.

    A record's terms are taken once each, however often the text repeats them:
    what this retriever is measuring is whether a heading appears in the text at
    all — verbatim, for 53.7% of German gold assignments against 23.0% of
    English ones — and a title that says "Stahl" three times is not three times
    the evidence for it.
    """

    def __init__(self, inverted: _InvertedIndex):
        self._inverted = inverted

    def __len__(self) -> int:
        return len(self._inverted)

    def rank(
        self, texts: Sequence[str], top_k: int
    ) -> list[list[tuple[Code, float]]]:
        return [
            self._inverted.rank(
                {term: 1.0 for term in lexical_terms(text)}, top_k
            )
            for text in texts
        ]


class SparseTermIndex:
    """Label term weights from the encoder's own forward pass.

    The score is the sum over shared terms of the two sides' weights, which is
    the matching function the models that emit them are trained for.
    """

    def __init__(self, inverted: _InvertedIndex, encoder: SparseEncoder):
        self._inverted = inverted
        self._encoder = encoder

    def __len__(self) -> int:
        return len(self._inverted)

    def rank(
        self, texts: Sequence[str], top_k: int
    ) -> list[list[tuple[Code, float]]]:
        weights = self._encoder.encode_sparse_documents(list(texts))
        return [self._inverted.rank(row, top_k) for row in weights]


def build_lexical_index(
    codes: Sequence[Code], variants: Sequence[Sequence[str]]
) -> LexicalIndex:
    """Index each label's own strings — its name and its synonyms — under BM25.

    `variants` holds one group of strings per code, as
    `label_text.label_variants` renders them. The field-marked rendering is
    deliberately not what is indexed: `Fachgebiet: Chemie` is shared by
    thousands of labels and describes none of them.
    """
    _check_paired(codes, variants, "variant groups")

    owners: list[int] = []
    counted: list[dict[str, int]] = []
    for position, group in enumerate(variants):
        for string in group:
            terms = lexical_terms(string)
            if not terms:
                continue
            owners.append(position)
            counts: dict[str, int] = {}
            for term in terms:
                counts[term] = counts.get(term, 0) + 1
            counted.append(counts)

    return LexicalIndex(_InvertedIndex(_bm25_weights(counted), owners, codes))


def _bm25_weights(
    counted: Sequence[Mapping[str, int]]
) -> list[dict[str, float]]:
    """Okapi BM25 term weights, entry by entry, with the query side left to 1."""
    total = len(counted)
    if not total:
        return []

    lengths = [sum(counts.values()) for counts in counted]
    average = sum(lengths) / total

    document_frequency: dict[str, int] = {}
    for counts in counted:
        for term in counts:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    idf = {
        term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }

    weights = []
    for counts, length in zip(counted, lengths):
        # `average` is above zero because an entry with no terms is not
        # indexed at all, so there is nothing to guard against here.
        normalised = LEXICAL_K1 * (1 - LEXICAL_B + LEXICAL_B * length / average)
        weights.append(
            {
                term: idf[term] * frequency * (LEXICAL_K1 + 1) / (frequency + normalised)
                for term, frequency in counts.items()
            }
        )
    return weights


def build_sparse_term_index(
    codes: Sequence[Code], texts: Sequence[str], encoder: SparseEncoder
) -> SparseTermIndex:
    """Index the label texts by the encoder's own term weights.

    One text per code here, not the surface strings BM25 indexes: these weights
    come from the same forward pass over the same rendering that the dense label
    index uses, which is the whole reason to prefer them.
    """
    _check_paired(codes, texts, "label texts")
    weights = encoder.encode_sparse_labels(list(texts))
    return SparseTermIndex(
        _InvertedIndex(weights, range(len(codes)), codes), encoder
    )
