"""Uniform interface over the four candidate encoders.

Hides their differing pooling, prefix and instruction conventions, so an
encoder swap is a config change. Emits dense vectors and, where the model
supports it, sparse term weights.

Vectors are L2-normalised on the way out, so every retriever downstream can
treat a dot product as cosine similarity and no stage has to know which model
produced the numbers.

Tickets 04 and 05 need one encoder and ticket 09 screens all four, which is why
`PREFIXES` is a table rather than a branch. A model absent from that table gets
no prefixes at all rather than E5's, which is the right default — a convention
guessed for the wrong family costs more than none — but it does mean adding an
encoder means adding its row, or its label tower is encoded under the document
convention. Ticket 06 declares the sparse interface a model may also offer; the
adapter that implements it arrives with the encoder that has it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..config import EncoderConfig


class Encoder(Protocol):
    """What every encoder adapter offers the rest of the pipeline."""

    @property
    def dimensions(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_labels(self, texts: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class SparseEncoder(Protocol):
    """An encoder whose forward pass also emits sparse term weights.

    BGE-M3 is the one of the four candidates that does. Where a model offers
    them, they are the lexical retriever's scores, since they arrive from the
    pass the dense tower has already paid for; where it does not, BM25 over the
    label strings is built instead.
    """

    def encode_sparse_documents(
        self, texts: Sequence[str]
    ) -> Sequence[Mapping[str, float]]: ...

    def encode_sparse_labels(
        self, texts: Sequence[str]
    ) -> Sequence[Mapping[str, float]]: ...


def sparse_weights(encoder: Encoder) -> SparseEncoder | None:
    """The encoder itself if it emits sparse term weights, otherwise nothing.

    Asked rather than configured, so that switching to a model that has them
    switches the lexical retriever over with it and a model that has not is
    never asked for something it cannot do.
    """
    return encoder if isinstance(encoder, SparseEncoder) else None


@dataclass(frozen=True)
class Prefixes:
    """The instruction strings a family expects in front of its two inputs.

    E5 was trained with `query:` and `passage:` and degrades measurably without
    them. Document-to-document retrieval is symmetric, so both sides of it get
    the document prefix; only the label tower is the asymmetric case.
    """

    document: str = ""
    label: str = ""


# Matched as a substring of the lowercased model name, longest match first, so a
# fine-tuned checkpoint named after its base model inherits its conventions.
PREFIXES = {
    "e5": Prefixes(document="query: ", label="passage: "),
}

NO_PREFIXES = Prefixes()

# Above this many texts, encoding is slow enough that silence looks like a hang.
PROGRESS_THRESHOLD = 2048


def prefixes_for(name: str) -> Prefixes:
    """The prefix convention for a model name, or none if it has no convention."""
    lowered = name.lower()
    matches = sorted(
        (key for key in PREFIXES if key in lowered), key=len, reverse=True
    )
    return PREFIXES[matches[0]] if matches else NO_PREFIXES


class SentenceTransformerEncoder:
    """A `sentence-transformers` model behind the `Encoder` protocol."""

    def __init__(self, model, prefixes: Prefixes, batch_size: int):
        self._model = model
        self._prefixes = prefixes
        self._batch_size = batch_size

    @property
    def dimensions(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self._prefixes.document)

    def encode_labels(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self._prefixes.label)

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        vectors = self._model.encode(
            [prefix + text for text in texts],
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > PROGRESS_THRESHOLD,
        )
        return np.asarray(vectors, dtype=np.float32)


def load(config: EncoderConfig, device: str) -> Encoder:
    """Build the adapter named by the config, on the given device."""
    if config.adapter is not None:
        raise NotImplementedError("ticket 15: fine-tuned adapters")

    # Imported here rather than at module scope: `transformers` and `torch` cost
    # seconds to import, and every test of this package that does not encode
    # anything would pay it.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        config.name,
        device=device,
        revision=config.revision,
    )
    model.max_seq_length = config.max_length
    return SentenceTransformerEncoder(
        model, prefixes_for(config.name), config.batch_size
    )
