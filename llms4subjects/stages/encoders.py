"""Uniform interface over the four candidate encoders.

Hides their differing pooling, prefix and instruction conventions, so an
encoder swap is a config change. Emits dense vectors and, where the model
supports it, sparse term weights.

Vectors are L2-normalised on the way out, so every retriever downstream can
treat a dot product as cosine similarity and no stage has to know which model
produced the numbers.

Ticket 04 needs one encoder; ticket 05 adds the remaining three and the sparse
term weights, which is why `PREFIXES` is a table rather than a branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from ..config import EncoderConfig


class Encoder(Protocol):
    """What every encoder adapter offers the rest of the pipeline."""

    @property
    def dimensions(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_labels(self, texts: Sequence[str]) -> np.ndarray: ...


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
