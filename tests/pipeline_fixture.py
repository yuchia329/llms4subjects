"""The committed corpus fixture, and a stand-in encoder, for pipeline tests.

The fixture is built by `scripts/build_pipeline_fixture.py`; this module only
reads it. The encoder is deliberately fake: docs/spec.md's testing decisions
rule out asserting that a particular encoder returns a particular vector, so the
seam tests assert invariants that hold whatever the vectors are, and a fake one
keeps the suite runnable on CPU in seconds and offline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from llms4subjects.contracts import Record, VocabularyEntry
from llms4subjects.corpus import entry_from_release

FIXTURE_FILE = Path(__file__).resolve().parent / "fixtures" / "corpus.json"

DIMENSIONS = 64


def fixture() -> dict:
    return json.loads(FIXTURE_FILE.read_text(encoding="utf-8"))


def records(entries: Sequence[dict]) -> list[Record]:
    return [
        Record(
            id=entry["id"],
            type=entry["type"],
            lang=entry["lang"],
            title=entry["title"],
            abstract=entry["abstract"],
            subjects=tuple(entry["subjects"]),
        )
        for entry in entries
    ]


def vocabulary(entries: Sequence[dict]) -> dict[str, VocabularyEntry]:
    return {entry["Code"]: entry_from_release(entry) for entry in entries}


class FakeEncoder:
    """Deterministic vectors from a text digest, counting what it was asked for.

    Two texts that differ anywhere get unrelated vectors, which is all the
    invariant tests need: they assert what happens to a ranking, never what the
    ranking is. `calls` is what makes the embedding cache observable.
    """

    def __init__(self, dimensions: int = DIMENSIONS):
        self._dimensions = dimensions
        self.calls: list[tuple[str, int]] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def encoded(self) -> int:
        """Texts this encoder actually turned into vectors."""
        return sum(count for _, count in self.calls)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(("documents", len(texts)))
        return self._vectors(texts)

    def encode_labels(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(("labels", len(texts)))
        return self._vectors(texts)

    def _vectors(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self._vector(text) for text in texts]
        if not rows:
            return np.zeros((0, self._dimensions), dtype=np.float32)
        return np.vstack(rows)

    def _vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        values = np.random.default_rng(seed).normal(size=self._dimensions)
        vector = values.astype(np.float32)
        return vector / np.linalg.norm(vector)


class FakeCrossEncoder:
    """Deterministic pair scores in [0, 1] from a digest, counting what it read.

    A real cross-encoder's ordering is a property of its weights, which the
    testing decisions rule out asserting. What the reranker tests need is an
    ordering that differs from the one fusion produced, is reproducible between
    runs, and moves when either side of the pair changes — a digest gives all
    three. `pairs` is what makes `input_k` observable.
    """

    def __init__(self):
        self.calls: list[int] = []

    @property
    def pairs(self) -> int:
        """Document-label pairs this model was actually asked to score."""
        return sum(self.calls)

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        self.calls.append(len(pairs))
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        return np.array(
            [self._score(document, label) for document, label in pairs],
            dtype=np.float32,
        )

    @staticmethod
    def _score(document: str, label: str) -> float:
        digest = hashlib.sha256(f"{document}\x00{label}".encode()).digest()
        return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
