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
import re
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np

from llms4subjects.contracts import Record, VocabularyEntry
from llms4subjects.corpus import entry_from_release
from llms4subjects.models import MODEL_CUTOFF, ModelRelease, Registry

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


class FakeLanguageModel:
    """An adjudicator that answers from the prompt, counting what it was asked.

    It reads the candidate codes out of the prompt it was given and returns some
    of them in an order of its own, which is everything the adjudication tests
    need: a valid response that differs from the ranking that produced it, and
    no API key, network or bill. `answer` overrides that with a fixed response,
    which is how the constraint-violation path is tested.

    `prompts` is what makes the response cache observable — a model that is
    never asked is a cache hit.
    """

    def __init__(self, answer: str | None = None, select: int = 10):
        self._answer = answer
        self._select = select
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._answer is not None:
            return self._answer
        codes = _offered_codes(prompt)
        chosen = sorted(codes, key=lambda code: hashlib.sha256(code.encode()).digest())
        return json.dumps(chosen[: self._select])


def _offered_codes(prompt: str) -> list[str]:
    """The codes a v1 prompt numbers, in the order it lists them."""
    return re.findall(r"^\s*\d+\.\s+(\S+)", prompt, flags=re.MULTILINE)


def registry_with_recent_llm() -> Registry:
    """A registry holding one pre-cutoff LLM and one released after it.

    Built here rather than reached for in the committed file: the appendix rule
    is about a model the cutoff does not cover, and the committed registry
    should never hold one until an appendix run registers it.
    """
    return Registry(
        cutoff=MODEL_CUTOFF,
        verified_at=MODEL_CUTOFF,
        models={
            "claude-3-5-sonnet-20241022": ModelRelease(
                name="claude-3-5-sonnet-20241022",
                role="adjudicator",
                created=date(2024, 10, 22),
                revision="claude-3-5-sonnet-20241022",
                revision_date=date(2024, 10, 22),
                source="https://www.anthropic.com/news/3-5-models-and-computer-use",
                origin="api",
                provider="anthropic",
            ),
            "acme/tomorrow-llm": ModelRelease(
                name="acme/tomorrow-llm",
                role="adjudicator",
                created=date(2026, 6, 1),
                revision="acme/tomorrow-llm",
                revision_date=date(2026, 6, 1),
                source="https://example.invalid/tomorrow",
                origin="api",
                provider="anthropic",
            ),
        },
    )
