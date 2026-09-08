"""The cached-artifact convention.

Every stage writes its output under `artifacts/<stage>/<key>/`, where the key is
a fingerprint of exactly the configuration sections that stage depends on, plus
the dataset revision. Changing the reranker therefore leaves the index keys
untouched, and nothing is recomputed to answer a question it does not affect.

The directory is ignored by git; see docs/artifacts.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import ExperimentConfig
from .paths import ARTIFACT_DIR

DEFAULT_ROOT = ARTIFACT_DIR

MANIFEST_NAME = "manifest.json"

# Every section a full prediction run depends on, in dependency order. The late
# stages share it, so adding a section to the pipeline is one edit, not four.
FULL_PIPELINE = (
    "label_text",
    "encoder",
    "index",
    "retrievers",
    "fusion",
    "group_prior",
    "reranker",
    "adjudication",
)

# Which configuration sections each stage's output actually depends on. A stage
# is invalidated by its own sections and by everything it consumes.
STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "label_text": ("label_text",),
    # Vectors for one set of texts, keyed by the encoder alone, so that a
    # second run over the same corpus reads them back instead of spending the
    # encoder again — which is nearly all the wall clock of an ablation that
    # changes anything downstream of it.
    "embeddings": ("encoder",),
    "label_index": ("label_text", "encoder"),
    "document_index": ("encoder", "index"),
    "candidates": ("label_text", "encoder", "index", "retrievers", "fusion"),
    "group_prior": ("index", "group_prior"),
    "reranked": FULL_PIPELINE[: FULL_PIPELINE.index("adjudication")],
    "adjudicated": FULL_PIPELINE,
    "predictions": FULL_PIPELINE,
    "metrics": FULL_PIPELINE,
}

STAGES = tuple(STAGE_DEPENDENCIES)

KEY_LENGTH = 12


class UnknownStage(KeyError):
    """A stage name with no declared dependencies, usually a typo."""

    def __str__(self) -> str:  # KeyError quotes its argument; this reads better.
        return self.args[0]


class ArtifactStore:
    """Locations of cached stage outputs for one dataset revision."""

    def __init__(self, root: str | Path = DEFAULT_ROOT, *, data_revision: str):
        self.root = Path(root)
        self.data_revision = data_revision

    def key(self, stage: str, config: ExperimentConfig) -> str:
        """Fingerprint of the configuration this stage's output depends on."""
        payload = json.dumps(
            self._fingerprint_payload(stage, config), sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:KEY_LENGTH]

    def directory(self, stage: str, config: ExperimentConfig) -> Path:
        """Where this stage's output lives. Reading only; nothing is created."""
        return self.root / stage / self.key(stage, config)

    def path(self, stage: str, config: ExperimentConfig, filename: str) -> Path:
        """Path to one file of this stage's output. Nothing is created.

        Creating on read would make a cache miss indistinguishable from a hit
        on the next run, so a stage that is about to write calls
        `prepare` first.
        """
        return self.directory(stage, config) / filename

    def prepare(self, stage: str, config: ExperimentConfig) -> Path:
        """Create this stage's directory and write its manifest. For writers."""
        path = self.directory(stage, config)
        path.mkdir(parents=True, exist_ok=True)
        manifest = path / MANIFEST_NAME
        if not manifest.exists():
            manifest.write_text(
                json.dumps(
                    self._fingerprint_payload(stage, config),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
        return path

    def exists(
        self, stage: str, config: ExperimentConfig, filename: str = MANIFEST_NAME
    ) -> bool:
        """Whether this stage's output is already cached."""
        return self.path(stage, config, filename).exists()

    def manifest(self, stage: str, config: ExperimentConfig) -> dict[str, Any]:
        """What the key was computed from, as written next to the artifact."""
        path = self.root / stage / self.key(stage, config) / MANIFEST_NAME
        if path.exists():
            return json.loads(path.read_text())
        return self._fingerprint_payload(stage, config)

    def _fingerprint_payload(
        self, stage: str, config: ExperimentConfig
    ) -> dict[str, Any]:
        sections = STAGE_DEPENDENCIES.get(stage)
        if sections is None:
            raise UnknownStage(
                f"unknown stage {stage!r} (known: {', '.join(STAGES)})"
            )
        return {
            "stage": stage,
            "data_revision": self.data_revision,
            "config": {name: config.section(name) for name in sections},
        }


EMBEDDING_STAGE = "embeddings"

# Length of the digest that stands for one set of input texts. Long enough that
# two different corpora cannot collide, short enough to read in a directory.
INPUT_REVISION_LENGTH = 16


def input_revision(texts: Sequence[str]) -> str:
    """A digest of exactly these texts, in this order: the cache's other half.

    Changing a record's abstract, adding a document to the index, or reordering
    the corpus all produce a different revision, so a cached matrix is served
    only to the inputs that produced it.
    """
    digest = hashlib.sha256()
    digest.update(str(len(texts)).encode())
    for text in texts:
        digest.update(b"\x00")
        digest.update(text.encode())
    return digest.hexdigest()[:INPUT_REVISION_LENGTH]


class CachedEncoder:
    """An encoder that computes each set of texts once, then reads them back.

    Wrapping the encoder rather than caching at the index builder is what makes
    the saving general: the index corpus, the records being predicted and the
    label tower all go through the same door, so a second run over the same dev
    split with the same encoder loads vectors instead of recomputing them —
    which is most of the wall clock of an ablation that changes only fusion.

    The key is the encoder configuration and the dataset revision, through the
    store, plus a digest of the texts themselves. One file holds one whole
    matrix, so the unit of reuse is the exact set of texts: a 32,043-document
    index and an 8,000-document sample of it share nothing, because a per-text
    cache of 32,043 files would cost more in stat calls than it saves.
    """

    def __init__(
        self, encoder, store: "ArtifactStore", config: ExperimentConfig
    ):
        self._encoder = encoder
        self._store = store
        self._config = config

    @property
    def dimensions(self) -> int:
        return self._encoder.dimensions

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._cached("documents", texts, self._encoder.encode_documents)

    def encode_labels(self, texts: Sequence[str]) -> np.ndarray:
        return self._cached("labels", texts, self._encoder.encode_labels)

    def _cached(self, kind: str, texts: Sequence[str], compute) -> np.ndarray:
        if not texts:
            return compute(texts)

        filename = f"{kind}-{input_revision(texts)}.npy"
        path = self._store.path(EMBEDDING_STAGE, self._config, filename)
        if path.exists():
            return np.load(path)

        vectors = compute(texts)
        self._store.prepare(EMBEDDING_STAGE, self._config)
        # Written aside and moved into place: a run interrupted mid-write would
        # otherwise leave a truncated file that `exists` reports as a hit, and
        # every later run would fail to read it rather than recompute it.
        partial = path.with_name(path.name + ".partial")
        # Through a handle, because `np.save` appends `.npy` to any filename
        # that does not already end in it.
        with partial.open("wb") as handle:
            np.save(handle, vectors)
        partial.replace(path)
        return vectors
