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
from typing import Any

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
