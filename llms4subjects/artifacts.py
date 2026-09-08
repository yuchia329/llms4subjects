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
from typing import Any, Iterator, MutableMapping, Sequence

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
    # The trained 66-way head, which is a linear layer over the encoder's
    # document vectors for the index selection — so it is invalidated by the
    # encoder as surely as by the documents. Re-fitting it is seconds once the
    # vectors are cached, which is why the whole `group_prior` section keys it
    # rather than only the part training reads.
    "group_prior": ("encoder", "index", "group_prior"),
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


# What a model that emits sparse term weights offers on top of the dense
# interface. Named here because this wrapper forwards exactly these and refuses
# everything else, rather than standing in front of the whole encoder.
SPARSE_METHODS = ("encode_sparse_documents", "encode_sparse_labels")


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

    def __getattr__(self, name: str):
        """The wrapped encoder's sparse term weights, and nothing else, uncached.

        They are term-to-weight maps rather than one matrix, so this cache — one
        file per set of texts — has nothing to store them in yet. Forwarded
        rather than hidden because hiding them would be worse than not caching
        them: the lexical retriever asks the encoder whether it has weights, and
        a wrapper that answered no on its behalf would silently build BM25
        instead and report the wrong ablation.
        """
        if name not in SPARSE_METHODS:
            raise AttributeError(
                f"{type(self._encoder).__name__} behind a cache has no {name!r}"
            )
        return getattr(self._encoder, name)

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


GROUP_PRIOR_STAGE = "group_prior"

# The trained head, as two arrays and the column order they are in. An `.npz`
# rather than a pickled estimator, so that the artifact outlives the
# scikit-learn version that fitted it.
GROUP_PRIOR_FILE = "head.npz"


class MissingGroupPrior(FileNotFoundError):
    """`group_prior.enabled` is on and no head has been fitted for this config."""


def save_group_prior(store: ArtifactStore, config: ExperimentConfig, prior) -> Path:
    """Write a fitted head under this configuration's group-prior key."""
    directory = store.prepare(GROUP_PRIOR_STAGE, config)
    path = directory / GROUP_PRIOR_FILE
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez(
            handle,
            groups=np.array(prior.groups, dtype=object).astype("U"),
            coefficients=prior.coefficients,
            intercepts=prior.intercepts,
        )
    partial.replace(path)
    return path


def load_group_prior(store: ArtifactStore, config: ExperimentConfig):
    """The head fitted for this configuration, or how to fit it.

    Read here rather than asked of the caller for the reason the translation
    cache is: a harness that forgot to pass it would score an unboosted run and
    file the number under the boosted config.
    """
    from .stages.group_prior import GroupPrior

    path = store.path(GROUP_PRIOR_STAGE, config, GROUP_PRIOR_FILE)
    if not path.exists():
        raise MissingGroupPrior(
            f"{path} is missing, and `group_prior.enabled` needs it. Fit it "
            "with:\n  python scripts/train_group_prior.py <this config>"
        )
    with np.load(path, allow_pickle=False) as arrays:
        return GroupPrior(
            groups=tuple(str(group) for group in arrays["groups"]),
            coefficients=arrays["coefficients"],
            intercepts=arrays["intercepts"],
        )


ADJUDICATION_STAGE = "adjudicated"

# One JSON object per line, appended: the run that wrote line 900 of a thousand
# keeps 900 answers when it is interrupted, which a rewritten JSON file would
# not. Both files are read back by later lines overriding earlier ones, so a
# re-run that re-answers a record leaves a history rather than a conflict.
RESPONSES_FILE = "responses.jsonl"
REJECTIONS_FILE = "rejections.jsonl"


class ResponseCache(MutableMapping[str, str]):
    """The adjudicator's responses, written through as each one arrives.

    A plain dict would lose an interrupted run's answers, and those were paid
    for: the whole point of caching an LLM stage is that a response is bought
    once. So every assignment appends a line before it returns, and the file is
    the record of what has been billed for this configuration.

    Keyed through the store on the whole pipeline configuration, which includes
    the model, the prompt revision and every knob that changes a prompt — so a
    reworded prompt is a new cache rather than the old answers under a new name.
    """

    def __init__(self, path: Path, prepare, responses: dict[str, str]):
        self._path = path
        self._prepare = prepare
        self._responses = responses

    def __getitem__(self, record_id: str) -> str:
        return self._responses[record_id]

    def __setitem__(self, record_id: str, response: str) -> None:
        self._responses[record_id] = response
        self._prepare()
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"record_id": record_id, "response": response},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def __delitem__(self, record_id: str) -> None:
        # In memory only: the file is an append-only record of what was billed
        # for, and deleting from it would lose that.
        del self._responses[record_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._responses)

    def __len__(self) -> int:
        return len(self._responses)


def load_responses(store: ArtifactStore, config: ExperimentConfig) -> ResponseCache:
    """The responses already bought for this configuration, ready to be added to.

    Read here rather than asked of the caller, for the reason the translation
    cache and the group-prior head are: a harness that forgot it would re-bill
    a whole routed subset to be told what it already knows.
    """
    path = store.path(ADJUDICATION_STAGE, config, RESPONSES_FILE)
    return ResponseCache(
        path,
        lambda: store.prepare(ADJUDICATION_STAGE, config),
        _read_jsonl_map(path, "record_id", "response"),
    )


def log_rejections(
    store: ArtifactStore, config: ExperimentConfig, adjudications: Sequence
) -> Path | None:
    """Record every response that was refused, and why.

    The constraint is only enforced if its violations are visible: a model that
    invents identifiers has to leave evidence in the run's artifacts rather than
    a silently unchanged ranking. Returns the file written, or `None` when there
    was nothing new to write.

    A violation already logged for this configuration is not written again. Most
    of a second run's responses are replayed from the cache and bought nothing,
    so appending them would make the violation count grow with re-scores rather
    than with violations — and that count is the measurement the constraint is
    reported by.
    """
    rejected = [result for result in adjudications if result.rejected]
    if not rejected:
        return None

    logged = {
        (entry["record_id"], entry["reason"]) for entry in read_rejections(store, config)
    }
    rejected = [
        result
        for result in rejected
        if (result.record_id, result.reason) not in logged
    ]
    if not rejected:
        return None

    directory = store.prepare(ADJUDICATION_STAGE, config)
    path = directory / REJECTIONS_FILE
    with path.open("a", encoding="utf-8") as handle:
        for result in rejected:
            handle.write(
                json.dumps(
                    {"record_id": result.record_id, "reason": result.reason},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def read_rejections(
    store: ArtifactStore, config: ExperimentConfig
) -> list[dict[str, str]]:
    """The constraint violations logged for this configuration, newest last."""
    path = store.path(ADJUDICATION_STAGE, config, REJECTIONS_FILE)
    return _read_jsonl(path)


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_jsonl_map(path: Path, key: str, value: str) -> dict[str, str]:
    return {entry[key]: entry[value] for entry in _read_jsonl(path)}
