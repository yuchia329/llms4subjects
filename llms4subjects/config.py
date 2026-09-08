"""Experiment configuration, loaded from committed YAML rather than edited into
scripts.

A rung of the experiment ladder is described by one file in `configs/`. The
loader is strict: an unknown key is an error rather than a silently ignored
typo, because a misspelled ablation flag would otherwise read as "off" and the
run would look like a measurement of something it is not.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .contracts import CODES_PER_RECORD
from .paths import SPLIT_FILES
from .stages.label_text import QUALIFIER_MODES

RETRIEVER_NAMES = ("knn", "dense", "lexical")
CORPUS_NAMES = tuple(SPLIT_FILES)


class ConfigError(ValueError):
    """A configuration file the loader will not guess the meaning of."""


@dataclass(frozen=True)
class LabelTextConfig:
    """How a vocabulary entry becomes the string an encoder reads."""

    # GND's comma-qualifiers are homograph disambiguators; see ticket 01.
    qualifiers: str = "parenthetical"
    # `Definition` holds cataloguing instructions on 18.1% of labels, so it is
    # an ablation rather than a default.
    include_definition: bool = False
    bilingual: bool = True

    def __post_init__(self):
        if self.qualifiers not in QUALIFIER_MODES:
            raise ConfigError(
                f"label_text.qualifiers must be one of {QUALIFIER_MODES}, "
                f"got {self.qualifiers!r}"
            )


@dataclass(frozen=True)
class EncoderConfig:
    """One of the four interchangeable encoders, plus any adapter over it."""

    name: str
    revision: str | None = None
    max_length: int = 512
    batch_size: int = 32
    # Path to a fine-tuned adapter artifact; None means off-the-shelf weights.
    adapter: str | None = None


@dataclass(frozen=True)
class IndexConfig:
    """Which documents go into the retrieval index, and how many.

    Index size varies independently of training-set size, so that "more
    neighbours" and "more training signal" stay separately attributable.
    """

    corpora: tuple[str, ...] = ("core_train",)
    size: int | None = None
    # Stratify the sample by record type and language when size is set.
    stratify: bool = False
    seed: int = 42

    def __post_init__(self):
        unknown = [name for name in self.corpora if name not in CORPUS_NAMES]
        if unknown:
            raise ConfigError(
                f"unknown index.corpora entry/entries: {', '.join(unknown)} "
                f"(known: {', '.join(CORPUS_NAMES)})"
            )
        if "core_test" in self.corpora:
            # The gold test set is opened exactly once, at the end of the
            # project, and never as index or training data.
            raise ConfigError("index.corpora must not include the gold test split")


@dataclass(frozen=True)
class RetrieverConfig:
    enabled: bool = True
    # Candidates this retriever contributes before fusion.
    top_k: int = 100
    # Fusion weight, tuned on dev.
    weight: float = 1.0
    # kNN only: neighbouring documents whose gold subjects are harvested.
    neighbours: int = 20

    def __post_init__(self):
        if self.top_k < 1:
            # A retriever clamps `top_k` to the vocabulary it has, so a negative
            # one would quietly return all-but-three candidates rather than
            # failing, and be reported as a `top_k` ablation.
            raise ConfigError(f"top_k must be at least 1, got {self.top_k}")


@dataclass(frozen=True)
class FusionConfig:
    """Reciprocal rank fusion over the three retrievers."""

    # The pipeline contract: candidate generation emits the top 100.
    candidates: int = 100
    rrf_k: int = 60


@dataclass(frozen=True)
class GroupPriorConfig:
    """66-way classifier over GND classification groups, used as a boost.

    There is no `model` here, and the absence is the design: the head is one
    linear layer over the configured encoder's document vectors, so the model
    is `encoder.name` and a second name would be a flag nobody could honour.

    `weight` is in units of a record's own candidate score range — at 1.0 a
    group the prior is certain of can move a candidate across the whole list —
    so it is comparable between a fused run and a single-retriever one.
    """

    enabled: bool = False
    weight: float = 1.0
    # Inverse L2 strength, passed to the per-group logistic regression as `C`.
    regularization: float = 1.0

    def __post_init__(self):
        if self.weight < 0:
            # A negative weight boosts the groups the prior finds *unlikely*,
            # which is not an ablation of anything. Zero is: the prior runs,
            # costs its training, and contributes nothing.
            raise ConfigError(
                f"group_prior.weight must not be negative, got {self.weight}"
            )
        if self.regularization <= 0:
            raise ConfigError(
                "group_prior.regularization must be positive, got "
                f"{self.regularization}"
            )


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    model: str | None = None
    input_k: int = 100
    output_k: int = CODES_PER_RECORD


@dataclass(frozen=True)
class AdjudicationConfig:
    enabled: bool = False
    model: str | None = None
    # Roughly the least-confident 20% of records are routed onward.
    route_fraction: float = 0.2
    candidates: int = 30
    prompt_revision: str = "v1"


@dataclass(frozen=True)
class ExperimentConfig:
    """One rung of the ladder, or one ablation of one."""

    name: str
    encoder: EncoderConfig
    index: IndexConfig = field(default_factory=IndexConfig)
    label_text: LabelTextConfig = field(default_factory=LabelTextConfig)
    retrievers: dict[str, RetrieverConfig] = field(
        default_factory=lambda: {name: RetrieverConfig() for name in RETRIEVER_NAMES}
    )
    fusion: FusionConfig = field(default_factory=FusionConfig)
    group_prior: GroupPriorConfig = field(default_factory=GroupPriorConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    adjudication: AdjudicationConfig = field(default_factory=AdjudicationConfig)
    notes: str = ""

    def section(self, name: str) -> dict[str, Any]:
        """One section as plain data, for artifact keying and manifests."""
        return _plain(getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=True, allow_unicode=True)


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error
    try:
        return load_experiment_text(text)
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from None


def load_experiment_text(text: str) -> ExperimentConfig:
    """Load an experiment configuration from YAML text."""
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")
    return _build(ExperimentConfig, raw, path="")


def _build(cls, raw: dict[str, Any], path: str):
    """Instantiate a config dataclass from a mapping, rejecting unknown keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        where = f"{path} " if path else ""
        raise ConfigError(
            f"unknown {where}key(s): {', '.join(unknown)} "
            f"(known: {', '.join(sorted(known))})"
        )

    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        nested = f"{path}.{name}" if path else name
        kwargs[name] = _coerce(known[name], value, nested)

    try:
        return cls(**kwargs)
    except TypeError as error:
        # A missing required field, e.g. `name` or `encoder`.
        raise ConfigError(str(error).replace("__init__()", cls.__name__)) from None


_SECTION_TYPES = {
    "encoder": EncoderConfig,
    "index": IndexConfig,
    "label_text": LabelTextConfig,
    "fusion": FusionConfig,
    "group_prior": GroupPriorConfig,
    "reranker": RerankerConfig,
    "adjudication": AdjudicationConfig,
}


def _coerce(spec: dataclasses.Field, value: Any, path: str):
    section = _SECTION_TYPES.get(spec.name)
    if section is not None:
        if not isinstance(value, dict):
            raise ConfigError(f"{path} must be a mapping")
        return _build(section, value, path)

    if spec.name == "retrievers":
        if not isinstance(value, dict):
            raise ConfigError(f"{path} must be a mapping")
        unknown = sorted(set(value) - set(RETRIEVER_NAMES))
        if unknown:
            raise ConfigError(
                f"unknown retriever(s): {', '.join(unknown)} "
                f"(known: {', '.join(RETRIEVER_NAMES)})"
            )
        retrievers = {name: RetrieverConfig() for name in RETRIEVER_NAMES}
        for name, settings in value.items():
            if not isinstance(settings, dict):
                raise ConfigError(f"{path}.{name} must be a mapping")
            retrievers[name] = _build(RetrieverConfig, settings, f"{path}.{name}")
        return retrievers

    if spec.name == "corpora" and isinstance(value, list):
        return tuple(value)

    return value


def _plain(value):
    """Config objects to JSON/YAML-safe data, so keys and manifests are stable."""
    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
