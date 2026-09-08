"""Which models this project is allowed to load, and the evidence for it.

docs/spec.md restricts every component to models released on or before
2025-01-31, the close of the SemEval-2025 Task 5 evaluation window, so that the
comparison against teams who competed in January 2025 is fair rather than
flattered by later model progress. That is a claim about the weights a run
actually loaded, and a model name alone does not carry it: `intfloat/
multilingual-e5-base` had commits landed on it in April 2026, so an unpinned
name resolves to something the cutoff never covered.

So the claim is a committed reference artifact — one entry per model, with its
creation date, the revision it is pinned to and that revision's date — and a
loader that refuses a model the registry does not vouch for. Ticket 09 needed
this to screen four encoders; every later stage that loads weights reads the
same registry, so "inside the cutoff" is one fact in one place.

The registry is written only by `scripts/verify_model_releases.py`, which
re-derives every date from the hub API. Nothing here reaches the network: a run
reads the file, and re-verification is a deliberate command that leaves a commit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .paths import MODEL_RELEASES_FILE

# The close of the shared task's evaluation window (docs/spec.md, story 2).
MODEL_CUTOFF = date(2025, 1, 31)

WRITE_HINT = "  python scripts/verify_model_releases.py --force"

SCHEMA = 1


class MissingModelRegistry(FileNotFoundError):
    """The registry is committed; this says how to write it if it is not there."""


class UnregisteredModel(KeyError):
    """A model name the registry does not vouch for."""

    def __str__(self) -> str:  # KeyError quotes its argument; this reads better.
        return self.args[0]


class ModelTooRecent(ValueError):
    """A model, or the revision it is pinned to, postdating the cutoff."""


class UnverifiedRevision(ValueError):
    """A revision asked for by a config that the registry has not dated.

    The cutoff is a claim about the commit that gets loaded, and an arbitrary
    SHA has no date attached without asking the hub — which no run does. So a
    config may not name one: registering it is what makes it checkable.
    """


@dataclass(frozen=True)
class ModelRelease:
    """One model's provenance: what it is, when it landed, what gets loaded.

    `created` is when the repository first appeared on the hub — the release
    date the cutoff is about — and `revision` is the commit a run pins to. The
    two are separate because they answer different questions: a 2023 model
    updated in 2026 is inside the cutoff only if the revision loaded is.
    """

    name: str
    role: str
    created: date
    revision: str
    revision_date: date
    source: str
    # Loading this model executes modelling code shipped from the hub rather
    # than from `transformers`. Recorded here rather than in a config, because
    # it is a property of the model and a thing being trusted, not a knob.
    trust_remote_code: bool = False
    # The repository the remote code comes from, and the commit of it, which is
    # as much a part of the cutoff claim as the weights are.
    code_repository: str | None = None
    code_revision: str | None = None
    # Where the dates come from. `hub` entries are re-derived from the Hugging
    # Face API by `scripts/verify_model_releases.py`; `api` entries are hosted
    # models with no weights to pin, whose dates are declared in that script
    # against the provider's own announcement and whose `revision` is the dated
    # model id the request actually names — `claude-3-5-sonnet-20241022`, not a
    # commit. The distinction is recorded rather than hidden, because a
    # declared date is weaker evidence than a fetched one and the writeup
    # should be able to say which each model has.
    origin: str = "hub"
    # Which provider a hosted model is called through; `None` for hub models.
    provider: str | None = None
    # Registered *because* it postdates the cutoff: docs/spec.md allows one
    # clearly-labelled appendix row on a current model, in the adjudication
    # stage alone, to quantify what model progress adds to the same pipeline.
    # Recorded here so that `--offline` and the verifier can tell an expected
    # post-cutoff entry from a mistake; `stages.adjudicator.resolve` still reads
    # the dates rather than this flag, because the dates are the claim.
    appendix: bool = False
    notes: str = ""


@dataclass(frozen=True)
class Registry:
    """The registry as loaded: the cutoff it was verified against, and models."""

    cutoff: date
    verified_at: date
    models: dict[str, ModelRelease]

    def __getitem__(self, name: str) -> ModelRelease:
        try:
            return self.models[name]
        except KeyError:
            raise UnregisteredModel(
                f"{name!r} is not in reference/model_releases.json, so nothing "
                "records whether it predates the "
                f"{self.cutoff.isoformat()} model cutoff. Add it with:\n"
                f"{WRITE_HINT}"
            ) from None


def load_registry(path: str | Path | None = None) -> Registry:
    """The committed registry. Reads a file; never the network."""
    source = Path(path) if path is not None else MODEL_RELEASES_FILE
    if not source.exists():
        raise MissingModelRegistry(
            f"{source} is missing, and every model load reads it. Write it with:\n"
            f"{WRITE_HINT}"
        )
    return _registry(json.loads(source.read_text(encoding="utf-8")))


def release(name: str, registry: Registry | None = None) -> ModelRelease:
    """What the registry records about one model, or a `KeyError` saying it does not."""
    return (registry if registry is not None else load_registry())[name]


def check_cutoff(name: str, registry: Registry | None = None) -> ModelRelease:
    """The model's entry, or `ModelTooRecent` naming the date that fails.

    Called by every loader before any weights are fetched, so a model outside
    the cutoff costs a sentence rather than a download and a run whose numbers
    have to be thrown away.
    """
    registry = registry if registry is not None else load_registry()
    entry = registry[name]

    if entry.created > registry.cutoff:
        raise ModelTooRecent(
            f"{name} was created {entry.created.isoformat()}, after the "
            f"{registry.cutoff.isoformat()} model cutoff docs/spec.md declares"
        )
    if entry.revision_date > registry.cutoff:
        raise ModelTooRecent(
            f"{name} is pinned to {entry.revision[:12]}, dated "
            f"{entry.revision_date.isoformat()}, after the "
            f"{registry.cutoff.isoformat()} model cutoff docs/spec.md declares"
        )
    return entry


def _registry(document: dict[str, Any]) -> Registry:
    return Registry(
        cutoff=date.fromisoformat(document["cutoff"]),
        verified_at=date.fromisoformat(document["verified_at"]),
        models={
            name: _entry(name, raw)
            for name, raw in sorted(document["models"].items())
        },
    )


def _entry(name: str, raw: dict[str, Any]) -> ModelRelease:
    return ModelRelease(
        name=name,
        role=raw["role"],
        created=date.fromisoformat(raw["created"]),
        revision=raw["revision"],
        revision_date=date.fromisoformat(raw["revision_date"]),
        source=raw["source"],
        trust_remote_code=raw.get("trust_remote_code", False),
        code_repository=raw.get("code_repository"),
        code_revision=raw.get("code_revision"),
        origin=raw.get("origin", "hub"),
        provider=raw.get("provider"),
        appendix=raw.get("appendix", False),
        notes=raw.get("notes", ""),
    )
