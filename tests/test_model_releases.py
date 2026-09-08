"""The model cutoff, as a structure rather than a promise.

docs/spec.md restricts every component to models released on or before
2025-01-31, so that the comparison against teams who competed in January 2025
is fair rather than flattered by later model progress (story 2). A comment
saying so does not hold: an unpinned model name resolves to whatever the hub
serves today, and two of the four screened encoders have had commits landed on
them in 2026.

So the claim is carried by a committed registry — one entry per model any part
of this project loads, with its release date and the revision it is pinned to —
and by a loader that refuses a model the registry does not vouch for. These
tests are about that mechanism. They assert nothing about the hub, which is why
none of them reach the network: the registry is the record, and
`scripts/verify_model_releases.py` is what re-checks it against the hub.
"""

import json
import re
from datetime import date
from pathlib import Path

import pytest
import yaml

from llms4subjects.config import load_experiment
from llms4subjects.models import (
    MODEL_CUTOFF,
    MissingModelRegistry,
    ModelTooRecent,
    UnregisteredModel,
    UnverifiedRevision,
    check_cutoff,
    load_registry,
    release,
)
from llms4subjects.paths import CONFIG_DIR, MODEL_RELEASES_FILE
from llms4subjects.stages.encoders import resolve

# The close of the SemEval-2025 Task 5 evaluation window, quoted from
# docs/spec.md rather than imported, so that moving the constant in the source
# has to be an argued change here too.
CUTOFF_IN_THE_SPEC = date(2025, 1, 31)

FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@pytest.fixture(scope="module")
def document():
    return json.loads(MODEL_RELEASES_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def config_paths():
    return sorted(CONFIG_DIR.glob("*.yaml"))


def test_the_registry_is_committed():
    """`reference/` is tracked; a run must never have to write it."""
    assert MODEL_RELEASES_FILE.is_file()


def test_the_cutoff_is_the_one_the_spec_declares(registry):
    assert MODEL_CUTOFF == CUTOFF_IN_THE_SPEC
    assert registry.cutoff == CUTOFF_IN_THE_SPEC


def test_every_registered_model_was_released_before_the_cutoff(registry):
    """Except the one docs/spec.md allows, which has to say so about itself.

    The appendix row exists to quantify what model progress adds to an otherwise
    identical pipeline, so its model postdates the cutoff by construction. It is
    only allowed to because it is declared: `appendix` on the entry, and
    `adjudication.appendix` on the config that calls it.
    """
    for name, entry in registry.models.items():
        if entry.appendix:
            assert entry.role == "adjudicator", f"{name} is not the appendix stage"
            assert entry.created > registry.cutoff, (
                f"{name} is inside the cutoff and does not need the appendix flag"
            )
            continue
        assert entry.created <= registry.cutoff, f"{name} was created after the cutoff"


def test_every_registered_model_is_pinned_to_a_pre_cutoff_revision(registry):
    """A name alone resolves to today's weights; a revision is the actual claim."""
    for name, entry in registry.models.items():
        if entry.origin == "hub":
            assert FULL_COMMIT.match(entry.revision), f"{name} has no pinned revision"
        if not entry.appendix:
            assert entry.revision_date <= registry.cutoff, f"{name}'s revision is newer"


def test_a_hosted_model_is_pinned_to_the_dated_id_the_request_names(registry):
    """An API model has no commit to pin, so the id is the pin.

    Which is only a claim if the id is the one a request actually sends and the
    provider is recorded next to it — `claude-3-5-sonnet-20241022` names one set
    of weights, `claude-3-5-sonnet` names whichever is current.
    """
    hosted = [entry for entry in registry.models.values() if entry.origin == "api"]
    assert hosted, "the adjudicator's models are hosted; none is registered"
    for entry in hosted:
        assert entry.revision == entry.name, f"{entry.name} is pinned to something else"
        assert entry.provider, f"{entry.name} records no provider to call"
        assert entry.created == entry.revision_date, entry.name


def test_remote_code_is_pinned_too(registry):
    """Loading a model that carries its own modelling code runs that code.

    Whose revision is therefore as much a part of the cutoff claim as the
    weights' — and as much a part of what is being trusted.
    """
    for name, entry in registry.models.items():
        if not entry.trust_remote_code:
            assert entry.code_revision is None, f"{name} pins code it does not run"
            continue
        assert FULL_COMMIT.match(entry.code_revision or ""), f"{name} runs unpinned code"


def test_the_registry_records_what_it_was_verified_against(document):
    assert document["schema"] == 1
    assert document["verified_at"]
    for name, entry in document["models"].items():
        # Where the date came from, so a reader can re-check it by hand. A
        # hosted model has no hub page; what stands in its place is the
        # provider's announcement, and `origin` says which kind of evidence
        # this entry rests on rather than letting the two look alike.
        if entry.get("origin", "hub") == "hub":
            assert entry["source"].startswith("https://huggingface.co/"), name
        else:
            assert entry["source"].startswith("https://"), name
            assert not entry["source"].startswith("https://huggingface.co/"), name


@pytest.mark.parametrize("path", config_paths(), ids=lambda path: path.name)
def test_every_encoder_a_config_names_is_registered(path, registry):
    """The registry is only a record if nothing can be run that is missing from it."""
    config = load_experiment(path)
    assert config.encoder.name in registry.models, (
        f"{path.name} names {config.encoder.name}, which the registry does not "
        "vouch for; add it with scripts/verify_model_releases.py --force"
    )


@pytest.mark.parametrize("path", config_paths(), ids=lambda path: path.name)
def test_no_config_pins_a_revision_of_its_own(path):
    """`encoder.revision` is an override, and nothing should need it yet.

    The registry is where a revision is recorded and re-verified, so a config
    that carried one would be a second, unverified source of the same fact.
    """
    raw = yaml.safe_load(path.read_text()) or {}
    assert raw.get("encoder", {}).get("revision") is None, path.name


def test_release_names_the_script_that_would_add_a_missing_model():
    with pytest.raises(UnregisteredModel, match="verify_model_releases.py"):
        release("some-lab/some-model-nobody-registered")


def test_a_missing_registry_says_how_to_write_it(tmp_path):
    with pytest.raises(MissingModelRegistry, match="verify_model_releases.py"):
        load_registry(tmp_path / "absent.json")


def write_registry(path: Path, name: str, **overrides) -> Path:
    entry = {
        "role": "encoder",
        "created": "2024-01-01",
        "revision": "0" * 40,
        "revision_date": "2024-01-02",
        "source": f"https://huggingface.co/{name}",
    }
    entry.update(overrides)
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "cutoff": "2025-01-31",
                "verified_at": "2026-01-01",
                "models": {name: entry},
            }
        )
    )
    return path


def test_a_model_created_after_the_cutoff_is_refused(tmp_path):
    path = write_registry(tmp_path / "registry.json", "lab/late", created="2025-06-01")

    with pytest.raises(ModelTooRecent, match="2025-06-01"):
        check_cutoff("lab/late", registry=load_registry(path))


def test_a_pre_cutoff_model_pinned_to_a_post_cutoff_revision_is_refused(tmp_path):
    """The weights that get loaded are the revision's, not the release date's."""
    path = write_registry(
        tmp_path / "registry.json", "lab/updated", revision_date="2026-04-02"
    )

    with pytest.raises(ModelTooRecent, match="2026-04-02"):
        check_cutoff("lab/updated", registry=load_registry(path))


def test_an_in_cutoff_model_passes(tmp_path):
    path = write_registry(tmp_path / "registry.json", "lab/fine")

    assert check_cutoff("lab/fine", registry=load_registry(path)).name == "lab/fine"


# --- What the encoder loader resolves a config to ----------------------------


def test_the_loader_pins_the_revision_the_registry_vouches_for(registry):
    config = load_experiment(CONFIG_DIR / "rung1.yaml")

    resolved = resolve(config.encoder)

    assert resolved.revision == registry.models[config.encoder.name].revision


def test_a_config_may_restate_the_registrys_pin():
    """Being explicit is allowed; disagreeing is not."""
    import dataclasses

    config = load_experiment(CONFIG_DIR / "rung1.yaml")
    pin = release(config.encoder.name).revision
    restated = dataclasses.replace(config.encoder, revision=pin)

    assert resolve(restated).revision == pin


def test_a_config_revision_the_registry_has_not_dated_is_refused():
    """Otherwise the override carries the run past the cutoff, silently.

    A bare SHA has no date attached, and no run asks the hub for one, so a
    config that could name any commit could load anything while every date in
    the registry stayed true.
    """
    import dataclasses

    config = load_experiment(CONFIG_DIR / "rung1.yaml")
    elsewhere = dataclasses.replace(config.encoder, revision="f" * 40)

    with pytest.raises(UnverifiedRevision, match="Register that commit"):
        resolve(elsewhere)


def test_the_pin_reaches_the_artifact_key():
    """What the vectors were computed by has to be part of what keys them.

    A config file names no revision, so keying on the encoder section as
    written would serve one model's cached vectors to every revision of it: a
    re-pin to different weights would read the old ones back and report them as
    the new. `pinned` is what the harnesses key on.
    """
    from llms4subjects.artifacts import ArtifactStore
    from llms4subjects.stages.encoders import pinned

    config = load_experiment(CONFIG_DIR / "rung1.yaml")
    store = ArtifactStore("artifacts", data_revision="test")

    assert config.encoder.revision is None
    assert pinned(config).encoder.revision == release(config.encoder.name).revision
    assert store.key("embeddings", pinned(config)) != store.key("embeddings", config)


def test_the_loader_refuses_a_model_the_registry_does_not_vouch_for():
    """Refused before any weights are fetched, so the failure is a sentence."""
    import dataclasses

    config = load_experiment(CONFIG_DIR / "rung1.yaml")
    unknown = dataclasses.replace(config.encoder, name="some-lab/unregistered")

    with pytest.raises(UnregisteredModel):
        resolve(unknown)
