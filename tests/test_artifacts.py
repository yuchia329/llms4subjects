"""The cache convention: changing the reranker must not invalidate the index."""

import pytest

from llms4subjects.artifacts import STAGES, ArtifactStore, UnknownStage
from llms4subjects.config import load_experiment_text

BASE = "name: base\nencoder: {name: intfloat/multilingual-e5-base}\n"


def store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts", data_revision="rev0")


def test_key_is_stable_across_calls(tmp_path):
    s = store(tmp_path)
    config = load_experiment_text(BASE)
    assert s.key("document_index", config) == s.key("document_index", config)


def test_changing_the_reranker_does_not_invalidate_the_index(tmp_path):
    s = store(tmp_path)
    before = load_experiment_text(BASE)
    after = load_experiment_text(
        BASE + "reranker: {enabled: true, model: BAAI/bge-reranker-v2-m3}\n"
    )

    assert s.key("document_index", after) == s.key("document_index", before)
    assert s.key("label_index", after) == s.key("label_index", before)
    assert s.key("candidates", after) == s.key("candidates", before)
    assert s.key("reranked", after) != s.key("reranked", before)
    assert s.key("predictions", after) != s.key("predictions", before)


def test_changing_the_encoder_invalidates_everything_downstream(tmp_path):
    s = store(tmp_path)
    before = load_experiment_text(BASE)
    after = load_experiment_text("name: base\nencoder: {name: BAAI/bge-m3}\n")

    for stage in ("document_index", "label_index", "candidates", "reranked", "predictions"):
        assert s.key(stage, after) != s.key(stage, before), stage


def test_label_text_flags_do_not_touch_the_document_index(tmp_path):
    s = store(tmp_path)
    before = load_experiment_text(BASE)
    after = load_experiment_text(BASE + "label_text: {include_definition: true}\n")

    assert s.key("document_index", after) == s.key("document_index", before)
    assert s.key("label_index", after) != s.key("label_index", before)


def test_data_revision_invalidates_every_stage(tmp_path):
    config = load_experiment_text(BASE)
    a = ArtifactStore(tmp_path / "artifacts", data_revision="rev0")
    b = ArtifactStore(tmp_path / "artifacts", data_revision="rev1")

    for stage in STAGES:
        assert a.key(stage, config) != b.key(stage, config), stage


def test_path_lives_under_the_stage_and_key(tmp_path):
    s = store(tmp_path)
    config = load_experiment_text(BASE)

    path = s.path("document_index", config, "vectors.npy")

    assert path.parent.parent.name == "document_index"
    assert path.parent.name == s.key("document_index", config)
    assert path.name == "vectors.npy"
    assert s.root in path.parents


def test_asking_for_a_path_creates_nothing(tmp_path):
    """A cache miss must stay a miss; only a writer creates the directory."""
    s = store(tmp_path)
    config = load_experiment_text(BASE)

    path = s.path("document_index", config, "vectors.npy")

    assert not path.parent.exists()
    assert not s.exists("document_index", config)
    assert not s.exists("document_index", config, "vectors.npy")


def test_prepare_creates_the_directory_and_the_manifest(tmp_path):
    s = store(tmp_path)
    config = load_experiment_text(BASE)

    directory = s.prepare("document_index", config)

    assert directory.is_dir()
    assert (directory / "manifest.json").is_file()
    assert s.exists("document_index", config)


def test_manifest_records_the_configuration_that_produced_the_key(tmp_path):
    s = store(tmp_path)
    config = load_experiment_text(BASE)

    s.prepare("label_index", config)
    manifest = s.manifest("label_index", config)

    assert manifest["stage"] == "label_index"
    assert manifest["data_revision"] == "rev0"
    assert manifest["config"]["encoder"]["name"] == "intfloat/multilingual-e5-base"
    # Only the sections the stage depends on are recorded, so the manifest
    # explains the key rather than describing the whole experiment.
    assert "reranker" not in manifest["config"]


def test_unknown_stage_is_refused(tmp_path):
    s = store(tmp_path)
    with pytest.raises(UnknownStage, match="rerank"):
        s.key("rerank", load_experiment_text(BASE))
