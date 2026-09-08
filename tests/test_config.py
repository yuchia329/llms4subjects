"""Experiment configuration is data, so the loader is what has to be trusted."""

import textwrap
from pathlib import Path

import pytest

from llms4subjects.config import ConfigError, load_experiment, load_experiment_text

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_CONFIGS = [
    REPO_ROOT / "configs" / name
    for name in ("rung1.yaml", "rung2.yaml", "rung3.yaml")
]


@pytest.mark.parametrize("path", REPO_CONFIGS)
def test_committed_configs_load(path):
    config = load_experiment(path)
    assert config.name
    assert config.encoder.name


def test_rung_configs_differ_only_where_the_ladder_says():
    """The ladder varies index size only until rung 3 introduces training."""
    rung1, rung2, rung3 = (load_experiment(p) for p in REPO_CONFIGS)

    assert rung1.index.size == 8000
    assert rung2.index.size is None
    assert rung1.index.stratify is True

    assert rung1.encoder.adapter is None
    assert rung2.encoder.adapter is None
    assert rung3.encoder.adapter is not None

    assert rung1.label_text == rung2.label_text


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="rerankr"):
        load_experiment_text(
            textwrap.dedent(
                """
                name: typo
                encoder: {name: intfloat/multilingual-e5-base}
                rerankr: {enabled: true}
                """
            )
        )


def test_unknown_nested_key_is_rejected():
    with pytest.raises(ConfigError, match="definitoin"):
        load_experiment_text(
            textwrap.dedent(
                """
                name: typo
                encoder: {name: intfloat/multilingual-e5-base}
                label_text: {definitoin: true}
                """
            )
        )


def test_qualifier_mode_is_constrained_to_the_three_documented_modes():
    for mode in ("parenthetical", "raw", "stripped"):
        config = load_experiment_text(
            f"name: q\nencoder: {{name: e}}\nlabel_text: {{qualifiers: {mode}}}\n"
        )
        assert config.label_text.qualifiers == mode

    with pytest.raises(ConfigError, match="qualifiers"):
        load_experiment_text(
            "name: q\nencoder: {name: e}\nlabel_text: {qualifiers: paranthetical}\n"
        )


def test_unknown_corpus_is_rejected():
    with pytest.raises(ConfigError, match="core_trian"):
        load_experiment_text(
            "name: c\nencoder: {name: e}\nindex: {corpora: [core_trian]}\n"
        )


def test_the_gold_test_split_cannot_be_indexed():
    """It is opened once, at the end of the project, and never as index data."""
    with pytest.raises(ConfigError, match="test"):
        load_experiment_text(
            "name: c\nencoder: {name: e}\nindex: {corpora: [core_test]}\n"
        )


def test_rung3_indexes_the_all_subjects_split_once():
    config = load_experiment(REPO_ROOT / "configs" / "rung3.yaml")
    # all_train already contains every core_train record.
    assert config.index.corpora == ("all_train",)


def test_name_and_encoder_are_required():
    with pytest.raises(ConfigError, match="name"):
        load_experiment_text("encoder: {name: e}\n")
    with pytest.raises(ConfigError, match="encoder"):
        load_experiment_text("name: n\n")


def test_defaults_follow_the_spec_pipeline_contract():
    config = load_experiment_text("name: d\nencoder: {name: e}\n")

    assert config.fusion.candidates == 100
    assert config.reranker.output_k == 50
    assert config.adjudication.candidates == 30
    assert config.adjudication.route_fraction == pytest.approx(0.2)
    # Definition holds cataloguing instructions, so it is off unless ablated in.
    assert config.label_text.include_definition is False
    assert config.label_text.bilingual is True


def test_to_dict_round_trips_through_the_loader():
    original = load_experiment(REPO_ROOT / "configs" / "rung2.yaml")
    assert load_experiment_text(original.to_yaml()) == original
