"""Rung 3's two deliverables, at the seams that do not need an encoder.

The report answers two questions. Whether fine-tuning moved the score, and by
how much per band — arithmetic over two scored runs. And band migration: how
many of the labels the frozen bands call zero-shot the larger corpus turns into
seen ones. The second is the project's sharpest finding, so what is asserted
here is that it is computed against the frozen boundaries and never against
recomputed bands.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from llms4subjects.config import load_experiment_text


def _script(name: str):
    """`scripts/` is not importable as a package, so it is loaded by path."""
    import sys

    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


report = _script("rung3_report")
check_pairs = report.check_pairs
migration = report.migration
unreachable = report.unreachable
NotAComparison = report.NotAComparison

BOUNDARIES = [
    {"band": "head", "min": 101, "max": None},
    {"band": "torso", "min": 10, "max": 100},
    {"band": "tail", "min": 1, "max": 9},
    {"band": "zero", "min": 0, "max": 0},
]


def config(name="c", adapter=None, encoder="BAAI/bge-m3", corpora="all_train"):
    adapter_line = f", adapter: {adapter}" if adapter else ""
    return load_experiment_text(
        f"name: {name}\n"
        f"encoder: {{name: {encoder}{adapter_line}}}\n"
        f"index: {{corpora: [{corpora}]}}\n"
    )


# --- One variable between the two rows ---------------------------------------


def test_a_pair_that_differs_only_in_its_adapter_is_a_comparison():
    check_pairs([(config(), config(adapter="artifacts/adapters/a"))])


def test_the_untrained_row_must_actually_be_untrained():
    with pytest.raises(NotAComparison, match="adapter"):
        check_pairs(
            [(config(adapter="artifacts/adapters/a"), config(adapter="b"))]
        )


def test_the_trained_row_must_actually_be_trained():
    with pytest.raises(NotAComparison, match="adapter"):
        check_pairs([(config(), config())])


def test_a_pair_over_two_models_measures_the_model_not_the_training():
    with pytest.raises(NotAComparison, match="gte-multilingual-base"):
        check_pairs(
            [
                (
                    config(),
                    config(
                        encoder="Alibaba-NLP/gte-multilingual-base",
                        adapter="artifacts/adapters/a",
                    ),
                )
            ]
        )


def test_a_pair_at_two_index_sizes_measures_the_index_not_the_training():
    with pytest.raises(NotAComparison, match="index"):
        check_pairs(
            [(config(corpora="core_train"), config(adapter="a"))]
        )


# --- Band migration ----------------------------------------------------------


def test_a_label_the_larger_corpus_reaches_leaves_the_zero_band():
    moved = migration(
        labels=["z"],
        frozen={},
        counts={"z": 3},
        boundaries=BOUNDARIES,
    )

    assert moved == {"zero": {"tail": 1}}


def test_a_label_the_larger_corpus_still_does_not_reach_stays_zero():
    moved = migration(
        labels=["z"], frozen={}, counts={}, boundaries=BOUNDARIES
    )

    assert moved == {"zero": {"zero": 1}}


def test_migration_is_reported_from_the_frozen_band_never_a_recomputed_one():
    """A tail label with 200 occurrences now is still a frozen-tail label."""
    moved = migration(
        labels=["t"], frozen={"t": 4}, counts={"t": 200}, boundaries=BOUNDARIES
    )

    assert moved == {"tail": {"head": 1}}


def test_labels_can_be_weighted_by_what_they_are_worth():
    """One vote per label answers a different question from one per assignment."""
    moved = migration(
        labels=["z1", "z2"],
        frozen={},
        counts={"z1": 5},
        boundaries=BOUNDARIES,
        weights={"z1": 1, "z2": 9},
    )

    assert moved == {"zero": {"tail": 1, "zero": 9}}


def test_what_stays_unreachable_is_the_number_the_finding_rests_on():
    moved = {"zero": {"tail": 180, "zero": 812}}

    assert unreachable(moved) == 812
    assert unreachable({"zero": {"tail": 5}}) == 0


# --- The tables --------------------------------------------------------------


def scored_row(config, ranking, ks=(5, 10, 100)):
    """One `Screened` row over a two-record toy split, through the evaluator."""
    from llms4subjects.stages.evaluator import evaluate

    gold = {"r1": ["h", "z"], "r2": ["t"]}
    scored_report = evaluate(
        gold=gold,
        predictions={"r1": ranking, "r2": ranking},
        bands={"h": "head", "t": "tail"},
        cells={"r1": ("Book", "de"), "r2": ("Article", "en")},
        ks=ks,
    )
    return report.Screened(
        config=config,
        dimensions=1024,
        parameters=None,
        fused=scored_report,
        per_retriever={"knn": scored_report, "dense": scored_report},
        load_seconds=1.0,
        retrieve_seconds=2.0,
        wrote=0,
    )


def a_pair():
    untrained = scored_row(config(name="rung3-bge-m3"), ["a", "b"])
    trained = scored_row(
        config(name="rung3-bge-m3-finetuned", adapter="artifacts/adapters/a"),
        ["h", "t", "z"],
    )
    return [(untrained, trained)]


def test_the_headline_table_carries_a_row_for_what_training_added():
    rendered = report.render_training(a_pair(), 10)

    assert "off the shelf" in rendered
    assert "fine-tuned" in rendered
    assert "what training added" in rendered
    # A gain is signed, so a reader never has to subtract two rows themselves.
    assert "| +1.0000 |" in rendered


def test_the_band_table_names_every_frozen_band():
    rendered = report.render_bands(a_pair(), 10)

    for band in ("head", "torso", "tail", "zero"):
        assert band in rendered
    assert "delta" in rendered


def test_the_retriever_table_shows_both_states_of_each_retriever():
    rendered = report.render_per_retriever(a_pair(), 10)

    assert "knn" in rendered and "dense" in rendered
    assert rendered.count("| off the shelf |") == 1
    assert rendered.count("| fine-tuned |") == 1


def test_attribution_needs_the_encoder_in_the_earlier_screen():
    """A rung-2 screen without this encoder is a missing row, not a crash."""

    class Empty:
        path = "reference/screens/rung2.json"
        data_revision = "rev0"

        def row(self, name):
            raise KeyError(name)

    rendered = report.render_attribution(a_pair(), Empty(), 10)

    assert "| — | — | — | — | — |" in rendered


def test_the_rung3_document_is_not_a_screen(tmp_path):
    """Two rows share a model name here, and a screen has one row per model."""
    written = report.document(
        a_pair(),
        split="core_dev",
        records=2,
        revision="rev1",
        device="mps",
        rows_read=70633,
        indexed=70579,
        dropped=9,
    )

    assert written["schema"] == report.SCHEMA
    assert written["index"]["documents"] == 70579
    assert written["index"]["dropped_held_out"] == 9
    assert [pair["encoder"] for pair in written["pairs"]] == ["BAAI/bge-m3"]

    path = tmp_path / "rung3-report.json"
    path.write_text(json.dumps(written))
    compare = _script("compare_rungs")
    with pytest.raises(compare.NotAComparison, match="schema"):
        compare.load_screen(path)


# --- The guards on a report ---------------------------------------------------


def test_a_second_pair_at_another_index_is_refused():
    """Every row is scored against the first pair's corpus and gold."""
    with pytest.raises(NotAComparison, match="one index"):
        check_pairs(
            [
                (config(), config(adapter="a")),
                (
                    config(encoder="Alibaba-NLP/gte-multilingual-base",
                           corpora="core_train"),
                    config(encoder="Alibaba-NLP/gte-multilingual-base",
                           corpora="core_train", adapter="b"),
                ),
            ]
        )


def test_two_pairs_that_differ_only_in_their_encoders_are_a_report():
    check_pairs(
        [
            (config(), config(adapter="a")),
            (
                config(encoder="Alibaba-NLP/gte-multilingual-base"),
                config(encoder="Alibaba-NLP/gte-multilingual-base", adapter="b"),
            ),
        ]
    )


def test_an_earlier_screen_of_another_split_is_refused():
    """"The index adds N" is only that if both columns scored the same split."""

    class Screen:
        path = "reference/screens/rung2.json"
        split = "core_train"

        def row(self, name):
            raise KeyError(name)

    with pytest.raises(NotAComparison, match="different questions"):
        report.check_against(Screen(), [(config(), config(adapter="a"))], "core_dev")


def test_an_earlier_screen_taken_under_another_configuration_is_refused():
    class Row:
        held_equal = {"fusion": {"candidates": 30, "rrf_k": 60}}

    class Screen:
        path = "reference/screens/rung2.json"
        split = "core_dev"

        def row(self, name):
            return Row()

    with pytest.raises(NotAComparison, match="fusion"):
        report.check_against(Screen(), [(config(), config(adapter="a"))], "core_dev")
