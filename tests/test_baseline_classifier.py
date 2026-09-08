"""The archived classifier's output layer, and the zero it scores by construction.

Ticket 14 exists to turn "a dense output layer cannot serve this problem" into a
measurement. These tests cover the part of that measurement which is structural
rather than trained: the output layer is exactly the training label set in a
frozen order, a prediction is drawn from that layer and nothing else, and a gold
label the layer does not contain is therefore unreachable at every k.

The trained numbers come from a GPU run and land in docs/results.md. What is
asserted here is the property that makes those numbers interpretable, and it
needs no weights: if the zero-shot band recall were anything but zero, the
prediction path would have to be reaching outside its own output layer.
"""

import json

import numpy as np
import pytest

from baseline.labels import (
    load_labels,
    output_layer,
    rank_codes,
    save_labels,
    target_rows,
    unreachable_assignments,
)
from llms4subjects.contracts import CODES_PER_RECORD, Record
from llms4subjects.corpus import (
    MissingDataset,
    frequency_bands,
    load_split,
    load_vocabulary,
)
from llms4subjects.stages.evaluator import OFFICIAL_KS, evaluate


def record(record_id, *subjects, type="Book", lang="de"):
    return Record(
        id=record_id,
        type=type,
        lang=lang,
        title=f"title {record_id}",
        abstract=f"abstract {record_id}",
        subjects=tuple(subjects),
    )


FIXTURE = [
    record("a", "gnd:3", "gnd:1"),
    record("b", "gnd:2", "gnd:3"),
    record("c"),
]


def split(name):
    try:
        return load_split(name)
    except MissingDataset as error:
        pytest.skip(str(error))


# --- the output layer -------------------------------------------------------


def test_the_output_layer_is_the_training_label_set_in_sorted_order():
    assert output_layer(FIXTURE) == ("gnd:1", "gnd:2", "gnd:3")


def test_the_output_layer_is_the_same_order_every_time_it_is_built():
    """The defect this replaces: dev columns came from iterating a set.

    Set iteration order varies between processes for str keys, so the previous
    code's dev label indices did not line up with the train ones it had trained
    against — the columns were the same labels in a different order.
    """
    shuffled = list(reversed(FIXTURE))
    assert output_layer(shuffled) == output_layer(FIXTURE)


def test_the_output_layer_holds_no_duplicates():
    codes = output_layer(FIXTURE + FIXTURE)
    assert len(set(codes)) == len(codes)


def test_the_output_layer_round_trips_through_disk(tmp_path):
    path = tmp_path / "labels.json"
    save_labels(path, output_layer(FIXTURE))
    assert load_labels(path) == output_layer(FIXTURE)


def test_a_saved_output_layer_keeps_its_order_rather_than_being_re_sorted(tmp_path):
    """The head's columns are positional, so the saved order is authoritative.

    Re-deriving the order at load time would silently re-map every column if the
    layer were ever built differently, which is exactly the failure being fixed.
    """
    path = tmp_path / "labels.json"
    save_labels(path, ("gnd:9", "gnd:1"))
    assert load_labels(path) == ("gnd:9", "gnd:1")


def test_the_output_layer_is_the_training_split_label_set():
    train = split("core_train")
    codes = output_layer(train)
    assert len(codes) == len({code for r in train for code in r.subjects})
    # README and docs/spec.md both quote this width of the output layer.
    assert len(codes) == 14607


def test_every_output_layer_code_is_a_tib_core_vocabulary_code():
    """Predictions must stay inside the 79,427-code tib-core vocabulary."""
    codes = output_layer(split("core_train"))
    try:
        vocabulary = load_vocabulary("tib-core")
    except MissingDataset as error:
        pytest.skip(str(error))
    assert not set(codes) - set(vocabulary)


def test_target_rows_are_the_columns_the_output_layer_assigns():
    """The alignment the module exists to hold: gold code to its own column.

    Stated against `output_layer`'s own indexing rather than against literals,
    because the failure being prevented is the two disagreeing — a target on
    column 2 while the head learned that code on column 0.
    """
    codes = output_layer(FIXTURE)
    rows = target_rows(FIXTURE, codes)

    for record, row in zip(FIXTURE, rows):
        assert {codes[column] for column in row} == set(record.subjects)


def test_target_rows_are_sorted_and_hold_one_column_per_gold_code():
    codes = output_layer(FIXTURE)
    for row in target_rows(FIXTURE, codes):
        assert list(row) == sorted(row)
        assert len(set(row)) == len(row)


def test_target_rows_of_a_record_with_no_gold_are_empty():
    assert target_rows([record("f")], output_layer(FIXTURE)) == [()]


# --- what the layer cannot represent ---------------------------------------


def test_unreachable_assignments_counts_gold_outside_the_output_layer():
    codes = output_layer(FIXTURE)
    held_out = [record("d", "gnd:1", "gnd:99")]
    total, unreachable = unreachable_assignments(held_out, codes)
    assert (total, unreachable) == (2, 1)


def test_unreachable_assignments_ignores_records_with_no_gold():
    assert unreachable_assignments([record("e")], output_layer(FIXTURE)) == (0, 0)


# --- the prediction contract ------------------------------------------------


def test_rank_codes_returns_the_codes_of_the_highest_scores_in_order():
    assert rank_codes([0.1, 0.9, 0.5], ("gnd:1", "gnd:2", "gnd:3"), k=2) == (
        "gnd:2",
        "gnd:3",
    )


def test_rank_codes_returns_fifty_distinct_codes_by_default():
    codes = tuple(f"gnd:{i}" for i in range(200))
    scores = np.random.default_rng(0).normal(size=len(codes))
    ranked = rank_codes(scores, codes)
    assert len(ranked) == CODES_PER_RECORD
    assert len(set(ranked)) == CODES_PER_RECORD


def test_rank_codes_is_non_increasing_in_score():
    codes = tuple(f"gnd:{i}" for i in range(200))
    scores = np.random.default_rng(1).normal(size=len(codes))
    by_code = dict(zip(codes, scores))
    ranked = rank_codes(scores, codes)
    assert list(ranked) == sorted(ranked, key=lambda code: -by_code[code])


def test_rank_codes_only_ever_returns_output_layer_codes():
    codes = tuple(f"gnd:{i}" for i in range(60))
    scores = np.random.default_rng(2).normal(size=len(codes))
    assert not set(rank_codes(scores, codes)) - set(codes)


def test_rank_codes_refuses_scores_from_a_different_output_layer():
    """A silent mismatch would file each score under the wrong code."""
    with pytest.raises(ValueError):
        rank_codes([0.1, 0.2, 0.3], ("gnd:1", "gnd:2"))


def test_rank_codes_cannot_return_more_codes_than_the_layer_holds():
    ranked = rank_codes([0.2, 0.4], ("gnd:1", "gnd:2"))
    assert ranked == ("gnd:2", "gnd:1")


# --- the measurement the ticket exists for ---------------------------------


def test_the_zero_shot_band_recall_is_zero_whatever_the_classifier_scores():
    """A closed output layer cannot reach a label it has no column for.

    Built from the real training split and the frozen bands, with the scores
    random: no training run can change this, which is why the row belongs in
    the results table as a structural zero rather than as a weak number.
    """
    train = split("core_train")
    dev = split("core_dev")
    codes = output_layer(train)
    bands = frequency_bands()

    rng = np.random.default_rng(3)
    predictions = {
        r.id: rank_codes(rng.normal(size=len(codes)), codes) for r in dev[:200]
    }
    gold = {r.id: r.subjects for r in dev[:200]}
    cells = {r.id: (r.type, r.lang) for r in dev[:200]}

    report = evaluate(gold, predictions, bands, cells)

    zero = report.by_band["zero"]
    assert zero.assignments > 0, "the slice has to be non-empty to be a measurement"
    assert all(zero.recall(k) == 0.0 for k in OFFICIAL_KS)
    # And the other bands are reachable, so the zero is about the band and not
    # about the wiring of this test.
    assert sum(report.by_band[band].assignments for band in ("head", "torso", "tail"))


def test_no_zero_band_label_has_a_column_in_the_output_layer():
    """The same fact stated at the layer rather than through the metric."""
    codes = set(output_layer(split("core_train")))
    bands = frequency_bands()
    assert not {code for code in codes if bands.get(code, "zero") == "zero"}


# --- what a run has to record ----------------------------------------------


def test_a_training_run_records_its_configuration_and_wall_clock(tmp_path):
    from baseline.train import training_summary

    summary = training_summary(
        run="mbert-dense",
        device="cuda",
        codes=output_layer(FIXTURE),
        records=2,
        args_dict={"epochs": 15, "batch_size": 64, "lr": 2e-5, "max_length": 256},
        epochs=[{"epoch": 1, "train_loss": 0.1, "dev_loss": 0.2, "seconds": 3.0}],
        seconds=3.0,
    )

    for key in ("run", "model", "device", "labels", "records", "config", "seconds"):
        assert key in summary, key
    assert summary["config"]["epochs"] == 15
    assert summary["seconds"] == 3.0
    assert summary["graph_component"]["revived"] is False
    assert summary["graph_component"]["reason"]
    # Round-trips as JSON, since that is how it reaches the results document.
    assert json.loads(json.dumps(summary)) == summary
