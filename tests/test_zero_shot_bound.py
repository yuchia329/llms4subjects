"""The bound the writeup leads with, at the seams that need no encoder.

Ticket 18 puts one number at the top of the results document: the share of this
benchmark's gold assignments that no document-similarity method can reach at
any corpus size, because the labels carrying them appear on no document. It is
a property of the data rather than of a run, so it is derived by arithmetic over
three committed things — the frozen bands, the corpus a config indexes, and a
split's gold — and what is asserted here is that the arithmetic reads the frozen
bands, counts what the score is actually made of, and cannot be run against the
gold test split before ticket 17's receipt says the split was opened.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from llms4subjects.contracts import Record
from llms4subjects.testset import SplitUnread, require_read


def _script(name: str):
    """`scripts/` is not importable as a package, so it is loaded by path."""
    import sys

    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


script = _script("zero_shot_bound")
bound = script.bound
render = script.render
document = script.document
require_readable = script.require_readable
SCHEMA = script.SCHEMA

BOUNDARIES = [
    {"band": "head", "min": 101, "max": None},
    {"band": "torso", "min": 10, "max": 100},
    {"band": "tail", "min": 1, "max": 9},
    {"band": "zero", "min": 0, "max": 0},
]


def record(identifier: str, *subjects: str) -> Record:
    return Record(
        id=identifier,
        type="Book",
        lang="de",
        title=identifier,
        abstract="",
        subjects=subjects,
    )


def measure(gold, index, frozen=None):
    return bound(
        gold_records=gold,
        index_records=index,
        frozen=frozen or {},
        boundaries=BOUNDARIES,
    )


# --- What the bound counts ---------------------------------------------------


def test_a_zero_shot_label_some_indexed_document_carries_is_reached():
    """One document carrying it is enough: kNN can propose it, rarely."""
    measured = measure(gold=[record("t", "z")], index=[record("i", "z")])

    assert measured.labels == 1
    assert measured.reached == 1
    assert measured.unreachable == 0


def test_a_zero_shot_label_no_document_carries_is_the_bound():
    measured = measure(gold=[record("t", "z")], index=[record("i", "other")])

    assert measured.reached == 0
    assert measured.unreachable == 1
    assert measured.assignments_unreachable == 1


def test_the_bound_reads_the_frozen_band_and_never_the_corpus_counts():
    """A label the corpus carries 200 times is still frozen-tail, not head."""
    measured = measure(
        gold=[record("t", "seen")],
        index=[record(f"i{n}", "seen") for n in range(200)],
        frozen={"seen": 4},
    )

    assert measured.labels == 0, "a frozen-tail label is not zero-shot"
    assert measured.unreachable == 0


def test_labels_are_weighted_by_the_gold_assignments_they_carry():
    """Two records sharing an unreachable label is two assignments, one label."""
    measured = measure(
        gold=[record("t1", "z"), record("t2", "z")], index=[record("i", "other")]
    )

    assert measured.unreachable == 1
    assert measured.assignments_unreachable == 2


def test_the_share_is_of_every_gold_assignment_not_only_the_zero_shot_ones():
    measured = measure(
        gold=[record("t", "z", "seen")],
        index=[record("i", "seen")],
        frozen={"seen": 4},
    )

    assert measured.total_assignments == 2
    assert measured.share == pytest.approx(0.5)


def test_reached_labels_are_reported_by_the_band_they_land_in():
    """A label one document carries lands in tail, which is what +0.02 was worth."""
    measured = measure(gold=[record("t", "z")], index=[record("i", "z")])

    assert measured.landed == {"tail": 1}


def test_a_split_whose_gold_is_all_seen_has_no_bound_and_says_so():
    measured = measure(
        gold=[record("t", "seen")], index=[record("i", "seen")], frozen={"seen": 4}
    )

    assert measured.labels == 0
    assert measured.share == 0.0


def test_the_bound_names_the_labels_it_counted_so_a_run_can_be_scored_on_them():
    measured = measure(
        gold=[record("t", "z", "reachable")], index=[record("i", "reachable")]
    )

    assert measured.unreachable_labels == ("z",)
    assert measured.reached_labels == ("reachable",)


# --- What the run reached on them --------------------------------------------


def test_recall_counts_only_the_assignments_the_named_labels_carry():
    hits, assignments = script.recall_over(
        labels=("z",),
        gold_records=[record("t", "z", "seen")],
        predictions={"t": ["z", "seen"]},
        k=10,
    )

    assert (hits, assignments) == (1, 1)


def test_a_label_below_k_is_not_a_hit():
    hits, assignments = script.recall_over(
        labels=("z",),
        gold_records=[record("t", "z")],
        predictions={"t": ["other", "z"]},
        k=1,
    )

    assert (hits, assignments) == (0, 1)


def test_a_record_the_submission_has_no_file_for_scores_zero_rather_than_vanishing():
    """The denominator is the split's gold, not the tree's coverage."""
    hits, assignments = script.recall_over(
        labels=("z",), gold_records=[record("t", "z")], predictions={}, k=10
    )

    assert (hits, assignments) == (0, 1)


def test_a_submission_tree_is_read_as_one_ranked_list_per_record(tmp_path):
    leaf = tmp_path / "Book" / "de"
    leaf.mkdir(parents=True)
    (leaf / "3A123.json").write_text(json.dumps({"dcterms:subject": ["a", "b"]}))

    assert script.read_submission(tmp_path) == {"3A123": ["a", "b"]}


def test_the_rendering_reports_what_a_run_reached_on_the_bound():
    measured = measure(gold=[record("t", "z")], index=[record("i", "other")])

    text = render(
        measured,
        split="core_test",
        corpora=["all_train"],
        documents=1,
        gold_records=[record("t", "z")],
        predictions={"t": ["z"]},
    )

    assert "1.000" in text or "100.0%" in text


# --- The gold split, which this derives from rather than opens ---------------


def test_the_gold_split_is_refused_while_the_receipt_says_it_is_unread(tmp_path):
    with pytest.raises(SplitUnread, match="core_test"):
        require_readable("core_test", tmp_path / "test_run.json")


def test_the_gold_split_is_derivable_once_a_receipt_records_the_read(tmp_path):
    receipt = tmp_path / "test_run.json"
    receipt.write_text(json.dumps({"schema": "test-receipt/1", "read": "2026-09-09"}))

    require_readable("core_test", receipt)


def test_any_other_split_needs_no_receipt(tmp_path):
    require_readable("core_dev", tmp_path / "missing.json")


def test_the_refusal_names_the_run_that_would_be_the_read(tmp_path):
    with pytest.raises(SplitUnread, match="final_test.py"):
        require_read(tmp_path / "test_run.json")


def test_the_receipt_is_returned_so_the_derivation_can_cite_the_read(tmp_path):
    receipt = tmp_path / "test_run.json"
    receipt.write_text(json.dumps({"schema": "test-receipt/1", "read": "2026-09-09"}))

    assert require_read(receipt)["read"] == "2026-09-09"


# --- What it writes ----------------------------------------------------------


def test_the_rendering_states_the_derivation_rather_than_the_number_alone():
    measured = measure(
        gold=[record("t", "z"), record("t2", "seen")],
        index=[record("i", "seen")],
        frozen={"seen": 4},
    )

    text = render(measured, split="core_test", corpora=["all_train"], documents=1)

    assert "1" in text and "core_test" in text
    assert "all_train" in text, "the corpus the bound is against is part of it"


def test_the_rendering_names_every_band_a_reached_label_landed_in():
    measured = measure(gold=[record("t", "z")], index=[record("i", "z")])

    assert "tail" in render(
        measured, split="core_dev", corpora=["all_train"], documents=1
    )


def test_the_document_records_what_it_was_derived_under():
    measured = measure(gold=[record("t", "z")], index=[record("i", "other")])

    written = document(
        measured,
        split="core_test",
        config="configs/test.yaml",
        corpora=["all_train"],
        documents=70_579,
        revision="abc123",
        read="2026-09-09",
    )

    assert written["schema"] == SCHEMA
    assert written["split"] == "core_test"
    assert written["config"] == "configs/test.yaml"
    assert written["index"] == {"documents": 70_579, "corpora": ["all_train"]}
    assert written["data_revision"] == "abc123"
    assert written["read"] == "2026-09-09"
    assert written["bound"]["labels"] == 1
    assert written["bound"]["unreachable"] == 1


def test_the_document_carries_the_share_the_writeup_quotes():
    measured = measure(
        gold=[record("t", "z", "seen")], index=[record("i", "seen")], frozen={"seen": 4}
    )

    written = document(
        measured,
        split="core_test",
        config="configs/test.yaml",
        corpora=["all_train"],
        documents=1,
        revision="abc123",
        read="",
    )

    assert written["bound"]["share_of_gold_assignments"] == pytest.approx(0.5)
    assert written["bound"]["total_assignments"] == 2


def test_the_document_carries_what_a_run_reached_when_it_was_given_one():
    measured = measure(gold=[record("t", "z")], index=[record("i", "other")])

    written = document(
        measured,
        split="core_test",
        config="configs/test.yaml",
        corpora=["all_train"],
        documents=1,
        revision="abc123",
        read="2026-09-09",
        gold_records=[record("t", "z")],
        predictions={"t": ["z"]},
        submission="artifacts/test/headline/submission",
    )

    assert written["reached"]["submission"] == "artifacts/test/headline/submission"
    assert written["reached"]["unreachable"]["10"] == pytest.approx(1.0)


def test_a_document_without_a_run_says_nothing_about_one():
    measured = measure(gold=[record("t", "z")], index=[record("i", "other")])

    written = document(
        measured,
        split="core_test",
        config="configs/test.yaml",
        corpora=["all_train"],
        documents=1,
        revision="abc123",
        read="2026-09-09",
    )

    assert "reached" not in written
