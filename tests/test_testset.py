"""Opening the gold test split once, and the two things that discipline needs.

The rule is stated in docs/spec.md (story 3) and enforced everywhere else by
refusal: `scripts/run_experiment.py` will not score `core_test`, and
`IndexConfig` will not index it. Ticket 17 is the one run that does open it, so
the discipline it needs is different in kind — not "never", but "once, under a
configuration nobody could have chosen after seeing the answer".

That is two mechanisms and they are tested separately:

- a **plan**, committed before the split is read, naming every configuration the
  run will score by a digest of the sections that decide what it predicts; and
- a **receipt**, written after, so a second run is a refusal that names the
  first rather than a number quietly replacing another number.

The third thing here is the caveat the spec commits to reporting: the test
records whose title and abstract exactly duplicate an indexed training record,
on which a neighbour retriever is unusually strong.
"""

import json
from datetime import date

import pytest

from llms4subjects.config import load_experiment_text
from llms4subjects.contracts import Cell, Record
from llms4subjects.testset import (
    PLAN_SCHEMA,
    RECEIPT_SCHEMA,
    SplitAlreadyRead,
    RunUnplanned,
    configuration_digest,
    duplicated_from,
    load_plan,
    plan,
    receipt,
    require_plan,
    require_unread,
    unreadable_cells,
)

CONFIG = """
name: whichever
encoder:
  name: Alibaba-NLP/gte-multilingual-base
index:
  corpora: [all_train]
fusion:
  candidates: 100
  rrf_k: 60
reranker:
  enabled: true
  model: BAAI/bge-reranker-base
  mix: fuse
"""


def config(text: str = CONFIG):
    return load_experiment_text(text)


def record(id, title="t", abstract="a", type="Book", lang="de", subjects=()):
    return Record(
        id=id, type=type, lang=lang, title=title, abstract=abstract,
        subjects=tuple(subjects),
    )


# --- The digest: what counts as the same configuration ------------------------


def test_a_comment_or_a_rename_is_the_same_configuration():
    """The plan fixes what a run predicts, not what the file is called.

    Comments never reach the loader at all, and `name` and `notes` are prose
    about the row rather than inputs to it. A digest that moved when either did
    would make the refusal fire on a typo fix and be turned off by the next
    person who hit it.
    """
    renamed = CONFIG.replace("name: whichever", "name: something-else") + (
        "\nnotes: >\n  A sentence about the row.\n"
    )

    assert configuration_digest(config(renamed)) == configuration_digest(config())


def test_a_changed_pipeline_field_is_a_different_configuration():
    swapped = CONFIG.replace("mix: fuse", "mix: replace")

    assert configuration_digest(config(swapped)) != configuration_digest(config())


def test_the_digest_covers_every_stage_the_prediction_passes_through():
    """Not just the sections one stage reads: the whole pipeline decides a row."""
    for edit, replacement in (
        ("name: Alibaba-NLP/gte-multilingual-base", "name: BAAI/bge-m3"),
        ("corpora: [all_train]", "corpora: [core_train]"),
        ("rrf_k: 60", "rrf_k: 30"),
    ):
        assert configuration_digest(
            config(CONFIG.replace(edit, replacement))
        ) != configuration_digest(config()), edit


# --- The plan: fixed before the split is read ---------------------------------


def written_plan(tmp_path, rows=None, **kwargs):
    document = plan(
        headline="headline",
        rows=rows if rows is not None else {"headline": config()},
        fixed=date(2026, 9, 9),
        **kwargs,
    )
    path = tmp_path / "test_plan.json"
    path.write_text(json.dumps(document, indent=2))
    return path


def test_a_plan_records_the_digest_of_every_row_it_names(tmp_path):
    document = load_plan(written_plan(tmp_path))

    assert document["schema"] == PLAN_SCHEMA
    assert document["headline"] == "headline"
    assert document["fixed"] == "2026-09-09"
    assert document["rows"]["headline"]["digest"] == configuration_digest(config())


def test_a_plan_names_the_config_file_of_every_row(tmp_path):
    """A digest says two runs agree; the path says which file to read."""
    document = plan(
        headline="headline",
        rows={"headline": config()},
        paths={"headline": "configs/test.yaml"},
        fixed=date(2026, 9, 9),
    )

    assert document["rows"]["headline"]["config"] == "configs/test.yaml"


def test_the_headline_must_be_one_of_the_rows():
    with pytest.raises(ValueError, match="headline"):
        plan(headline="absent", rows={"headline": config()}, fixed=date(2026, 9, 9))


def test_running_without_a_plan_is_refused_and_says_how_to_write_one(tmp_path):
    with pytest.raises(RunUnplanned, match="fix-plan"):
        require_plan({"headline": config()}, path=tmp_path / "nothing.json")


def test_a_row_the_plan_does_not_name_is_refused(tmp_path):
    path = written_plan(tmp_path)

    with pytest.raises(RunUnplanned, match="extra"):
        require_plan({"headline": config(), "extra": config()}, path=path)


def test_a_row_whose_configuration_moved_since_the_plan_is_refused(tmp_path):
    """The whole point: the configuration is fixed before the answer is seen."""
    path = written_plan(tmp_path)
    edited = config(CONFIG.replace("mix: fuse", "mix: replace"))

    with pytest.raises(RunUnplanned, match="changed since"):
        require_plan({"headline": edited}, path=path)


def test_a_planned_row_that_is_not_run_is_not_a_refusal(tmp_path):
    """A blocked row — no API key, say — leaves the rest of the run scoreable."""
    path = written_plan(tmp_path, rows={"headline": config(), "adjudicated": config()})

    assert require_plan({"headline": config()}, path=path)["headline"] == "headline"


def test_the_matching_plan_is_returned_for_the_report_to_quote(tmp_path):
    document = require_plan({"headline": config()}, path=written_plan(tmp_path))

    assert document["fixed"] == "2026-09-09"


# --- The receipt: read exactly once -------------------------------------------


def test_the_first_run_is_allowed_and_needs_no_justification(tmp_path):
    assert require_unread(path=tmp_path / "test_run.json") == ()


def test_a_second_run_is_refused_and_names_the_first(tmp_path):
    path = tmp_path / "test_run.json"
    path.write_text(json.dumps(receipt(read=date(2026, 9, 9), under={}, rows={})))

    with pytest.raises(SplitAlreadyRead, match="2026-09-09"):
        require_unread(path=path)


def test_a_second_run_with_a_justification_is_allowed_and_carries_it(tmp_path):
    path = tmp_path / "test_run.json"
    path.write_text(json.dumps(receipt(read=date(2026, 9, 9), under={}, rows={})))

    assert require_unread(path=path, justification="the scorer was misread") == (
        {"read": "2026-09-09", "justification": "the scorer was misread"},
    )


def test_a_receipt_records_what_was_read_and_under_which_plan(tmp_path):
    document = receipt(
        read=date(2026, 9, 9),
        under={"fixed": "2026-09-08", "headline": "headline"},
        rows={"headline": {"micro": {"10": 0.5}}},
        records=4910,
    )

    assert document["schema"] == RECEIPT_SCHEMA
    assert document["read"] == "2026-09-09"
    assert document["plan"]["fixed"] == "2026-09-08"
    assert document["records"] == 4910
    assert document["earlier"] == ()


def test_earlier_reads_accumulate_rather_than_being_overwritten(tmp_path):
    path = tmp_path / "test_run.json"
    path.write_text(json.dumps(receipt(read=date(2026, 9, 9), under={}, rows={})))
    earlier = require_unread(path=path, justification="a fixed scorer bug")

    document = receipt(read=date(2026, 9, 10), under={}, rows={}, earlier=earlier)

    assert document["earlier"] == (
        {"read": "2026-09-09", "justification": "a fixed scorer bug"},
    )


# --- The duplicate caveat -----------------------------------------------------


def test_a_test_record_duplicating_an_indexed_document_is_named():
    duplicates = duplicated_from(
        [record("test-1", title="Polymers", abstract="A study.")],
        [record("train-1", title="Polymers", abstract="A study.")],
    )

    assert duplicates == ("test-1",)


def test_the_duplicate_must_carry_a_different_identifier():
    """A record matching itself is the same document, not a leaked one."""
    same = record("shared", title="Polymers", abstract="A study.")

    assert duplicated_from([same], [same]) == ()


def test_the_match_is_on_title_and_abstract_together():
    index = [record("train-1", title="Polymers", abstract="A study.")]

    assert duplicated_from(
        [record("test-1", title="Polymers", abstract="Another study.")], index
    ) == ()
    assert duplicated_from(
        [record("test-1", title="Polymer", abstract="A study.")], index
    ) == ()


def test_the_match_is_exact_rather_than_normalised():
    """The caveat is about identical records, and a looser rule is a new claim."""
    index = [record("train-1", title="Polymers", abstract="A study.")]

    assert duplicated_from(
        [record("test-1", title=" polymers ", abstract="A study.")], index
    ) == ()


def test_two_test_records_duplicating_each_other_are_not_duplicates_of_training():
    """Nothing is leaked by a pair of held-out records; only the index leaks."""
    pair = [
        record("test-1", title="Polymers", abstract="A study."),
        record("test-2", title="Polymers", abstract="A study."),
    ]

    assert duplicated_from(pair, []) == ()


def test_duplicates_come_back_sorted_so_the_set_is_comparable_between_runs():
    index = [record("train-1", title="t", abstract="a")]
    records = [record("b"), record("a"), record("c")]

    assert duplicated_from(records, index) == ("a", "b", "c")


# --- What the official scorer cannot read -------------------------------------


def test_a_language_outside_de_and_en_is_named_rather_than_scored():
    """Their reader seeds its result with `de` and `en` and indexes into it.

    `tests/test_submission.py` holds the `KeyError` itself. This is the harness
    half: the run has to know which records it may hand the official scorer, so
    that the rest are reported by the local evaluator instead of crashing it.
    """
    unreadable = unreadable_cells(
        {"a": ("Book", "en"), "b": ("Book", "fr"), "c": ("Thesis", "ja")}
    )

    assert unreadable == {
        Cell("Book", "fr"): ("b",),
        Cell("Thesis", "ja"): ("c",),
    }


def test_a_record_type_outside_the_five_is_named_too():
    assert unreadable_cells({"a": ("Dataset", "en")}) == {
        Cell("Dataset", "en"): ("a",)
    }


def test_a_split_the_official_scorer_can_read_whole_leaves_nothing_named():
    assert unreadable_cells({"a": ("Book", "de"), "b": ("Article", "en")}) == {}


def test_a_receipt_carries_the_split_facts_the_caveats_are_read_from(tmp_path):
    """So the duplicate rows are re-derivable without re-opening the split."""
    document = receipt(
        read=date(2026, 9, 9),
        under={},
        rows={},
        facts={"duplicates": {"indexed": ["a"], "core_train": ["a"]}},
    )

    assert document["facts"]["duplicates"]["core_train"] == ["a"]
