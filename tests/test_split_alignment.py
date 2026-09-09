"""The all-subjects index is only safe if the alignment claim is checked.

Rung 3 indexes 38,545 documents from a second shared-task track. The claim that
makes that safe — the two tracks are split-aligned, so no held-out record is in
the larger training split — is checked here rather than believed, and the check
is a committed attestation the pipeline reads before it indexes anything.
"""

import json

import pytest

from llms4subjects.contracts import Record
from llms4subjects.splits import (
    ATTESTATION_SCHEMA,
    ConflictingRecords,
    SplitAlignmentUnverified,
    attest,
    clear_index,
    load_attestation,
    merge,
    overlap,
    require_alignment,
)


def record(id, subjects=(), title="t", type="Book", lang="de"):
    return Record(id=id, type=type, lang=lang, title=title, abstract="a",
                  subjects=tuple(subjects))


# --- Merging the index corpus ------------------------------------------------


def test_a_corpus_without_duplicates_passes_through_in_order():
    records = [record("a"), record("b"), record("c")]
    merged = merge(records)

    assert [r.id for r in merged.records] == ["a", "b", "c"]
    assert merged.rows == 3
    assert merged.duplicate_ids == 0
    assert merged.merged_subjects == 0


def test_a_record_filed_twice_is_indexed_once():
    """Otherwise one document votes twice in every neighbour harvest."""
    merged = merge([record("a", ["x"]), record("b"), record("a", ["x"])])

    assert [r.id for r in merged.records] == ["a", "b"]
    assert merged.rows == 3
    assert merged.duplicate_ids == 1
    assert merged.merged_subjects == 0


def test_copies_that_disagree_on_gold_keep_every_assignment():
    """Both rows are the release's own gold for one document, so both count."""
    merged = merge([record("a", ["x", "y"]), record("a", ["y", "z"])])

    assert merged.records[0].subjects == ("x", "y", "z")
    assert merged.merged_subjects == 1


def test_copies_that_disagree_on_the_document_itself_are_refused():
    """Two different documents under one id is a rebuild bug, not a duplicate."""
    with pytest.raises(ConflictingRecords, match="a"):
        merge([record("a", title="one"), record("a", title="another")])


def test_merging_is_what_makes_the_all_subjects_count_the_documented_one():
    """core_train ⊂ all_train, so naming both must not index the overlap twice."""
    core = [record(f"c{n}") for n in range(3)]
    everything = core + [record(f"x{n}") for n in range(2)]

    assert len(merge(core + everything).records) == 5


# --- Overlap against the held-out splits -------------------------------------


def test_overlap_counts_shared_record_ids():
    index = [record("a"), record("b")]
    assert overlap(index, {"core_test": {"b", "z"}}) == {"core_test": 1}


def test_no_overlap_is_the_answer_the_attestation_needs():
    index = [record("a"), record("b")]
    assert overlap(index, {"core_test": {"z"}}) == {"core_test": 0}


# --- The attestation ---------------------------------------------------------


def document(revision="rev1", corpora=("all_train",), overlaps=0):
    """An attestation over fabricated corpora, with fabricated digests.

    The digests are injected rather than computed so that these tests say what
    the attestation *means* — this corpus, checked in this state — without
    reading 100 MB of CSV to find out.
    """
    return attest(
        revision="dataset1",
        corpora={
            name: merge([record("a"), record("b")]) for name in corpora
        },
        held_out={"core_test": {"z"} if not overlaps else {"a"}},
        revisions={name: revision for name in corpora},
    )


def test_an_attestation_records_what_it_checked():
    written = document()

    assert written["schema"] == ATTESTATION_SCHEMA
    assert written["data_revision"] == "dataset1"
    assert written["corpora"]["all_train"]["revision"] == "rev1"
    assert written["corpora"]["all_train"]["records"] == 2
    assert written["held_out"]["core_test"]["records"] == 1
    assert written["overlaps"]["all_train"]["core_test"] == 0
    assert written["excluded"]["all_train"]["core_test"] == []


def written_to(tmp_path, **kwargs):
    path = tmp_path / "split_alignment.json"
    path.write_text(json.dumps(document(**kwargs)))
    return path


def test_a_clean_corpus_excludes_nothing(tmp_path):
    path = written_to(tmp_path)

    assert require_alignment(
        ("all_train",), path=path, revisions={"all_train": "rev1"}
    ) == frozenset()


def test_a_corpus_that_has_changed_since_the_check_is_refused(tmp_path):
    """The check was about one version of the file; this is another."""
    path = written_to(tmp_path, revision="rev1")

    with pytest.raises(SplitAlignmentUnverified, match="rev2"):
        require_alignment(
            ("all_train",), path=path, revisions={"all_train": "rev2"}
        )


def test_a_corpus_the_attestation_does_not_name_is_refused(tmp_path):
    path = written_to(tmp_path, corpora=("core_train",))

    with pytest.raises(SplitAlignmentUnverified, match="all_train"):
        require_alignment(
            ("all_train",), path=path, revisions={"all_train": "rev1"}
        )


def test_an_unrelated_corpus_changing_does_not_refuse_this_one(tmp_path):
    """A checkout that never built the all-subjects split still runs rung 2.

    The regression this pins: keyed on the dataset as a whole, the attestation
    refused every run on a checkout without `all_train.csv` — including the
    rung-1 and rung-2 configs that only ever index `core_train`, over a file
    they do not read and that checkout does not have.
    """
    path = tmp_path / "split_alignment.json"
    path.write_text(
        json.dumps(
            attest(
                revision="dataset1",
                corpora={
                    name: merge([record("a")]) for name in ("core_train", "all_train")
                },
                held_out={"core_test": {"z"}},
                revisions={"core_train": "same", "all_train": "gone"},
            )
        )
    )

    assert require_alignment(
        ("core_train",),
        path=path,
        revisions={"core_train": "same", "all_train": "different"},
    ) == frozenset()


def test_an_attested_overlap_comes_back_as_ids_to_drop(tmp_path):
    """The fix for a held-out record in the corpus is to index the others."""
    path = written_to(tmp_path, overlaps=1)

    assert require_alignment(
        ("all_train",), path=path, revisions={"all_train": "rev1"}
    ) == {"a"}


def test_the_excluded_records_are_the_ones_dropped():
    index = [record("a"), record("b")]

    assert [r.id for r in clear_index(index, {"a"})] == ["b"]
    assert [r.id for r in clear_index(index, ())] == ["a", "b"]


def test_a_missing_attestation_says_how_to_write_one(tmp_path):
    with pytest.raises(SplitAlignmentUnverified, match="verify_split_alignment"):
        require_alignment(("all_train",), path=tmp_path / "absent.json")


# --- The committed attestation -----------------------------------------------


def test_the_committed_attestation_clears_the_corpora_this_checkout_has():
    """Against the real files, so a stale reference fails here and not in a run."""
    from llms4subjects.corpus import MissingDataset
    from llms4subjects.splits import require_alignment as check

    for corpus in ("core_train", "all_train"):
        try:
            check((corpus,))
        except MissingDataset:
            pytest.skip(f"{corpus} is not built in this checkout")


def test_the_committed_attestation_covers_every_indexable_corpus():
    written = load_attestation()

    assert written["schema"] == ATTESTATION_SCHEMA
    for corpus in ("core_train", "all_train"):
        assert written["corpora"][corpus]["records"] > 0
        # The claim rung 3 rests on, checked rather than cited: the gold test
        # split is in neither index corpus.
        assert written["overlaps"][corpus]["core_test"] == 0


def test_the_all_subjects_split_carries_nine_dev_records():
    """Not the documented alignment, which is why it is checked before use."""
    written = load_attestation()

    assert written["overlaps"]["core_train"]["core_dev"] == 0
    assert written["overlaps"]["all_train"]["core_dev"] == 9
    assert len(written["excluded"]["all_train"]["core_dev"]) == 9


def test_the_committed_attestation_records_the_all_subjects_duplicates():
    """70,633 rows and 70,588 documents: the 45 are the release's, not a bug."""
    everything = load_attestation()["corpora"]["all_train"]

    assert everything["rows"] == 70633
    assert everything["records"] == 70588
    assert everything["duplicate_ids"] == 45
