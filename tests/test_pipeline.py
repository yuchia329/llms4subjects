"""Seam 1: `predict(records, config) -> ranked candidate lists`.

Every assertion here is a contract invariant that must survive an encoder swap,
a new retriever or a reranker being switched on, because everything below this
call is defined by external model weights. Nothing asserts a score or a
particular code.

The invariant this file exists for is the last one: no gold label of an input
record may influence its own candidate set. An earlier draft in this repository
built its candidate universe from the evaluation split's own gold subjects, and
any number from that construction is meaningless.
"""

import dataclasses

import pipeline_fixture as corpus
import pytest

from llms4subjects.artifacts import ArtifactStore
from llms4subjects.config import load_experiment_text
from llms4subjects.contracts import CODES_PER_RECORD
from llms4subjects.pipeline import predict

# kNN only: the dense and lexical retrievers arrive with tickets 05 and 06.
KNN_ONLY = """
name: fixture-knn
encoder: {name: fixture/fake}
index: {corpora: [core_train]}
retrievers:
  knn: {enabled: true, neighbours: 5, top_k: 100}
  dense: {enabled: false}
  lexical: {enabled: false}
"""


def config(extra: str = "") -> object:
    return load_experiment_text(KNN_ONLY + extra)


@pytest.fixture(scope="module")
def fixture():
    return corpus.fixture()


@pytest.fixture(scope="module")
def queries(fixture):
    return corpus.records(fixture["queries"])


@pytest.fixture(scope="module")
def index_records(fixture):
    return corpus.records(fixture["index"])


@pytest.fixture(scope="module")
def vocabulary(fixture):
    return corpus.vocabulary(fixture["vocabulary"])


def run(queries, index_records, vocabulary, tmp_path, extra="", encoder=None):
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")
    return predict(
        queries,
        config(extra),
        vocabulary,
        index_records,
        store,
        encoder=encoder or corpus.FakeEncoder(),
    )


@pytest.fixture(scope="module")
def candidates(queries, index_records, vocabulary, tmp_path_factory):
    return run(
        queries, index_records, vocabulary, tmp_path_factory.mktemp("predict")
    )


# --- The output contract ----------------------------------------------------


def test_one_candidate_list_per_record_in_the_order_given(candidates, queries):
    assert [result.record_id for result in candidates] == [r.id for r in queries]


def test_exactly_fifty_codes_per_record(candidates):
    """The submission format takes 50 and has no representation for fewer."""
    assert {len(result.candidates) for result in candidates} == {CODES_PER_RECORD}


def test_scores_are_non_increasing(candidates):
    for result in candidates:
        scores = [candidate.score for candidate in result.candidates]
        assert scores == sorted(scores, reverse=True), result.record_id


def test_no_duplicate_codes_within_a_record(candidates):
    for result in candidates:
        assert len(set(result.codes)) == len(result.codes), result.record_id


def test_every_code_carries_the_retriever_that_found_it(candidates):
    for result in candidates:
        for candidate in result.candidates:
            assert candidate.sources == {"knn": candidate.sources.get("knn")}
            assert candidate.sources["knn"] >= 0


# --- The vocabulary restriction ---------------------------------------------


def test_every_returned_code_exists_in_the_vocabulary(candidates, vocabulary):
    for result in candidates:
        unknown = set(result.codes) - set(vocabulary)
        assert not unknown, f"{result.record_id} returned {sorted(unknown)}"


def test_codes_the_index_carries_but_the_vocabulary_does_not_are_never_returned(
    candidates, fixture, index_records
):
    """Index composition must not widen the label universe. See docs/spec.md."""
    withheld = set(fixture["withheld_codes"])
    assert withheld & {code for r in index_records for code in r.subjects}
    for result in candidates:
        assert not withheld & set(result.codes), result.record_id


# --- No record may see its own gold labels ----------------------------------


def test_perturbing_a_records_gold_labels_leaves_its_candidates_unchanged(
    queries, index_records, vocabulary, tmp_path, fixture
):
    perturbed = [
        dataclasses.replace(record, subjects=(fixture["unseen_code"],) + record.subjects[:1])
        for record in queries
    ]

    before = run(queries, index_records, vocabulary, tmp_path / "before")
    after = run(perturbed, index_records, vocabulary, tmp_path / "after")

    assert [result.codes for result in after] == [result.codes for result in before]


def test_a_record_present_in_the_index_does_not_retrieve_itself(
    index_records, vocabulary, tmp_path
):
    """Its own gold subjects would otherwise arrive as its nearest neighbour."""
    query = index_records[0]

    result = run([query], index_records, vocabulary, tmp_path)[0]
    without_self = run([query], index_records[1:], vocabulary, tmp_path / "b")[0]

    assert result.codes == without_self.codes


# --- Retrievers on and off --------------------------------------------------


def test_disabling_every_retriever_yields_empty_results(
    queries, index_records, vocabulary, tmp_path
):
    """An ablation that turns everything off is a row in the table, not a crash."""
    results = run(
        queries,
        index_records,
        vocabulary,
        tmp_path,
        extra="\nretrievers: {knn: {enabled: false}, dense: {enabled: false}, "
        "lexical: {enabled: false}}\n",
    )

    assert [result.record_id for result in results] == [r.id for r in queries]
    assert all(result.candidates == () for result in results)


def test_disabling_the_only_retriever_changes_the_output(
    queries, index_records, vocabulary, tmp_path, candidates
):
    off = run(
        queries,
        index_records,
        vocabulary,
        tmp_path,
        extra="\nretrievers: {knn: {enabled: false}, dense: {enabled: false}, "
        "lexical: {enabled: false}}\n",
    )

    assert [r.codes for r in off] != [r.codes for r in candidates]


@pytest.mark.parametrize("name", ["dense", "lexical"])
def test_a_retriever_that_is_not_built_yet_says_so(
    queries, index_records, vocabulary, tmp_path, name
):
    """Silently ignoring an enabled retriever would report an untrue ablation."""
    with pytest.raises(NotImplementedError, match="ticket"):
        run(
            queries,
            index_records,
            vocabulary,
            tmp_path,
            extra=f"\nretrievers: {{{name}: {{enabled: true}}}}\n",
        )


def test_a_flag_nothing_reads_yet_is_refused(
    queries, index_records, vocabulary, tmp_path
):
    """`bilingual: false` would otherwise score exactly like the default."""
    with pytest.raises(NotImplementedError, match="ticket 08"):
        run(
            queries,
            index_records,
            vocabulary,
            tmp_path,
            extra="\nlabel_text: {bilingual: false}\n",
        )


# --- Embeddings are computed once -------------------------------------------


def test_document_embeddings_are_cached_across_runs(
    queries, index_records, vocabulary, tmp_path
):
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")
    first, second = corpus.FakeEncoder(), corpus.FakeEncoder()

    for encoder in (first, second):
        predict(
            queries,
            config(),
            vocabulary,
            index_records,
            store,
            encoder=encoder,
        )

    assert first.encoded == len(queries) + len(index_records)
    assert second.encoded == 0


def test_a_different_encoder_is_not_served_the_cached_vectors(
    queries, index_records, vocabulary, tmp_path
):
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")
    first, second = corpus.FakeEncoder(), corpus.FakeEncoder()

    predict(queries, config(), vocabulary, index_records, store, encoder=first)
    predict(
        queries,
        load_experiment_text(KNN_ONLY.replace("fixture/fake", "fixture/other")),
        vocabulary,
        index_records,
        store,
        encoder=second,
    )

    assert second.encoded == len(queries) + len(index_records)
