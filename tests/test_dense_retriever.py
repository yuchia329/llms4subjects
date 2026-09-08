"""The label index and the document-to-label retriever built on it.

The retriever is the component the whole design exists to provide: a neighbour
harvest can only return labels some indexed record already carries, and 8.5% of
dev gold assignments are carried by none. Scoring the document against every
vocabulary vector is the only mechanism that reaches them, so the assertion this
file exists for is the reachability one — a code in no index record at all is
still returnable.

Nothing here asserts that a particular encoder ranks a particular label first;
the vectors come from `pipeline_fixture.FakeEncoder` for the reason docs/spec.md
gives under "Testing Decisions".
"""

import dataclasses

import numpy as np
import pipeline_fixture as corpus
import pytest

from llms4subjects.config import RetrieverConfig
from llms4subjects.stages.indexes import LabelIndex, build_label_index
from llms4subjects.stages.label_text import render
from llms4subjects.stages.retrievers import DenseLabelRetriever


@pytest.fixture(scope="module")
def fixture():
    return corpus.fixture()


@pytest.fixture(scope="module")
def vocabulary(fixture):
    return corpus.vocabulary(fixture["vocabulary"])


@pytest.fixture(scope="module")
def labels(vocabulary):
    return render(vocabulary.values())


@pytest.fixture(scope="module")
def queries(fixture):
    return corpus.records(fixture["queries"])


@pytest.fixture(scope="module")
def index(labels):
    return build_label_index(
        [label.code for label in labels],
        [label.text for label in labels],
        corpus.FakeEncoder(),
    )


def retrieve(index, records, **settings):
    retriever = DenseLabelRetriever(
        index, corpus.FakeEncoder(), RetrieverConfig(**settings)
    )
    return retriever.retrieve(records)


# --- The index --------------------------------------------------------------


def test_every_vocabulary_entry_is_indexed(index, labels):
    """All 79,427 of them in a real run: an entry left out is unreachable."""
    assert index.codes == tuple(label.code for label in labels)
    assert index.vectors.shape[0] == len(labels)
    assert len(index) == len(labels)


def test_the_label_tower_is_encoded_as_labels_not_as_documents(labels):
    """E5 was trained with `passage:` on this side, and degrades without it."""
    encoder = corpus.FakeEncoder()

    build_label_index(
        [label.code for label in labels], [label.text for label in labels], encoder
    )

    assert [kind for kind, _ in encoder.calls] == ["labels"]


def test_a_code_without_its_text_is_refused(labels):
    """Silently zipping short would index a code under another code's vector."""
    with pytest.raises(ValueError, match="79|code|text"):
        build_label_index(
            [label.code for label in labels],
            [label.text for label in labels][:-1],
            corpus.FakeEncoder(),
        )


# --- The output contract ----------------------------------------------------


def test_one_candidate_list_per_record_in_the_order_given(index, queries):
    results = retrieve(index, queries)

    assert [result.record_id for result in results] == [r.id for r in queries]


def test_top_k_candidates_per_record(index, queries):
    results = retrieve(index, queries, top_k=25)

    assert {len(result.candidates) for result in results} == {25}


def test_a_top_k_beyond_the_vocabulary_returns_all_of_it(index, queries, labels):
    results = retrieve(index, queries[:1], top_k=len(labels) * 2)

    assert len(results[0].candidates) == len(labels)


def test_scores_are_non_increasing(index, queries):
    for result in retrieve(index, queries):
        scores = [candidate.score for candidate in result.candidates]
        assert scores == sorted(scores, reverse=True), result.record_id


def test_no_duplicate_codes_within_a_record(index, queries):
    for result in retrieve(index, queries):
        assert len(set(result.codes)) == len(result.codes), result.record_id


def test_every_candidate_carries_its_rank(index, queries):
    for result in retrieve(index, queries):
        ranks = [candidate.sources["dense"] for candidate in result.candidates]
        assert ranks == list(range(len(result.candidates)))


def test_the_scores_are_the_similarities_to_the_label_vectors(index, queries):
    """The score is a cosine similarity, not a rank: fusion reads both."""
    record = queries[0]
    vector = corpus.FakeEncoder().encode_documents([record.text])[0]
    # Accelerate leaves the floating-point flags set after an ordinary product;
    # see `_similarities`, which is the code under test here.
    with np.errstate(all="ignore"):
        expected = dict(zip(index.codes, index.vectors @ vector))

    for candidate in retrieve(index, [record], top_k=5)[0].candidates:
        assert candidate.score == pytest.approx(expected[candidate.code], abs=1e-6)


def tied_index(codes):
    """One index in which every entry renders identically, so all scores tie.

    Not a contrivance: one pair of real vocabulary entries renders to the same
    text under the default qualifier mode, and 87 pairs under `stripped`.
    """
    vector = corpus.FakeEncoder().encode_labels(["Schlagwort: Ordnung"])
    return LabelIndex(
        codes=tuple(codes), vectors=np.vstack([vector] * len(codes))
    )


def test_ties_are_broken_by_code_rather_than_by_index_order(queries):
    """Two entries rendering identically must not rank by vocabulary order."""
    result = retrieve(tied_index(("gnd:b", "gnd:a")), queries[:1], top_k=2)[0]

    assert result.codes == ("gnd:a", "gnd:b")


def test_a_tie_at_the_cutoff_is_cut_by_code_rather_than_by_index_order(queries):
    """The tie that decides *membership*, which a plain partition would not.

    `top_k` is 100 of 79,427, so the candidate list is chosen by partition
    rather than by sorting; a partition leaves the boundary tie in whatever
    order the vector happened to be in.
    """
    index = tied_index(("gnd:9", "gnd:8", "gnd:7", "gnd:6", "gnd:5", "gnd:4"))

    result = retrieve(index, queries[:1], top_k=3)[0]

    assert result.codes == ("gnd:4", "gnd:5", "gnd:6")


def test_a_tie_below_the_cutoff_leaves_the_better_scores_alone(queries):
    """Only the boundary is contested: nothing above it may be displaced."""
    encoder = corpus.FakeEncoder()
    tied = encoder.encode_labels(["Schlagwort: Ordnung"])
    best = encoder.encode_documents([queries[0].text])
    index = LabelIndex(
        codes=("gnd:z", "gnd:a", "gnd:b"), vectors=np.vstack([best, tied, tied])
    )

    result = retrieve(index, queries[:1], top_k=2)[0]

    assert result.codes == ("gnd:z", "gnd:a")


# --- Reachability, which is the point of the ticket -------------------------


def test_labels_no_record_carries_are_returned_at_the_configured_top_k(
    index, queries, fixture
):
    """The zero-shot case, at the `top_k` a run actually uses rather than all.

    `unseen_code` is in the fixture vocabulary and in no fixture record; the
    property is that a candidate list is drawn from the vocabulary rather than
    from what some record carries, so codes outside the corpus reach it.
    """
    carried = {
        code
        for record in corpus.records(fixture["index"] + fixture["queries"])
        for code in record.subjects
    }
    returned = {code for result in retrieve(index, queries, top_k=100)
                for code in result.codes}

    assert returned - carried


def test_every_code_including_the_unseen_one_is_reachable(index, queries, fixture, labels):
    """Nothing in the vocabulary is structurally excluded from a ranking."""
    results = retrieve(index, queries[:1], top_k=len(labels))

    assert set(results[0].codes) == set(index.codes)
    assert fixture["unseen_code"] in results[0].codes


def test_the_ranking_does_not_depend_on_the_records_own_subjects(index, queries):
    """No gold label of a record may influence its own candidates, ever."""
    perturbed = [
        dataclasses.replace(record, subjects=("gnd:4000000-0",)) for record in queries
    ]

    assert [r.codes for r in retrieve(index, perturbed)] == [
        r.codes for r in retrieve(index, queries)
    ]


# --- Degenerate inputs ------------------------------------------------------


def test_no_records_is_not_an_error(index):
    assert retrieve(index, []) == []


def test_an_empty_vocabulary_yields_empty_candidate_lists(queries):
    empty = LabelIndex(codes=(), vectors=np.zeros((0, corpus.DIMENSIONS), np.float32))

    results = retrieve(empty, queries)

    assert [result.candidates for result in results] == [()] * len(queries)
