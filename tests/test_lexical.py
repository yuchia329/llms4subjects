"""The lexical retriever: label strings matched verbatim against record text.

This retriever is not a model wrapper, so unlike the encoders it is testable on
its own terms: the ranking it produces is a property of code in this repository
rather than of external weights. What is asserted here is what "lexical" is
supposed to mean — a label whose own name or one of whose synonyms appears in
the text is a candidate, a rare term counts for more than a common one, and
nothing else about the label's rendering leaks into the match.

The contract invariants of the pipeline as a whole live in test_pipeline.py;
the two tests at the end of this file are the ones that need the lexical
retriever specifically, because it is the first retriever that can return a
label no indexed document carries.
"""

import pipeline_fixture as corpus
import pytest

from llms4subjects.artifacts import ArtifactStore
from llms4subjects.config import RetrieverConfig, load_experiment_text
from llms4subjects.contracts import CODES_PER_RECORD, Record, VocabularyEntry
from llms4subjects.pipeline import predict
from llms4subjects.stages import encoders, indexes, label_text, retrievers

POLYMERISATION = VocabularyEntry(
    code="gnd:4046699-1",
    name="Polymerisation",
    classification_name="Chemie",
    alternate_names=("Polyreaktion", "Kettenpolymerisation"),
)

STAHLBAU = VocabularyEntry(
    code="gnd:4056824-2",
    name="Stahlbau",
    classification_name="Bauwesen",
    alternate_names=("Stahlkonstruktion",),
)

ORDNUNG = VocabularyEntry(
    code="gnd:4043744-9",
    name="Ordnung",
    classification_name="Mathematik",
    definition="Allgemeinbegriff, verknüpfe mit Anwendungsgebiet",
)

VOCABULARY = (POLYMERISATION, STAHLBAU, ORDNUNG)


def build(entries=VOCABULARY, **settings) -> retrievers.LexicalLabelRetriever:
    """A lexical retriever over these entries, at the given retriever settings."""
    variants = label_text.variants(entries)
    return retrievers.LexicalLabelRetriever(
        indexes.build_lexical_index(list(variants), list(variants.values())),
        RetrieverConfig(**{"top_k": 10, **settings}),
    )


def record(text: str, identifier: str = "r1", lang: str = "de") -> Record:
    return Record(id=identifier, type="Book", lang=lang, title=text, abstract="")


def codes(retriever, text: str) -> tuple[str, ...]:
    return retriever.retrieve([record(text)])[0].codes


# --- What "lexical" means ---------------------------------------------------


def test_a_label_whose_name_appears_verbatim_is_the_first_candidate():
    ranked = codes(build(), "Radikalische Polymerisation von Styrol")

    assert ranked[0] == POLYMERISATION.code


def test_a_label_reached_through_a_synonym_is_a_candidate():
    """53.7% of German gold assignments have the name *or a synonym* in the text."""
    ranked = codes(build(), "Kettenpolymerisation im Reaktor")

    assert ranked[0] == POLYMERISATION.code


def test_a_label_sharing_no_term_with_the_text_is_not_a_candidate():
    ranked = codes(build(), "Polymerisation")

    assert ranked == (POLYMERISATION.code,)


def test_a_record_that_matches_nothing_gets_an_empty_candidate_list():
    """Lexical matching genuinely finds nothing for some records; that is a result."""
    result = build().retrieve([record("Cold-formed steel design")])[0]

    assert result.record_id == "r1"
    assert result.candidates == ()


def test_case_and_punctuation_do_not_decide_a_match():
    assert codes(build(), "POLYMERISATION, radikalisch.") == (POLYMERISATION.code,)


def test_a_rarer_term_outranks_a_term_many_labels_share():
    """Otherwise every label of a large subject area matches every record in it."""
    common = tuple(
        VocabularyEntry(code=f"gnd:{index}", name=f"Technik {index}")
        for index in range(20)
    )
    retriever = build((POLYMERISATION,) + common)

    ranked = codes(retriever, "Technik der Polymerisation")

    assert ranked[0] == POLYMERISATION.code


def test_the_classification_name_is_not_part_of_the_match():
    """`Fachgebiet: Chemie` is shared by thousands of labels and describes none.

    The dense retriever reads the field-marked rendering, which includes it. The
    BM25 path matches the label's own strings instead, which is what the 53.7%
    figure in docs/spec.md was measured over.

    The sparse-weight path is the deliberate exception: it reads the same
    rendering the dense tower does, because its whole reason to exist is that
    those weights come free from the tower's forward pass. So what "lexical"
    matches against depends on the encoder, and this assertion pins the BM25
    path only.
    """
    assert codes(build(), "Chemie") == ()
    assert codes(build(), "Mathematik") == ()


def test_the_definition_is_not_part_of_the_match():
    """It holds cataloguing instructions to librarians, in every subject area."""
    assert codes(build(), "verknüpfe mit Anwendungsgebiet") == ()


# --- The ranking is a ranking ------------------------------------------------


def test_candidates_are_scored_in_non_increasing_order():
    retriever = build()

    result = retriever.retrieve([record("Polymerisation und Stahlbau")])[0]

    scores = [candidate.score for candidate in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert len(set(result.codes)) == len(result.codes)


def test_every_candidate_carries_this_retrievers_provenance():
    result = build().retrieve([record("Polymerisation und Stahlbau")])[0]

    assert [candidate.sources for candidate in result.candidates] == [
        {"lexical": 0},
        {"lexical": 1},
    ]


def test_top_k_caps_the_candidate_list():
    retriever = build(top_k=1)

    assert len(codes(retriever, "Polymerisation und Stahlbau")) == 1


def test_a_label_matched_twice_over_is_not_returned_twice():
    """Name and synonym both hit; the label is one candidate, at its best score."""
    result = build().retrieve([record("Polymerisation als Polyreaktion")])[0]

    assert result.codes == (POLYMERISATION.code,)


def test_a_tie_is_broken_by_code_rather_than_by_input_order():
    """A ranking that depends on dict order is not reproducible. See retrievers."""
    twins = (
        VocabularyEntry(code="gnd:b", name="Polymerisation"),
        VocabularyEntry(code="gnd:a", name="Polymerisation"),
    )

    forwards = codes(build(twins), "Polymerisation")
    backwards = codes(build(twins[::-1]), "Polymerisation")

    assert forwards == ("gnd:a", "gnd:b") == backwards


def test_records_are_answered_in_the_order_they_were_given():
    given = [record("Polymerisation", "a"), record("Stahlbau", "b")]

    assert [r.record_id for r in build().retrieve(given)] == ["a", "b"]


def test_no_records_and_no_labels_are_both_answerable():
    assert build().retrieve([]) == []
    assert codes(build(()), "Polymerisation") == ()


# --- The encoder's own term weights, where it emits them --------------------


class FakeSparseEncoder(corpus.FakeEncoder):
    """An encoder whose forward pass also emits sparse term weights.

    Only some models do — BGE-M3 among the four candidates — and where one does,
    the lexical retriever must use them rather than build a second index, since
    they come from the pass the dense tower already paid for. The weights here
    are deliberately not BM25's: a test that could not tell the two apart would
    not be testing which one ran.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        super().__init__()
        self._weights = weights or {}

    def encode_sparse_documents(self, texts):
        return [self._sparse(text) for text in texts]

    def encode_sparse_labels(self, texts):
        return [self._sparse(text) for text in texts]

    def _sparse(self, text: str) -> dict[str, float]:
        return {
            term: self._weights.get(term, 1.0)
            for term in indexes.lexical_terms(text)
        }


def test_an_encoder_that_emits_term_weights_is_recognised():
    assert encoders.sparse_weights(FakeSparseEncoder()) is not None
    assert encoders.sparse_weights(corpus.FakeEncoder()) is None


def test_the_encoders_term_weights_rank_the_candidates_when_it_has_them():
    """The weights decide the order, which BM25 over the same strings would not."""
    encoder = FakeSparseEncoder({"stahlbau": 10.0, "polymerisation": 0.1})
    texts = {entry.code: entry.name for entry in VOCABULARY}
    retriever = retrievers.LexicalLabelRetriever(
        indexes.build_sparse_term_index(
            list(texts), list(texts.values()), encoder
        ),
        RetrieverConfig(top_k=10),
    )

    ranked = codes(retriever, "Polymerisation und Stahlbau")

    assert ranked == (STAHLBAU.code, POLYMERISATION.code)
    assert encoder.encoded == 0, "sparse weights cost no dense encoding"


# --- Through the seam -------------------------------------------------------

LEXICAL_ONLY = """
name: fixture-lexical
encoder: {name: fixture/fake}
index: {corpora: [core_train]}
retrievers:
  knn: {enabled: false}
  dense: {enabled: false}
  lexical: {enabled: true, top_k: 100}
"""


@pytest.fixture(scope="module")
def fixture():
    return corpus.fixture()


def run(queries, index_records, vocabulary, tmp_path, encoder=None):
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")
    return predict(
        queries,
        load_experiment_text(LEXICAL_ONLY),
        vocabulary,
        index_records,
        store,
        encoder=encoder or corpus.FakeEncoder(),
    )


def test_a_label_no_indexed_document_carries_is_still_returned(fixture, tmp_path):
    """The property the whole design exists to provide, as a measurement.

    `unseen_code` is in the vocabulary and in none of the fixture's indexed
    records, so no neighbour harvest can reach it at any index size.
    """
    vocabulary = corpus.vocabulary(fixture["vocabulary"])
    unseen = vocabulary[fixture["unseen_code"]]
    query = record(unseen.name, "asks-for-the-unseen-label")

    result = run([query], corpus.records(fixture["index"]), vocabulary, tmp_path)[0]

    assert fixture["unseen_code"] in result.codes


def test_the_lexical_retriever_needs_no_document_vectors(fixture, tmp_path):
    """It is the one retriever that runs without the encoder, and cheaply."""
    encoder = corpus.FakeEncoder()

    results = run(
        corpus.records(fixture["queries"]),
        corpus.records(fixture["index"]),
        corpus.vocabulary(fixture["vocabulary"]),
        tmp_path,
        encoder=encoder,
    )

    assert encoder.encoded == 0
    assert any(result.candidates for result in results)


def test_the_seam_contract_holds_for_lexical_candidates(fixture, tmp_path):
    vocabulary = corpus.vocabulary(fixture["vocabulary"])

    results = run(
        corpus.records(fixture["queries"]),
        corpus.records(fixture["index"]),
        vocabulary,
        tmp_path,
    )

    for result in results:
        scores = [candidate.score for candidate in result.candidates]
        assert scores == sorted(scores, reverse=True), result.record_id
        assert len(set(result.codes)) == len(result.codes), result.record_id
        assert not set(result.codes) - set(vocabulary), result.record_id
        assert len(result.codes) <= CODES_PER_RECORD, result.record_id
