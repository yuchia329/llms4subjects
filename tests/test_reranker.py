"""Cross-encoder reranking: the contract, not the model's taste.

What a cross-encoder ranks first is a property of its weights, which
docs/spec.md's testing decisions rule out asserting. What this file asserts is
everything around that: reranking may reorder the candidate list and cut it to
the submission's 50, and may not introduce a code, lose a candidate's
provenance, or read a label the vocabulary does not have text for. Those are the
properties that must survive swapping the reranker, which is the swap ticket 12
exists to make cheap.

The confidence measure gets the same treatment. Its numeric value is the
model's; what is asserted is that it is a probability, that it moves with the
scores of the top candidates and not with the tail, and that it is defined for a
record the retrievers found nothing for — the adjudicator routes on it, so an
undefined confidence would route by accident.
"""

import dataclasses

import pipeline_fixture as corpus
import pytest

from llms4subjects.artifacts import ArtifactStore
from llms4subjects.config import (
    ConfigError,
    RerankerConfig,
    load_experiment,
    load_experiment_text,
)
from llms4subjects.contracts import (
    CODES_PER_RECORD,
    Candidate,
    CandidateList,
    Record,
)
from llms4subjects.models import MODEL_CUTOFF, UnverifiedRevision
from llms4subjects.paths import CONFIG_DIR
from llms4subjects.pipeline import predict
from llms4subjects.stages.reranker import (
    CONFIDENCE_K,
    confidence,
    pinned,
    rerank,
    resolve,
)

CONFIG = RerankerConfig(
    enabled=True, model="fixture/fake-reranker", input_k=100, output_k=50
)


def record(record_id: str = "r1", title: str = "Polymerisation") -> Record:
    return Record(
        id=record_id, type="Book", lang="de", title=title, abstract="abstract"
    )


def candidates(record_id: str, *codes: str, scores=None) -> CandidateList:
    """A fused list for one record: codes best first, descending scores."""
    return CandidateList(
        record_id=record_id,
        candidates=tuple(
            Candidate(
                code=code,
                score=scores[rank] if scores else 1.0 / (rank + 1),
                sources={"knn": rank},
            )
            for rank, code in enumerate(codes)
        ),
    )


def label_texts(*codes: str) -> dict[str, str]:
    return {code: f"Schlagwort: {code} name" for code in codes}


def reranked(
    records, lists, texts=None, config=CONFIG, model=None
) -> list[CandidateList]:
    codes = [code for result in lists for code in result.codes]
    return rerank(
        records,
        lists,
        texts if texts is not None else label_texts(*codes),
        config,
        model=model or corpus.FakeCrossEncoder(),
    )


# --- The output contract ----------------------------------------------------


def test_one_list_per_record_in_the_order_given():
    records = [record("r1"), record("r2", title="Katalyse")]
    lists = [candidates("r1", "a", "b", "c"), candidates("r2", "b", "d")]

    results = reranked(records, lists)

    assert [result.record_id for result in results] == ["r1", "r2"]


def test_reranking_cuts_the_list_to_the_submission_length():
    """The pipeline contract: 100 candidates in, the final 50 out."""
    codes = [f"gnd:{index}" for index in range(100)]
    results = reranked([record()], [candidates("r1", *codes)])

    assert len(results[0].candidates) == CODES_PER_RECORD


def test_a_shorter_candidate_list_is_not_padded():
    """The lexical retriever can return nothing for an English title."""
    results = reranked([record()], [candidates("r1", "a", "b")])

    assert len(results[0].candidates) == 2


def test_scores_are_non_increasing():
    codes = [f"gnd:{index}" for index in range(60)]
    results = reranked([record()], [candidates("r1", *codes)])

    scores = [candidate.score for candidate in results[0].candidates]
    assert scores == sorted(scores, reverse=True)


def test_no_code_is_introduced_that_was_not_a_candidate():
    """The reranker orders a candidate set; inventing a code would be a bug
    the submission would carry all the way to the scorer."""
    offered = candidates("r1", *[f"gnd:{index}" for index in range(80)])

    results = reranked([record()], [offered])

    assert set(results[0].codes) <= set(offered.codes)


def test_no_duplicate_codes_within_a_record():
    codes = [f"gnd:{index}" for index in range(70)]
    results = reranked([record()], [candidates("r1", *codes)])

    assert len(set(results[0].codes)) == len(results[0].codes)


def test_a_candidate_keeps_the_retriever_that_found_it():
    """Attribution is the project's claim, and it has to survive reranking."""
    offered = candidates("r1", "a", "b", "c")

    results = reranked([record()], [offered])

    before = {c.code: c.sources for c in offered.candidates}
    for candidate in results[0].candidates:
        assert candidate.sources == before[candidate.code]


def test_reranking_reorders_what_fusion_ranked():
    """Otherwise the stage is a truncation wearing a cross-encoder's name."""
    codes = [f"gnd:{index}" for index in range(100)]
    offered = candidates("r1", *codes)

    results = reranked([record()], [offered])

    assert results[0].codes != offered.codes[:CODES_PER_RECORD]


def test_a_record_with_no_candidates_survives():
    """An ablation that turns every retriever off is a row in the table."""
    results = reranked([record()], [CandidateList("r1", ())])

    assert results == [CandidateList("r1", ())]


def test_scoring_is_per_record_so_two_records_do_not_share_a_ranking():
    """One pair batch covers every record, so a positional slip is invisible."""
    records = [record("r1", title="Polymerisation"), record("r2", title="Katalyse")]
    lists = [candidates("r1", "a", "b", "c"), candidates("r2", "a", "b", "c")]

    first, second = reranked(records, lists)

    assert first.codes != second.codes


# --- What it refuses --------------------------------------------------------


def test_only_the_top_input_k_candidates_are_scored():
    """`input_k` is a cost decision, so it must be the number of pairs paid for."""
    codes = [f"gnd:{index}" for index in range(100)]
    model = corpus.FakeCrossEncoder()

    reranked(
        [record()],
        [candidates("r1", *codes)],
        config=dataclasses.replace(CONFIG, input_k=20, output_k=20),
        model=model,
    )

    assert model.pairs == 20


def test_a_candidate_beyond_input_k_cannot_reach_the_output():
    codes = [f"gnd:{index}" for index in range(100)]

    results = reranked(
        [record()],
        [candidates("r1", *codes)],
        config=dataclasses.replace(CONFIG, input_k=20, output_k=20),
    )

    assert set(results[0].codes) <= set(codes[:20])


# --- Replacing the ranking, or fusing with it -------------------------------


FUSING = dataclasses.replace(CONFIG, mix="fuse")


def test_fusing_with_the_incoming_order_is_not_the_same_as_replacing_it():
    codes = [f"gnd:{index}" for index in range(100)]
    lists = [candidates("r1", *codes)]

    replaced = reranked([record()], lists)
    fused = reranked([record()], lists, config=FUSING)

    assert fused[0].codes != replaced[0].codes


def test_a_mix_weight_of_zero_recovers_the_order_it_was_given():
    """The ablation that pays for the pass and changes nothing, which is what
    makes every other `mix_weight` row attributable to the cross-encoder."""
    codes = [f"gnd:{index}" for index in range(100)]
    offered = candidates("r1", *codes)

    fused = reranked(
        [record()],
        [offered],
        config=dataclasses.replace(FUSING, mix_weight=0.0),
    )

    assert fused[0].codes == offered.codes[:CODES_PER_RECORD]


def test_fusing_keeps_the_output_contract():
    codes = [f"gnd:{index}" for index in range(100)]
    offered = candidates("r1", *codes)

    result = reranked([record()], [offered], config=FUSING)[0]

    assert len(result.candidates) == CODES_PER_RECORD
    scores = [candidate.score for candidate in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert set(result.codes) <= set(offered.codes)
    assert len(set(result.codes)) == len(result.codes)


class ScriptedCrossEncoder:
    """A model whose scores the test chooses, for asserting the arithmetic.

    The fake digest model is right for "did the flag reach the pair"; a fused
    order is arithmetic this repository owns, so it gets the treatment
    test_fusion.py gives reciprocal rank fusion — a hand-computable case.
    """

    def __init__(self, *scores: float):
        self._scores = scores

    def score(self, pairs):
        import numpy as np

        return np.array(self._scores[: len(pairs)], dtype=np.float32)


def test_fusing_reads_both_orders_rather_than_either_alone():
    """Hand-computed at `mix_rrf_k` 60 and weight 1.0, retrieval order a, b, c
    and model order b, c, a: b scores 1/61 + 1/62, a 1/63 + 1/61, c 1/62 + 1/63,
    so the fused order is b, a, c — which is neither input."""
    offered = candidates("r1", "a", "b", "c")

    result = reranked(
        [record()],
        [offered],
        config=FUSING,
        model=ScriptedCrossEncoder(0.1, 0.9, 0.5),
    )[0]

    assert result.codes == ("b", "a", "c")


def test_an_unknown_mix_is_refused():
    with pytest.raises(ConfigError, match="reranker.mix"):
        load_experiment_text(
            "name: bad-mix\n"
            "encoder: {name: fixture/fake}\n"
            "reranker: {enabled: true, model: m, mix: blend}\n"
        )


def test_a_candidate_with_no_label_text_is_named_rather_than_skipped():
    """A missing rendering would silently drop a code from the submission."""
    with pytest.raises(KeyError, match="gnd:missing"):
        reranked(
            [record()],
            [candidates("r1", "a", "gnd:missing")],
            texts=label_texts("a"),
        )


def test_which_side_is_the_query_changes_the_ranking():
    """These models are asymmetric, so the flag has to reach the pair it names.

    A flag that were silently ignored would make the screening table in
    docs/results.md two copies of one measurement.
    """
    codes = [f"gnd:{index}" for index in range(100)]
    lists = [candidates("r1", *codes)]

    document = reranked([record()], lists)
    label = reranked(
        [record()], lists, config=dataclasses.replace(CONFIG, query="label")
    )

    assert document[0].codes != label[0].codes


def test_an_unknown_query_side_is_refused():
    with pytest.raises(ConfigError, match="reranker.query"):
        load_experiment_text(
            "name: bad-side\n"
            "encoder: {name: fixture/fake}\n"
            "reranker: {enabled: true, model: m, query: sideways}\n"
        )


def test_records_and_candidates_must_line_up():
    """Reranking is positional; scoring one record's text against another's
    candidates would be invisible in the output and fatal to the numbers."""
    with pytest.raises(ValueError, match="r2"):
        reranked([record("r1")], [candidates("r2", "a", "b")])


# --- Confidence -------------------------------------------------------------


def test_confidence_is_a_probability():
    result = candidates("r1", "a", "b", "c", scores=[0.9, 0.4, 0.1])

    assert 0.0 <= confidence(result) <= 1.0


def test_a_stronger_top_of_the_list_is_more_confident():
    """What routing needs: the records the reranker is least sure of."""
    sure = candidates("r1", "a", "b", "c", scores=[0.95, 0.9, 0.85])
    unsure = candidates("r1", "a", "b", "c", scores=[0.2, 0.1, 0.05])

    assert confidence(sure) > confidence(unsure)


def test_confidence_reads_the_top_of_the_list_and_not_its_tail():
    """A record is not unconfident because its 50th candidate is weak; every
    record's 50th candidate is weak, and 45 of the 50 are never looked at."""
    codes = [f"gnd:{index}" for index in range(CONFIDENCE_K + 5)]
    strong = [0.9] * CONFIDENCE_K
    result = candidates("r1", *codes, scores=strong + [0.5] * 5)
    weaker_tail = candidates("r1", *codes, scores=strong + [0.01] * 5)

    assert confidence(result) == confidence(weaker_tail)


def test_a_record_the_retrievers_found_nothing_for_is_least_confident():
    """Routing must be defined for it rather than raising inside the router."""
    assert confidence(CandidateList("r1", ())) == 0.0


def test_a_short_list_is_scored_on_what_it_has():
    result = candidates("r1", "a", scores=[0.8])

    assert confidence(result) == pytest.approx(0.8)


# --- The configuration ------------------------------------------------------


def test_enabling_the_reranker_without_naming_a_model_is_refused():
    """The alternative is a traceback out of a model load after indexing."""
    with pytest.raises(ConfigError, match="reranker.model"):
        load_experiment_text(
            "name: no-model\n"
            "encoder: {name: fixture/fake}\n"
            "reranker: {enabled: true}\n"
        )


def test_every_committed_reranker_is_one_the_cutoff_registry_vouches_for():
    """The 2025-01-31 cutoff covers rerankers too (docs/spec.md, story 2).

    `resolve` is what refuses an unregistered or too-recent model, and it is
    checked here over the committed configs so that a screening run cannot
    quietly load weights the comparison is not entitled to.
    """
    enabled = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        config = load_experiment(path)
        if config.reranker.enabled:
            enabled.append(path.name)
            entry = resolve(config.reranker).release
            assert entry.role == "reranker", path.name
            assert entry.created <= MODEL_CUTOFF, path.name

    assert enabled, "no committed config enables the reranker"


def test_a_reranker_revision_the_registry_has_not_dated_is_refused():
    """The cutoff is a claim about the commit that loads, as for the encoder."""
    with pytest.raises(UnverifiedRevision, match="2025-01-31"):
        resolve(
            dataclasses.replace(
                CONFIG, model="BAAI/bge-reranker-base", revision="f" * 40
            )
        )


def test_a_config_may_restate_the_registrys_pin():
    config = dataclasses.replace(CONFIG, model="BAAI/bge-reranker-base")
    pin = resolve(config).revision

    assert resolve(dataclasses.replace(config, revision=pin)).revision == pin


def test_the_pinned_config_carries_the_revision_an_artifact_key_needs():
    """A key over the section as written would serve one model's scores to the
    next revision of it; `scripts/rerank_report.py` keys on this instead."""
    config = load_experiment(CONFIG_DIR / "rung1-rerank-base.yaml")

    assert config.reranker.revision is None
    assert pinned(config).reranker.revision == resolve(config.reranker).revision


def test_an_output_shorter_than_one_code_is_refused():
    with pytest.raises(ConfigError, match="output_k"):
        load_experiment_text(
            "name: empty-output\n"
            "encoder: {name: fixture/fake}\n"
            "reranker: {enabled: true, model: m, output_k: 0}\n"
        )


def test_a_reranker_cannot_read_more_candidates_than_fusion_emits():
    """Otherwise the run reranks 50 pairs a record and reports 100."""
    with pytest.raises(ConfigError, match="fusion.candidates"):
        load_experiment_text(
            "name: too-deep\n"
            "encoder: {name: fixture/fake}\n"
            "fusion: {candidates: 50}\n"
            "reranker: {enabled: true, model: m, input_k: 100, output_k: 50}\n"
        )


def test_a_reranker_cannot_be_asked_for_more_output_than_input():
    with pytest.raises(ConfigError, match="output_k"):
        load_experiment_text(
            "name: too-wide\n"
            "encoder: {name: fixture/fake}\n"
            "reranker: {enabled: true, model: m, input_k: 20, output_k: 50}\n"
        )


# --- Through the pipeline seam ----------------------------------------------

KNN_ONLY = """
name: fixture-rerank
encoder: {name: fixture/fake}
index: {corpora: [core_train]}
retrievers:
  knn: {enabled: true, neighbours: 5, top_k: 100}
  dense: {enabled: false}
  lexical: {enabled: false}
"""

RERANKING = "\nreranker: {enabled: true, model: fixture/fake-reranker}\n"


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


def run(queries, index_records, vocabulary, tmp_path, extra="", **kwargs):
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")
    return predict(
        queries,
        load_experiment_text(KNN_ONLY + extra),
        vocabulary,
        index_records,
        store,
        encoder=corpus.FakeEncoder(),
        **kwargs,
    )


@pytest.fixture(scope="module")
def unranked(queries, index_records, vocabulary, tmp_path_factory):
    return run(
        queries, index_records, vocabulary, tmp_path_factory.mktemp("plain")
    )


@pytest.fixture(scope="module")
def through_the_reranker(queries, index_records, vocabulary, tmp_path_factory):
    return run(
        queries,
        index_records,
        vocabulary,
        tmp_path_factory.mktemp("reranked"),
        extra=RERANKING,
        cross_encoder=corpus.FakeCrossEncoder(),
    )


def test_the_pipeline_returns_the_reranked_fifty(through_the_reranker, queries):
    assert [r.record_id for r in through_the_reranker] == [r.id for r in queries]
    assert {len(r.candidates) for r in through_the_reranker} == {CODES_PER_RECORD}


def test_reranking_keeps_every_contract_the_unranked_pipeline_keeps(
    through_the_reranker, vocabulary
):
    for result in through_the_reranker:
        scores = [candidate.score for candidate in result.candidates]
        assert scores == sorted(scores, reverse=True), result.record_id
        assert len(set(result.codes)) == len(result.codes), result.record_id
        assert not set(result.codes) - set(vocabulary), result.record_id
        for candidate in result.candidates:
            assert candidate.sources


def test_reranking_changes_the_ranking_it_was_given(
    through_the_reranker, unranked
):
    before = [result.codes[:CODES_PER_RECORD] for result in unranked]
    assert [result.codes for result in through_the_reranker] != before


def test_the_reranked_codes_come_from_the_candidate_set(
    through_the_reranker, unranked
):
    for after, before in zip(through_the_reranker, unranked):
        assert set(after.codes) <= set(before.codes)


def test_the_pipeline_is_unchanged_with_reranking_disabled(
    queries, index_records, vocabulary, tmp_path, unranked
):
    """The toggle has to be a toggle: off must be exactly the earlier pipeline."""
    off = run(
        queries,
        index_records,
        vocabulary,
        tmp_path,
        extra="\nreranker: {enabled: false, model: fixture/fake-reranker}\n",
    )

    assert [r.codes for r in off] == [r.codes for r in unranked]


def test_the_label_form_changes_what_the_reranker_is_shown(
    queries, index_records, vocabulary, tmp_path_factory
):
    """`label_form: name` shows the heading alone instead of the field-marked
    text, which is a different string and must therefore be a different ranking.

    The screening table in docs/results.md compares the two forms, so a flag
    that did not reach the pair would make those rows one measurement twice.
    """
    rendered = run(
        queries,
        index_records,
        vocabulary,
        tmp_path_factory.mktemp("rendered"),
        extra=RERANKING,
        cross_encoder=corpus.FakeCrossEncoder(),
    )
    names = run(
        queries,
        index_records,
        vocabulary,
        tmp_path_factory.mktemp("names"),
        extra="\nreranker: {enabled: true, model: fixture/fake-reranker, "
        "label_form: name}\n",
        cross_encoder=corpus.FakeCrossEncoder(),
    )

    assert [r.codes for r in names] != [r.codes for r in rendered]


def test_the_reranker_reads_the_bilingual_label_text(
    queries, index_records, vocabulary, tmp_path_factory
):
    """A kNN-only run renders no label text, so the reranker is the only reader
    here — and if the flag did not reach it, every bilingual reranking row would
    be a report on German-only text."""
    german = run(
        queries,
        index_records,
        vocabulary,
        tmp_path_factory.mktemp("de"),
        extra=RERANKING + "\nlabel_text: {bilingual: false}\n",
        cross_encoder=corpus.FakeCrossEncoder(),
    )
    both = run(
        queries,
        index_records,
        vocabulary,
        tmp_path_factory.mktemp("both"),
        extra=RERANKING,
        cross_encoder=corpus.FakeCrossEncoder(),
    )

    assert [r.codes for r in german] != [r.codes for r in both]
