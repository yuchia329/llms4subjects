"""LLM adjudication: the constraint, not the model's taste.

Which of thirty candidates a language model prefers is a property of its
weights, and docs/spec.md's testing decisions rule out asserting it. What this
file asserts is the constraint the ticket exists for: the model chooses from a
list and never authors an identifier, a response that names a code outside the
offered set is rejected and logged rather than scored, and a rejected record
keeps the ranking it arrived with — so the stage can only ever reorder.

Everything else here is the wiring around that: routing sends the least
confident share onward and nobody else, a cached response is replayed instead of
re-billed, and the whole stage disabled leaves the pipeline byte-identical.
"""

import dataclasses
import json

import pipeline_fixture as corpus
import pytest

from llms4subjects.artifacts import (
    ArtifactStore,
    load_responses,
    log_rejections,
    read_rejections,
)
from llms4subjects.config import (
    AdjudicationConfig,
    ConfigError,
    load_experiment,
    load_experiment_text,
)
from llms4subjects.contracts import Candidate, CandidateList, Record
from llms4subjects.models import MODEL_CUTOFF, ModelTooRecent, UnregisteredModel
from llms4subjects.paths import CONFIG_DIR
from llms4subjects.pipeline import predict
from llms4subjects.stages.adjudicator import (
    ConstraintViolation,
    MalformedResponse,
    adjudicate,
    apply,
    parse,
    prompt,
    resolve,
    route,
    validate,
)

CONFIG = AdjudicationConfig(
    enabled=True, model="fixture/fake-llm", candidates=30, select=10
)


def record(record_id: str = "r1", title: str = "Polymerisation") -> Record:
    return Record(
        id=record_id, type="Book", lang="de", title=title, abstract="abstract"
    )


def candidates(record_id: str, *codes: str, scores=None) -> CandidateList:
    """A ranked list for one record: codes best first, descending scores."""
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


def adjudicated(records, lists, responses, config=CONFIG, texts=None):
    codes = [code for result in lists for code in result.codes]
    return adjudicate(
        records,
        lists,
        texts if texts is not None else label_texts(*codes),
        config,
        responses=responses,
    )


def answer(*codes: str) -> str:
    return json.dumps(list(codes))


# --- The constraint ---------------------------------------------------------


def test_a_returned_code_that_was_never_offered_is_refused():
    with pytest.raises(ConstraintViolation, match="gnd:invented"):
        validate(("a", "b", "c"), ("a", "gnd:invented"))


def test_a_subset_of_the_offered_codes_is_returned_unchanged():
    assert validate(("a", "b", "c"), ("c", "a")) == ("c", "a")


def test_an_invented_code_rejects_the_whole_response():
    """Not "keep the valid part": a response that invented one identifier is
    not evidence about the ones it did not invent."""
    offered = candidates("r1", "a", "b", "c")

    result = adjudicated([record()], [offered], {"r1": answer("a", "gnd:4711-0")})[0]

    assert result.rejected
    assert "gnd:4711-0" in result.reason
    assert result.codes == ()


def test_a_rejected_record_keeps_the_ranking_it_arrived_with():
    """The output contract has no representation for a missing record, so a
    constraint violation has to cost the adjudication and nothing else."""
    offered = candidates("r1", "a", "b", "c")
    results = adjudicated([record()], [offered], {"r1": answer("gnd:invented")})

    assert apply([offered], results, CONFIG) == [offered]


def test_a_response_that_is_not_a_list_of_codes_is_rejected_too():
    offered = candidates("r1", "a", "b")

    result = adjudicated([record()], [offered], {"r1": "I would pick a."})[0]

    assert result.rejected
    assert not result.codes


def test_parse_reads_the_array_out_of_a_chatty_response():
    """Models prepend prose and fence their JSON; neither is a violation."""
    assert parse('Here you go:\n```json\n["a", "b"]\n```') == ("a", "b")


def test_parse_skips_a_bracket_in_the_prose():
    """A citation or an aside is not the answer, and taking the first bracket
    pair would reject a perfectly valid response — permanently, since the
    rejection is cached like any other."""
    assert parse('[see below] I would choose:\n["a", "b"]') == ("a", "b")


def test_parse_refuses_a_response_with_no_array_in_it():
    with pytest.raises(MalformedResponse):
        parse("gnd:4043744-9 and gnd:4030309-2")


def test_parse_refuses_an_array_that_is_not_of_strings():
    with pytest.raises(MalformedResponse):
        parse("[1, 2, 3]")


def test_the_model_cannot_lengthen_the_list_it_was_given():
    offered = candidates("r1", *[f"gnd:{index}" for index in range(50)])
    results = adjudicated([record()], [offered], {"r1": answer("gnd:7", "gnd:2")})

    after = apply([offered], results, CONFIG)[0]

    assert len(after.candidates) == len(offered.candidates)
    assert set(after.codes) == set(offered.codes)


def test_the_model_reorders_what_it_was_shown():
    offered = candidates("r1", "a", "b", "c", "d")
    results = adjudicated([record()], [offered], {"r1": answer("c", "a")})

    after = apply([offered], results, CONFIG)[0]

    assert after.codes == ("c", "a", "b", "d")


def test_a_repeated_code_is_not_a_violation_but_appears_once():
    """A duplicate is sloppiness, not an invented identifier; the no-duplicates
    invariant still has to hold downstream."""
    offered = candidates("r1", "a", "b", "c")
    results = adjudicated([record()], [offered], {"r1": answer("b", "b", "a")})

    after = apply([offered], results, CONFIG)[0]

    assert not results[0].rejected
    assert after.codes == ("b", "a", "c")


def test_only_the_top_candidates_are_offered():
    """`adjudication.candidates` is the cost of a prompt and the reach of the
    stage: a code below it cannot be promoted, because it was never shown."""
    offered = candidates("r1", *[f"gnd:{index}" for index in range(50)])
    config = dataclasses.replace(CONFIG, candidates=5, select=5)

    text = prompt(record(), offered.candidates[:5], label_texts(*offered.codes), config)

    assert "gnd:4" in text
    assert "gnd:6" not in text


def test_a_code_below_the_offered_window_cannot_be_promoted():
    offered = candidates("r1", *[f"gnd:{index}" for index in range(50)])
    config = dataclasses.replace(CONFIG, candidates=5, select=5)

    result = adjudicated([record()], [offered], {"r1": answer("gnd:40")}, config)[0]

    assert result.rejected


def test_scores_stay_non_increasing_after_reordering():
    offered = candidates("r1", "a", "b", "c", scores=[0.9, 0.5, 0.1])
    results = adjudicated([record()], [offered], {"r1": answer("c")})

    after = apply([offered], results, CONFIG)[0]
    scores = [candidate.score for candidate in after.candidates]

    assert scores == sorted(scores, reverse=True)


def test_a_candidate_keeps_the_retriever_that_found_it():
    offered = candidates("r1", "a", "b", "c")
    results = adjudicated([record()], [offered], {"r1": answer("c", "b")})

    after = apply([offered], results, CONFIG)[0]

    before = {c.code: c.sources for c in offered.candidates}
    for candidate in after.candidates:
        assert candidate.sources == before[candidate.code]


# --- Routing ----------------------------------------------------------------


def test_only_the_least_confident_share_is_routed():
    lists = [
        candidates(f"r{index}", "a", "b", scores=[score, score])
        for index, score in enumerate([0.9, 0.8, 0.7, 0.2, 0.1])
    ]

    assert route(lists, dataclasses.replace(CONFIG, route_fraction=0.4)) == ["r4", "r3"]


def test_the_routed_fraction_is_what_the_config_asks_for():
    lists = [
        candidates(f"r{index}", "a", scores=[index / 100]) for index in range(100)
    ]

    for fraction in (0.1, 0.2, 0.5):
        routed = route(lists, dataclasses.replace(CONFIG, route_fraction=fraction))
        assert len(routed) == int(fraction * 100)


def test_the_routed_count_does_not_lose_a_record_to_binary_arithmetic():
    """`0.29 * 100` is 28.999999999999996; flooring that is a rounding bug
    reported as a routing decision."""
    lists = [candidates(f"r{index}", "a", scores=[index / 100]) for index in range(100)]

    routed = route(lists, dataclasses.replace(CONFIG, route_fraction=0.29))

    assert len(routed) == 29


def test_routing_nothing_is_a_free_ablation():
    """The pass runs, costs no API call, and changes no ranking — which is what
    makes every other routed fraction attributable to the model."""
    lists = [candidates("r1", "a", scores=[0.5])]

    assert route(lists, dataclasses.replace(CONFIG, route_fraction=0.0)) == []


def test_a_record_the_retrievers_found_nothing_for_is_routed_first():
    lists = [
        candidates("r1", "a", "b", scores=[0.9, 0.8]),
        CandidateList("r2", ()),
    ]

    assert route(lists, dataclasses.replace(CONFIG, route_fraction=0.5)) == ["r2"]


def test_routing_can_read_a_confidence_measured_on_another_ranking():
    """docs/results.md: the fused ranking's confidence correlates +0.41 with
    per-record P@5 and the cross-encoder's own relevance −0.05, so the routed
    set must be able to come from the ranking that calibrates, whichever stage
    produced the ranking being reordered."""
    lists = [candidates("r1", "a", scores=[0.1]), candidates("r2", "a", scores=[0.9])]

    routed = route(
        lists, dataclasses.replace(CONFIG, route_fraction=0.5), confidences=[0.9, 0.1]
    )

    assert routed == ["r2"]


def test_routing_is_deterministic_when_records_tie():
    lists = [candidates(f"r{index}", "a", scores=[0.5]) for index in range(4)]
    config = dataclasses.replace(CONFIG, route_fraction=0.5)

    assert route(lists, config) == route(list(reversed(lists)), config)


# --- The response cache -----------------------------------------------------


def test_a_cached_response_is_replayed_rather_than_re_billed():
    offered = candidates("r1", "a", "b")
    model = corpus.FakeLanguageModel()

    adjudicate(
        [record()],
        [offered],
        label_texts("a", "b"),
        CONFIG,
        model=model,
        responses={"r1": answer("b")},
    )

    assert model.prompts == []


def test_a_response_the_model_returns_is_written_to_the_cache():
    offered = candidates("r1", "a", "b")
    responses: dict[str, str] = {}

    adjudicate(
        [record()],
        [offered],
        label_texts("a", "b"),
        CONFIG,
        model=corpus.FakeLanguageModel(),
        responses=responses,
    )

    assert set(responses) == {"r1"}


def test_a_rejected_response_is_still_cached():
    """It was paid for, and re-billing to be told the same thing twice is the
    one thing the cache exists to prevent."""
    offered = candidates("r1", "a", "b")
    responses: dict[str, str] = {}

    adjudicate(
        [record()],
        [offered],
        label_texts("a", "b"),
        CONFIG,
        model=corpus.FakeLanguageModel(answer="[\"gnd:invented\"]"),
        responses=responses,
    )

    assert set(responses) == {"r1"}


def test_a_record_with_no_cached_response_and_no_model_is_named():
    with pytest.raises(ValueError, match="r1"):
        adjudicate(
            [record()], [candidates("r1", "a")], label_texts("a"), CONFIG,
            responses={},
        )


def test_the_cache_is_keyed_by_the_prompt_and_the_model(tmp_path):
    """A prompt change or a model change invalidates the answers to the old one
    rather than quietly serving them under the new name."""
    store = ArtifactStore(tmp_path, data_revision="fixture")
    config = load_experiment_text(ADJUDICATING)
    keyed_by = store.manifest("adjudicated", config)["config"]["adjudication"]

    assert keyed_by["prompt_revision"] == config.adjudication.prompt_revision
    assert keyed_by["model"] == config.adjudication.model
    assert keyed_by["document_chars"] == config.adjudication.document_chars

    other = dataclasses.replace(
        config,
        adjudication=dataclasses.replace(config.adjudication, model="fixture/other"),
    )
    assert store.key("adjudicated", config) != store.key("adjudicated", other)


def test_an_unknown_prompt_revision_is_refused():
    """Otherwise a typo is a new cache key and a prompt nobody wrote."""
    with pytest.raises(ConfigError, match="prompt_revision"):
        load_experiment_text(
            "name: bad-prompt\n"
            "encoder: {name: fixture/fake}\n"
            "adjudication: {enabled: true, model: m, prompt_revision: v99}\n"
        )


# --- Logging the violations -------------------------------------------------


def test_rejections_are_written_where_a_run_can_find_them(tmp_path):
    store = ArtifactStore(tmp_path, data_revision="fixture")
    config = load_experiment_text(ADJUDICATING)
    offered = candidates("r1", "a", "b")
    results = adjudicated([record()], [offered], {"r1": answer("gnd:invented")})

    log_rejections(store, config, results)

    logged = read_rejections(store, config)
    assert [entry["record_id"] for entry in logged] == ["r1"]
    assert "gnd:invented" in logged[0]["reason"]


def test_a_replayed_rejection_is_not_logged_twice(tmp_path):
    """Most of a re-run is replayed from the cache and buys nothing, so a
    violation count that grew with re-scores would not be a violation count."""
    store = ArtifactStore(tmp_path, data_revision="fixture")
    config = load_experiment_text(ADJUDICATING)
    offered = candidates("r1", "a", "b")
    results = adjudicated([record()], [offered], {"r1": answer("gnd:invented")})

    log_rejections(store, config, results)
    log_rejections(store, config, results)

    assert len(read_rejections(store, config)) == 1


def test_a_clean_pass_logs_nothing(tmp_path):
    store = ArtifactStore(tmp_path, data_revision="fixture")
    config = load_experiment_text(ADJUDICATING)
    offered = candidates("r1", "a", "b")
    results = adjudicated([record()], [offered], {"r1": answer("b", "a")})

    log_rejections(store, config, results)

    assert read_rejections(store, config) == []


# --- The model cutoff -------------------------------------------------------


def test_every_committed_adjudicator_is_one_the_registry_vouches_for():
    """The 2025-01-31 cutoff covers the headline LLM too (docs/spec.md)."""
    enabled = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        config = load_experiment(path)
        if not config.adjudication.enabled:
            continue
        enabled.append(path.name)
        resolved = resolve(config.adjudication)
        if not config.adjudication.appendix:
            assert resolved.release.created <= MODEL_CUTOFF, path.name
            assert not resolved.appendix, path.name

    assert enabled, "no committed config enables adjudication"


def test_an_unregistered_adjudicator_is_refused():
    with pytest.raises(UnregisteredModel):
        resolve(dataclasses.replace(CONFIG, model="acme/unknown-llm"))


def test_a_current_model_needs_the_appendix_flag():
    """docs/spec.md allows one clearly-labelled appendix row on a current model
    in this stage alone, so "current" has to be declared rather than slipped in.
    """
    registry = corpus.registry_with_recent_llm()
    config = dataclasses.replace(CONFIG, model="acme/tomorrow-llm")

    with pytest.raises(ModelTooRecent, match="appendix"):
        resolve(config, registry=registry)

    resolved = resolve(dataclasses.replace(config, appendix=True), registry=registry)
    assert resolved.appendix


def test_the_appendix_flag_is_refused_on_a_pre_cutoff_model():
    """Otherwise a headline-eligible run is filed as the model-progress row."""
    registry = corpus.registry_with_recent_llm()

    with pytest.raises(ConfigError, match="appendix"):
        resolve(
            dataclasses.replace(
                CONFIG, model="claude-3-5-sonnet-20241022", appendix=True
            ),
            registry=registry,
        )


# --- The configuration ------------------------------------------------------


def test_enabling_adjudication_without_naming_a_model_is_refused():
    with pytest.raises(ConfigError, match="adjudication.model"):
        load_experiment_text(
            "name: no-model\n"
            "encoder: {name: fixture/fake}\n"
            "adjudication: {enabled: true}\n"
        )


def test_a_route_fraction_outside_zero_to_one_is_refused():
    for fraction in ("-0.1", "1.5"):
        with pytest.raises(ConfigError, match="route_fraction"):
            load_experiment_text(
                "name: bad-fraction\n"
                "encoder: {name: fixture/fake}\n"
                f"adjudication: {{enabled: true, model: m, route_fraction: {fraction}}}\n"
            )


def test_selecting_more_codes_than_are_offered_is_refused():
    with pytest.raises(ConfigError, match="select"):
        load_experiment_text(
            "name: too-many\n"
            "encoder: {name: fixture/fake}\n"
            "adjudication: {enabled: true, model: m, candidates: 5, select: 10}\n"
        )


def test_adjudicating_more_candidates_than_the_reranker_emits_is_refused():
    """Otherwise the run prompts with 30 codes and reports that it prompted
    with the top 30 of a list that is 20 long."""
    with pytest.raises(ConfigError, match="reranker.output_k"):
        load_experiment_text(
            "name: too-deep\n"
            "encoder: {name: fixture/fake}\n"
            "reranker: {enabled: true, model: m, input_k: 20, output_k: 20}\n"
            "adjudication: {enabled: true, model: m, candidates: 30}\n"
        )


def test_adjudicating_more_candidates_than_fusion_emits_is_refused():
    with pytest.raises(ConfigError, match="fusion.candidates"):
        load_experiment_text(
            "name: too-deep\n"
            "encoder: {name: fixture/fake}\n"
            "fusion: {candidates: 20}\n"
            "adjudication: {enabled: true, model: m, candidates: 30}\n"
        )


# --- Through the pipeline seam ----------------------------------------------

KNN_ONLY = """
name: fixture-adjudicate
encoder: {name: fixture/fake}
index: {corpora: [core_train]}
retrievers:
  knn: {enabled: true, neighbours: 5, top_k: 100}
  dense: {enabled: false}
  lexical: {enabled: false}
"""

ADJUDICATION = (
    "\nadjudication: {enabled: true, model: fixture/fake-llm, "
    "route_fraction: 0.5, candidates: 30, select: 10}\n"
)

ADJUDICATING = KNN_ONLY + ADJUDICATION


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


def test_the_pipeline_adjudicates_only_the_routed_records(
    queries, index_records, vocabulary, tmp_path
):
    model = corpus.FakeLanguageModel()
    run(
        queries,
        index_records,
        vocabulary,
        tmp_path,
        extra=ADJUDICATION,
        language_model=model,
    )

    assert len(model.prompts) == int(0.5 * len(queries))


def test_adjudication_keeps_every_contract_the_plain_pipeline_keeps(
    queries, index_records, vocabulary, tmp_path
):
    results = run(
        queries,
        index_records,
        vocabulary,
        tmp_path,
        extra=ADJUDICATION,
        language_model=corpus.FakeLanguageModel(),
    )

    for result in results:
        scores = [candidate.score for candidate in result.candidates]
        assert scores == sorted(scores, reverse=True), result.record_id
        assert len(set(result.codes)) == len(result.codes), result.record_id
        assert not set(result.codes) - set(vocabulary), result.record_id
        for candidate in result.candidates:
            assert candidate.sources


def test_adjudication_changes_the_ranking_of_the_records_it_saw(
    queries, index_records, vocabulary, tmp_path_factory
):
    plain = run(
        queries, index_records, vocabulary, tmp_path_factory.mktemp("plain")
    )
    judged = run(
        queries,
        index_records,
        vocabulary,
        tmp_path_factory.mktemp("judged"),
        extra=ADJUDICATION,
        language_model=corpus.FakeLanguageModel(),
    )

    assert [r.codes for r in judged] != [r.codes for r in plain]


def test_an_unrouted_record_is_returned_untouched(
    queries, index_records, vocabulary, tmp_path_factory
):
    """Half the split is routed here, so the other half is the control."""
    plain = run(
        queries, index_records, vocabulary, tmp_path_factory.mktemp("plain")
    )
    judged = run(
        queries,
        index_records,
        vocabulary,
        tmp_path_factory.mktemp("judged"),
        extra=ADJUDICATION,
        language_model=corpus.FakeLanguageModel(),
    )

    unchanged = [
        after.record_id
        for before, after in zip(plain, judged)
        if before.codes == after.codes
    ]
    assert unchanged


def test_routing_reads_the_fused_confidence_even_when_a_reranker_ran(
    queries, index_records, vocabulary, tmp_path_factory
):
    """The signal that calibrates, not the one in hand.

    Ticket 12 measured the cross-encoder's own relevance to correlate −0.05 with
    per-record precision — its most confident decile has two thirds of its
    records with no correct label — and `mix: replace` writes exactly those
    scores onto the ranking this stage is handed. Routing on them would send the
    model the records the reranker was surest about, which are the ones it
    ruined.
    """
    fused = run(
        queries, index_records, vocabulary, tmp_path_factory.mktemp("fused")
    )
    expected = set(
        route(fused, load_experiment_text(ADJUDICATING).adjudication)
    )

    responses: dict[str, str] = {}
    run(
        queries,
        index_records,
        vocabulary,
        tmp_path_factory.mktemp("reranked"),
        extra="\nreranker: {enabled: true, model: fixture/fake-reranker}\n"
        + ADJUDICATION,
        cross_encoder=corpus.FakeCrossEncoder(),
        language_model=corpus.FakeLanguageModel(),
        responses=responses,
    )

    assert set(responses) == expected


def test_the_pipeline_is_unchanged_with_adjudication_disabled(
    queries, index_records, vocabulary, tmp_path_factory
):
    """The toggle has to be a toggle: off must be exactly the earlier pipeline."""
    plain = run(
        queries, index_records, vocabulary, tmp_path_factory.mktemp("plain")
    )
    off = run(
        queries,
        index_records,
        vocabulary,
        tmp_path_factory.mktemp("off"),
        extra="\nadjudication: {enabled: false, model: fixture/fake-llm}\n",
    )

    assert [r.codes for r in off] == [r.codes for r in plain]


def test_the_pipeline_writes_the_responses_it_paid_for(
    queries, index_records, vocabulary, tmp_path
):
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")
    config = load_experiment_text(ADJUDICATING)

    predict(
        queries,
        config,
        vocabulary,
        index_records,
        store,
        encoder=corpus.FakeEncoder(),
        language_model=corpus.FakeLanguageModel(),
    )

    assert len(load_responses(store, config)) == int(0.5 * len(queries))


def test_a_second_run_replays_the_cache_rather_than_re_billing(
    queries, index_records, vocabulary, tmp_path
):
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")
    config = load_experiment_text(ADJUDICATING)

    for _ in range(2):
        model = corpus.FakeLanguageModel()
        predict(
            queries,
            config,
            vocabulary,
            index_records,
            store,
            encoder=corpus.FakeEncoder(),
            language_model=model,
        )

    assert model.prompts == []


def test_the_pipeline_logs_a_constraint_violation(
    queries, index_records, vocabulary, tmp_path
):
    """The path the ticket exists for, end to end: a model that names a code it
    was not shown must leave evidence rather than a silently scored ranking."""
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")
    config = load_experiment_text(ADJUDICATING)

    results = predict(
        queries,
        config,
        vocabulary,
        index_records,
        store,
        encoder=corpus.FakeEncoder(),
        language_model=corpus.FakeLanguageModel(answer='["gnd:9999999-9"]'),
    )

    logged = read_rejections(store, config)
    assert len(logged) == int(0.5 * len(queries))
    assert all("gnd:9999999-9" in entry["reason"] for entry in logged)
    for result in results:
        assert not set(result.codes) - set(vocabulary)
