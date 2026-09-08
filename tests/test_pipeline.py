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
from llms4subjects.config import FusionConfig, load_experiment_text
from llms4subjects.contracts import CODES_PER_RECORD
from llms4subjects.pipeline import predict

# kNN only, which is the base every other configuration below varies from.
KNN_ONLY = """
name: fixture-knn
encoder: {name: fixture/fake}
index: {corpora: [core_train]}
retrievers:
  knn: {enabled: true, neighbours: 5, top_k: 100}
  dense: {enabled: false}
  lexical: {enabled: false}
"""


# What candidate generation emits per record, which is what the reranker and
# the recall ceiling are measured over. The submission takes the top 50 of it.
CANDIDATES = FusionConfig().candidates


RETRIEVER_NAMES = ("knn", "dense", "lexical")


def retrievers(off: tuple[str, ...] = (), weights: dict[str, float] | None = None) -> str:
    """A complete `retrievers:` block with everything but `off` enabled.

    Written out in full rather than as an override of the block above, because
    the loader fills every retriever it is not told about with its defaults:
    naming only the one being turned off would also move kNN's `neighbours`
    from 5 back to 20, and an ablation that changes two things measures neither.
    """
    weights = weights or {}
    lines = ["", "retrievers:"]
    for name in RETRIEVER_NAMES:
        settings = [f"enabled: {str(name not in off).lower()}", "top_k: 100"]
        if name == "knn":
            settings.append("neighbours: 5")
        settings.append(f"weight: {weights.get(name, 1.0)}")
        lines.append(f"  {name}: {{{', '.join(settings)}}}")
    return "\n".join(lines) + "\n"


# All three retrievers, fused: the rung-1 configuration, and the one every
# ablation below removes something from.
ALL_THREE = retrievers()


# The dense retriever alone, which is the only configuration that can return a
# label no indexed record carries.
DENSE_ONLY = """
retrievers:
  knn: {enabled: false}
  dense: {enabled: true, top_k: 100}
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


@pytest.fixture(scope="module")
def vocabulary_size(vocabulary):
    """Every entry is a label text, so the tower costs one encode per code."""
    return len(vocabulary)


def run(
    queries, index_records, vocabulary, tmp_path, extra="", encoder=None, **kwargs
):
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")
    return predict(
        queries,
        config(extra),
        vocabulary,
        index_records,
        store,
        encoder=encoder or corpus.FakeEncoder(),
        **kwargs,
    )


@pytest.fixture(scope="module")
def candidates(queries, index_records, vocabulary, tmp_path_factory):
    return run(
        queries, index_records, vocabulary, tmp_path_factory.mktemp("predict")
    )


# --- The output contract ----------------------------------------------------


def test_one_candidate_list_per_record_in_the_order_given(candidates, queries):
    assert [result.record_id for result in candidates] == [r.id for r in queries]


def test_enough_codes_per_record_to_fill_a_submission(candidates):
    """Candidate generation emits up to 100; the submission takes the top 50.

    Not exactly 100 here: this is one retriever whose candidates are then cut to
    the tib-core vocabulary, so the ceiling is what the index can reach. What
    must hold is the floor — the submission format has no representation for
    fewer than 50 codes.
    """
    lengths = {len(result.candidates) for result in candidates}
    assert min(lengths) >= CODES_PER_RECORD
    assert max(lengths) <= CANDIDATES


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


# --- Fusion -----------------------------------------------------------------


@pytest.fixture(scope="module")
def all_three(queries, index_records, vocabulary, tmp_path_factory):
    return run(
        queries,
        index_records,
        vocabulary,
        tmp_path_factory.mktemp("fused"),
        extra=ALL_THREE,
    )


def test_fusion_emits_the_configured_hundred_candidates_per_record(all_three):
    """The pipeline contract: candidate generation emits the top 100."""
    assert {len(result.candidates) for result in all_three} == {CANDIDATES}


def test_fused_candidates_keep_the_contract_the_single_retrievers_keep(
    all_three, queries, vocabulary
):
    assert [result.record_id for result in all_three] == [r.id for r in queries]
    for result in all_three:
        scores = [candidate.score for candidate in result.candidates]
        assert scores == sorted(scores, reverse=True), result.record_id
        assert len(set(result.codes)) == len(result.codes), result.record_id
        assert not set(result.codes) - set(vocabulary), result.record_id


def test_a_fused_candidate_names_every_retriever_that_proposed_it(all_three):
    """Attribution is the point: a fused list that forgets its sources is useless."""
    agreed = 0
    for result in all_three:
        for candidate in result.candidates:
            assert set(candidate.sources) <= {"knn", "dense", "lexical"}
            assert candidate.sources
            assert all(rank >= 0 for rank in candidate.sources.values())
            agreed += len(candidate.sources) > 1

    assert agreed, "no candidate was proposed by more than one retriever"


@pytest.mark.parametrize("removed", ["knn", "dense", "lexical"])
def test_removing_any_single_retriever_changes_the_output(
    queries, index_records, vocabulary, tmp_path, all_three, removed
):
    """Otherwise a retriever is carrying no weight and the ablation would not say so."""
    without = run(
        queries,
        index_records,
        vocabulary,
        tmp_path,
        extra=retrievers(off=(removed,)),
    )

    assert [r.codes for r in without] != [r.codes for r in all_three]


@pytest.mark.parametrize("kept", RETRIEVER_NAMES)
def test_fusion_does_not_reproduce_any_single_retrievers_ranking(
    queries, index_records, vocabulary, tmp_path, all_three, kept
):
    """Three retrievers combined, rather than one of them wearing three flags."""
    alone = run(
        queries,
        index_records,
        vocabulary,
        tmp_path,
        extra=retrievers(off=tuple(o for o in RETRIEVER_NAMES if o != kept)),
    )

    assert [r.codes for r in alone] != [r.codes for r in all_three]


def test_the_fusion_weights_change_the_fused_ranking(
    queries, index_records, vocabulary, tmp_path, all_three
):
    """Weights are tuned on dev, so a weight nothing reads would be a silent default."""
    reweighted = run(
        queries,
        index_records,
        vocabulary,
        tmp_path,
        extra=retrievers(weights={"dense": 9.0}),
    )

    assert [r.codes for r in reweighted] != [r.codes for r in all_three]


def test_rrf_k_changes_what_a_rank_is_worth(
    queries, index_records, vocabulary, tmp_path, all_three
):
    steep = run(
        queries,
        index_records,
        vocabulary,
        tmp_path,
        extra=ALL_THREE + "\nfusion: {rrf_k: 0}\n",
    )

    assert [r.codes for r in steep] != [r.codes for r in all_three]


# --- Bilingual label text ---------------------------------------------------


GERMAN_ONLY = DENSE_ONLY + "\nlabel_text: {bilingual: false}\n"


def test_the_bilingual_flag_moves_the_candidates(
    queries, index_records, vocabulary, tmp_path_factory
):
    """The ablation is a real one, which is what the ticket-08 refusal held out for.

    `bilingual: false` renders the same vocabulary German-only, so a difference
    here is the second language reaching the label tower and nothing else. If
    these two ever agreed, every "translation does not help" row in the results
    would be a report on a flag nobody read.
    """
    german = run(
        queries, index_records, vocabulary, tmp_path_factory.mktemp("de"),
        extra=GERMAN_ONLY,
    )
    both = run(
        queries, index_records, vocabulary, tmp_path_factory.mktemp("both"),
        extra=DENSE_ONLY,
    )
    assert [result.codes for result in german] != [result.codes for result in both]


def test_german_only_reads_no_translation_cache(
    queries, index_records, vocabulary, tmp_path_factory, monkeypatch
):
    """The ablation must run on a checkout that has no cache to read."""
    import llms4subjects.corpus as corpus

    def refuse(*args, **kwargs):
        raise AssertionError("German-only rendering loaded the translation cache")

    monkeypatch.setattr(corpus, "load_label_translations", refuse)
    run(
        queries, index_records, vocabulary, tmp_path_factory.mktemp("no-cache"),
        extra=GERMAN_ONLY,
    )


def test_translations_given_explicitly_are_the_ones_used(
    queries, index_records, vocabulary, tmp_path_factory, monkeypatch
):
    """A caller with its own cache does not also pay for the frozen one."""
    import llms4subjects.corpus as corpus

    def refuse(*args, **kwargs):
        raise AssertionError("an explicit translation map did not stop the load")

    monkeypatch.setattr(corpus, "load_label_translations", refuse)
    supplied = run(
        queries, index_records, vocabulary, tmp_path_factory.mktemp("supplied"),
        extra=DENSE_ONLY,
        translations={entry.name: f"{entry.name} translated" for entry in vocabulary.values()},
    )
    assert [result.record_id for result in supplied] == [r.id for r in queries]


# --- The dense label tower --------------------------------------------------


@pytest.fixture(scope="module")
def dense(queries, index_records, vocabulary, tmp_path_factory):
    return run(
        queries,
        index_records,
        vocabulary,
        tmp_path_factory.mktemp("dense"),
        extra=DENSE_ONLY,
    )


def test_the_dense_retriever_fills_the_output_contract(dense, queries):
    assert [result.record_id for result in dense] == [r.id for r in queries]
    assert {len(result.candidates) for result in dense} == {CANDIDATES}
    for result in dense:
        scores = [candidate.score for candidate in result.candidates]
        assert scores == sorted(scores, reverse=True), result.record_id
        assert len(set(result.codes)) == len(result.codes), result.record_id


def test_dense_candidates_are_attributed_to_the_dense_retriever(dense):
    for result in dense:
        for candidate in result.candidates:
            assert set(candidate.sources) == {"dense"}


def test_the_dense_retriever_returns_labels_the_index_does_not_carry(
    dense, index_records
):
    """The whole reason the ticket exists: a label needs no training example.

    kNN cannot do this at any index size or any k — `rung1-knn` scores 0.0000 in
    the zero-shot band by construction (docs/results.md).
    """
    harvestable = {code for record in index_records for code in record.subjects}
    unreachable = {code for result in dense for code in result.codes} - harvestable

    assert unreachable


def test_the_dense_retriever_stays_inside_the_vocabulary(dense, vocabulary):
    for result in dense:
        assert not set(result.codes) - set(vocabulary), result.record_id


def test_dense_and_knn_do_not_return_the_same_ranking(dense, candidates):
    """Two mechanisms, not one behind two flags."""
    assert [r.codes for r in dense] != [r.codes for r in candidates]


def test_label_embeddings_are_cached_across_runs(
    queries, index_records, vocabulary, vocabulary_size, tmp_path
):
    """Keyed by encoder and label-text revision, through `CachedEncoder`."""
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")
    first, second = corpus.FakeEncoder(), corpus.FakeEncoder()

    for encoder in (first, second):
        predict(
            queries,
            config(DENSE_ONLY),
            vocabulary,
            index_records,
            store,
            encoder=encoder,
        )

    assert first.encoded == len(queries) + vocabulary_size
    assert second.encoded == 0


def test_a_different_label_rendering_is_not_served_the_cached_vectors(
    queries, index_records, vocabulary, vocabulary_size, tmp_path
):
    """`qualifiers` changes the label text, so it must change the vectors."""
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")
    first, second = corpus.FakeEncoder(), corpus.FakeEncoder()

    predict(queries, config(DENSE_ONLY), vocabulary, index_records, store,
            encoder=first)
    predict(
        queries,
        config(DENSE_ONLY + "\nlabel_text: {qualifiers: stripped}\n"),
        vocabulary,
        index_records,
        store,
        encoder=second,
    )

    assert second.encoded == vocabulary_size


def test_the_definition_field_is_an_ablation_rather_than_a_default(
    queries, index_records, vocabulary, tmp_path
):
    """Cataloguing instructions on 18.1% of labels; measured, not assumed."""
    default = run(queries, index_records, vocabulary, tmp_path / "off",
                  extra=DENSE_ONLY)
    with_definition = run(
        queries,
        index_records,
        vocabulary,
        tmp_path / "on",
        extra=DENSE_ONLY + "\nlabel_text: {include_definition: true}\n",
    )

    assert [r.codes for r in with_definition] != [r.codes for r in default]


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


def test_a_knn_only_run_reads_no_translation_cache(
    queries, index_records, vocabulary, tmp_path, monkeypatch
):
    """The neighbour harvest renders no label, so no label-text flag reaches it."""
    import llms4subjects.corpus as corpus

    def refuse(*args, **kwargs):
        raise AssertionError("a kNN-only run loaded the translation cache")

    monkeypatch.setattr(corpus, "load_label_translations", refuse)
    run(queries, index_records, vocabulary, tmp_path)


# --- The subject-area group prior -------------------------------------------


PRIOR_ON = ALL_THREE + "\ngroup_prior: {enabled: true, weight: 1.0}\n"
PRIOR_OFF_BY_WEIGHT = ALL_THREE + "\ngroup_prior: {enabled: true, weight: 0.0}\n"


def fake_prior(vocabulary):
    """A head with arbitrary weights, of the width the fake encoder emits.

    The prior's own accuracy is a property of a trained classifier and belongs
    to `tests/test_group_prior.py`; what the seam has to assert is what a boost
    may do to a candidate list, which holds for any weights at all.
    """
    import numpy as np

    from llms4subjects.stages.group_prior import GroupPrior, groups

    names = groups(vocabulary.values())
    rng = np.random.default_rng(11)
    return GroupPrior(
        groups=names,
        coefficients=rng.normal(size=(len(names), corpus.DIMENSIONS)),
        intercepts=np.zeros(len(names)),
    )


@pytest.fixture(scope="module")
def boosted(queries, index_records, vocabulary, tmp_path_factory):
    return run(
        queries,
        index_records,
        vocabulary,
        tmp_path_factory.mktemp("boosted"),
        extra=PRIOR_ON,
        prior=fake_prior(vocabulary),
    )


def test_the_group_prior_reorders_and_removes_nothing(boosted, all_three):
    """docs/spec.md story 29: a boost, so that a record whose labels span three
    or more groups is not permanently lost. A stage that could drop a candidate
    would be a filter wearing the word "boost"."""
    for before, after in zip(all_three, boosted):
        assert after.record_id == before.record_id
        assert set(after.codes) == set(before.codes)


def test_the_group_prior_keeps_the_candidate_count(boosted):
    assert {len(result.candidates) for result in boosted} == {CANDIDATES}


def test_the_group_prior_moves_the_ranking(boosted, all_three):
    """Otherwise the flag reports an ablation of something nothing read."""
    assert [r.codes for r in boosted] != [r.codes for r in all_three]


def test_the_boosted_ranking_keeps_the_output_contract(boosted, vocabulary):
    for result in boosted:
        scores = [candidate.score for candidate in result.candidates]
        assert scores == sorted(scores, reverse=True), result.record_id
        assert len(set(result.codes)) == len(result.codes), result.record_id
        assert not set(result.codes) - set(vocabulary), result.record_id
        for candidate in result.candidates:
            assert set(candidate.sources) <= {"knn", "dense", "lexical"}


def test_a_zero_weight_prior_leaves_the_fused_ranking_alone(
    queries, index_records, vocabulary, tmp_path, all_three
):
    """The row where the prior runs and contributes nothing, which is not the
    same row as the prior being off."""
    unboosted = run(
        queries,
        index_records,
        vocabulary,
        tmp_path,
        extra=PRIOR_OFF_BY_WEIGHT,
        prior=fake_prior(vocabulary),
    )

    assert [r.codes for r in unboosted] == [r.codes for r in all_three]


def test_the_prior_reads_no_gold_label_of_the_record_it_boosts(
    queries, index_records, vocabulary, tmp_path, fixture
):
    """The invariant the whole file exists for, through the new stage."""
    perturbed = [
        dataclasses.replace(
            record, subjects=(fixture["unseen_code"],) + record.subjects[:1]
        )
        for record in queries
    ]
    prior = fake_prior(vocabulary)

    before = run(queries, index_records, vocabulary, tmp_path / "before",
                 extra=PRIOR_ON, prior=prior)
    after = run(perturbed, index_records, vocabulary, tmp_path / "after",
                extra=PRIOR_ON, prior=prior)

    assert [r.codes for r in after] == [r.codes for r in before]


def test_an_enabled_prior_with_no_fitted_head_refuses_rather_than_skipping(
    queries, index_records, vocabulary, tmp_path
):
    """A silently skipped boost would be filed as a measurement of the boost."""
    from llms4subjects.artifacts import MissingGroupPrior

    with pytest.raises(MissingGroupPrior, match="train_group_prior"):
        run(queries, index_records, vocabulary, tmp_path, extra=PRIOR_ON)


def test_the_prior_survives_every_retriever_being_disabled(
    queries, index_records, vocabulary, tmp_path, vocabulary_size
):
    """The all-off ablation reaches this stage with empty lists."""
    results = run(
        queries,
        index_records,
        vocabulary,
        tmp_path,
        extra="\nretrievers: {knn: {enabled: false}, dense: {enabled: false}, "
        "lexical: {enabled: false}}\ngroup_prior: {enabled: true}\n",
        prior=fake_prior(vocabulary),
    )

    assert all(result.candidates == () for result in results)


def test_an_enabled_prior_loads_the_configured_encoder_itself(
    queries, index_records, vocabulary, tmp_path, monkeypatch
):
    """The one branch `run` cannot reach, because it always supplies an encoder.

    A boosted run has to embed the records it is boosting, and it must do that
    with the encoder the config names rather than one the caller happened to
    hand in — otherwise the head would read vectors from a different model than
    the one it was fitted on.
    """
    from llms4subjects.stages import encoders

    loaded = []

    def fake_load(config, device):
        loaded.append((config.name, device))
        return corpus.FakeEncoder()

    monkeypatch.setattr(encoders, "load", fake_load)
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")

    results = predict(
        queries,
        config(PRIOR_ON),
        vocabulary,
        index_records,
        store,
        prior=fake_prior(vocabulary),
    )

    assert [name for name, _ in loaded] == ["fixture/fake"]
    assert {len(result.candidates) for result in results} == {CANDIDATES}


def test_a_stage_that_cannot_run_is_refused_before_any_model_loads(
    queries, index_records, vocabulary, tmp_path, monkeypatch
):
    """A refusal after a model load is a refusal that cost a download.

    The prior resolves its encoder before retrieval, so the refusals have to
    come first or enabling the prior would move them behind a model load. The
    adjudicator is the last stage and the most expensive one to discover late:
    a run that indexed, retrieved and fused before failing on an unregistered
    model name has spent hours on rung 2 to learn a spelling.
    """
    from llms4subjects.models import UnregisteredModel
    from llms4subjects.stages import encoders

    def refuse_load(*args, **kwargs):
        raise AssertionError("a refused configuration loaded model weights")

    monkeypatch.setattr(encoders, "load", refuse_load)
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")

    with pytest.raises(UnregisteredModel):
        predict(
            queries,
            config(
                PRIOR_ON
                + "\nadjudication: {enabled: true, model: acme/unregistered}\n"
            ),
            vocabulary,
            index_records,
            store,
            prior=fake_prior(vocabulary),
        )
