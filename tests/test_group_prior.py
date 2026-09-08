"""The group prior: a distribution over the 66 groups, used as a boost.

The assertions here are about what the boost may and may not do to a candidate
list, because that is the whole risk of the stage. docs/spec.md story 29 asks
for the predicted groups to *boost* candidates rather than filter them, so that
the records whose labels span three or more groups are not permanently lost —
and a boost that could drop a candidate, or introduce one, would be a filter
wearing the word "boost".

Nothing here asserts a trained model's numbers. The classifier's weights are a
property of an encoder this suite must not download, so the fitted model appears
only in `test_training_fits_a_separable_problem`, over vectors that are
separable by construction.
"""

import dataclasses

import numpy as np
import pytest

from llms4subjects.config import ConfigError, GroupPriorConfig
from llms4subjects.contracts import Candidate, CandidateList, VocabularyEntry
from llms4subjects.stages import group_prior

GROUPS = ("Chemie", "Mathematik", "Physik")

CONFIG = GroupPriorConfig(enabled=True, weight=1.0)

GROUP_OF_CODE = {
    "gnd:1": "Chemie",
    "gnd:2": "Mathematik",
    "gnd:3": "Physik",
    "gnd:4": "Chemie",
}


def candidates(scores: dict[str, float]) -> list[CandidateList]:
    """One record, whose candidates carry the scores given, best first."""
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    return [
        CandidateList(
            record_id="r1",
            candidates=tuple(
                Candidate(code=code, score=score, sources={"dense": rank})
                for rank, (code, score) in enumerate(ordered)
            ),
        )
    ]


FUSED = candidates({"gnd:1": 0.04, "gnd:2": 0.03, "gnd:3": 0.02, "gnd:4": 0.01})

# A prior certain the record belongs to the group of the two weakest candidates.
CERTAIN = [{"Chemie": 0.0, "Mathematik": 0.0, "Physik": 1.0}]

UNIFORM = [{group: 1 / len(GROUPS) for group in GROUPS}]


def boost(distributions, config=CONFIG, lists=FUSED):
    return group_prior.apply_prior(lists, distributions, GROUP_OF_CODE, config)


# --- The boost is a boost, not a filter -------------------------------------


def test_the_boost_returns_the_same_codes_it_was_given():
    """Never removes, never introduces: the ceiling is fusion's, not the prior's."""
    boosted = boost(CERTAIN)

    assert set(boosted[0].codes) == set(FUSED[0].codes)
    assert len(boosted[0].candidates) == len(FUSED[0].candidates)


def test_the_boost_keeps_one_list_per_record_in_order():
    lists = FUSED + [dataclasses.replace(FUSED[0], record_id="r2")]
    boosted = group_prior.apply_prior(
        lists, CERTAIN * 2, GROUP_OF_CODE, CONFIG
    )

    assert [result.record_id for result in boosted] == ["r1", "r2"]


def test_the_boost_reorders_towards_the_likely_group():
    """The point of the stage: a likely group's candidate climbs the list."""
    boosted = boost(CERTAIN)

    assert boosted[0].codes[0] == "gnd:3"


def test_the_boost_leaves_the_scores_non_increasing():
    boosted = boost(CERTAIN)

    scores = [candidate.score for candidate in boosted[0].candidates]
    assert scores == sorted(scores, reverse=True)


def test_the_boost_keeps_every_candidates_provenance():
    """Attribution is the project's deliverable; a rescoring may not erase it."""
    before = {c.code: c.sources for c in FUSED[0].candidates}
    boosted = boost(CERTAIN)

    assert {c.code: c.sources for c in boosted[0].candidates} == before


def test_the_prior_never_names_itself_as_a_retriever():
    """`sources` maps retriever to rank, and the prior retrieved nothing."""
    boosted = boost(CERTAIN)

    for candidate in boosted[0].candidates:
        assert set(candidate.sources) <= {"knn", "dense", "lexical"}


# --- When it must change nothing --------------------------------------------


def test_a_uniform_distribution_changes_no_ranking():
    """Every candidate boosted by the same amount is the same ranking."""
    boosted = boost(UNIFORM)

    assert boosted[0].codes == FUSED[0].codes


def test_a_zero_weight_is_the_identity():
    """The ablation row where the prior runs and contributes nothing."""
    boosted = boost(CERTAIN, dataclasses.replace(CONFIG, weight=0.0))

    assert boosted[0].candidates == FUSED[0].candidates


def test_the_weight_scales_how_far_a_candidate_can_climb():
    weak = boost(CERTAIN, dataclasses.replace(CONFIG, weight=0.05))
    strong = boost(CERTAIN, dataclasses.replace(CONFIG, weight=1.0))

    assert weak[0].codes[0] == "gnd:1"
    assert strong[0].codes[0] == "gnd:3"


def test_an_empty_candidate_list_is_not_an_error():
    """The ablation that disables every retriever still reaches this stage."""
    empty = [CandidateList("r1", ())]

    assert group_prior.apply_prior(empty, CERTAIN, GROUP_OF_CODE, CONFIG) == empty


def test_a_group_the_prior_does_not_name_is_not_a_boost():
    """An unscored group must read as "unlikely", never as an exception."""
    partial = [{"Physik": 1.0}]

    boosted = boost(partial)

    assert boosted[0].codes[0] == "gnd:3"


def test_a_record_count_mismatch_is_refused():
    """Positional, like fusion: boosting one record's list with another's prior."""
    with pytest.raises(ValueError, match="records"):
        group_prior.apply_prior(FUSED, CERTAIN * 2, GROUP_OF_CODE, CONFIG)


# --- Reading the groups off the vocabulary ----------------------------------


def entry(code: str, group: str) -> VocabularyEntry:
    return VocabularyEntry(code=code, name=code, classification_name=group)


def test_the_groups_come_from_the_vocabularys_own_classification_names():
    vocabulary = {
        "gnd:1": entry("gnd:1", "Chemie"),
        "gnd:2": entry("gnd:2", "Mathematik"),
        "gnd:3": entry("gnd:3", "Chemie"),
    }

    assert group_prior.group_of_code(vocabulary.values()) == {
        "gnd:1": "Chemie",
        "gnd:2": "Mathematik",
        "gnd:3": "Chemie",
    }
    assert group_prior.groups(vocabulary.values()) == ("Chemie", "Mathematik")


def test_a_label_with_no_classification_name_is_left_ungrouped():
    """Every tib-core entry has one; a vocabulary slice that does not may not
    silently acquire a group, because the boost would then be reading a guess."""
    ungrouped = {"gnd:9": VocabularyEntry(code="gnd:9", name="Kraft")}

    assert group_prior.group_of_code(ungrouped.values()) == {}


def test_a_records_groups_are_the_groups_of_its_gold_labels():
    from llms4subjects.contracts import Record

    records = [
        Record(id="a", type="Book", lang="de", title="t", abstract="",
               subjects=("gnd:1", "gnd:4", "gnd:2")),
        Record(id="b", type="Book", lang="de", title="t", abstract="",
               subjects=("gnd:99",)),
    ]

    assert group_prior.record_groups(records, GROUP_OF_CODE) == [
        ("Chemie", "Mathematik"),
        (),
    ]


# --- The trained head -------------------------------------------------------


def separable(count: int = 90, dimensions: int = 8):
    """Vectors on three axes, one axis per group: separable by construction."""
    rng = np.random.default_rng(0)
    vectors, truth = [], []
    for index in range(count):
        group = index % len(GROUPS)
        vector = rng.normal(scale=0.05, size=dimensions)
        vector[group] += 1.0
        vectors.append(vector)
        truth.append((GROUPS[group],))
    return np.array(vectors, dtype=np.float32), truth


def test_training_fits_a_separable_problem():
    vectors, truth = separable()

    prior = group_prior.train(vectors, truth, GROUPS, CONFIG)
    distributions = prior.distributions(vectors)

    assert len(distributions) == len(vectors)
    top = [max(row, key=row.get) for row in distributions]
    assert top == [groups[0] for groups in truth]


def test_a_distribution_covers_every_group_and_sums_to_one():
    """66 groups every time, so a boost never reads a missing key as certainty."""
    vectors, truth = separable()

    prior = group_prior.train(vectors, truth, GROUPS, CONFIG)

    for row in prior.distributions(vectors):
        assert set(row) == set(GROUPS)
        assert row  # a distribution over no groups would boost nothing
        assert sum(row.values()) == pytest.approx(1.0)
        assert all(value >= 0.0 for value in row.values())


def test_a_group_with_no_training_document_stays_reachable_and_unlikely():
    """One of the 66 groups has no tib-core train document at all.

    It must not raise — a group missing from the index is what a smaller rung
    looks like — and it must not become the most likely group for everything.
    """
    vectors, truth = separable()

    prior = group_prior.train(vectors, truth, GROUPS + ("Unbelegt",), CONFIG)
    distributions = prior.distributions(vectors)

    assert all("Unbelegt" in row for row in distributions)
    assert not any(max(row, key=row.get) == "Unbelegt" for row in distributions)


def test_a_record_with_no_groups_is_not_a_training_target():
    """A record whose gold labels are all out of vocabulary teaches nothing."""
    vectors, truth = separable()
    truth = [() if index == 0 else groups for index, groups in enumerate(truth)]

    prior = group_prior.train(vectors, truth, GROUPS, CONFIG)

    assert prior.distributions(vectors[:1])


def test_training_refuses_a_row_count_mismatch():
    vectors, truth = separable()

    with pytest.raises(ValueError, match="documents"):
        group_prior.train(vectors, truth[:-1], GROUPS, CONFIG)


def test_the_trained_head_survives_a_round_trip_through_plain_arrays():
    """Persisted as arrays rather than a pickle, so a run months later reads it."""
    vectors, truth = separable()
    prior = group_prior.train(vectors, truth, GROUPS, CONFIG)

    restored = group_prior.GroupPrior(
        groups=prior.groups,
        coefficients=prior.coefficients.copy(),
        intercepts=prior.intercepts.copy(),
    )

    assert restored.distributions(vectors) == prior.distributions(vectors)


def test_vectors_of_the_wrong_width_are_refused():
    """A prior trained on one encoder's vectors cannot read another's."""
    vectors, truth = separable()
    prior = group_prior.train(vectors, truth, GROUPS, CONFIG)

    with pytest.raises(ValueError, match="dimensions"):
        prior.distributions(np.zeros((2, 3), dtype=np.float32))


# --- Accuracy, which is a deliverable of the ticket -------------------------


def test_accuracy_reports_how_often_the_true_groups_fall_within_the_top_two():
    distributions = [{"Chemie": 0.6, "Mathematik": 0.3, "Physik": 0.1}] * 4
    truth = [
        ("Chemie",),                 # the top group, alone
        ("Mathematik", "Chemie"),    # both true groups within the top two
        ("Physik",),                 # outside the top two
        ("Mathematik",),             # the second group, alone
    ]

    report = group_prior.accuracy(distributions, truth, depths=(1, 2))

    assert report.records == 4
    assert report.any_within[1] == pytest.approx(2 / 4)
    assert report.all_within[1] == pytest.approx(1 / 4)
    assert report.any_within[2] == pytest.approx(3 / 4)
    assert report.all_within[2] == pytest.approx(3 / 4)
    assert report.groups_per_record == pytest.approx(5 / 4)


def test_accuracy_skips_records_with_no_true_group():
    distributions = [{"Chemie": 0.9, "Mathematik": 0.1}] * 2
    truth = [("Chemie",), ()]

    report = group_prior.accuracy(distributions, truth, depths=(1,))

    assert report.records == 1
    assert report.any_within[1] == pytest.approx(1.0)


def test_accuracy_over_no_scored_records_is_reported_as_zero_records():
    report = group_prior.accuracy([{"Chemie": 1.0}], [()], depths=(1,))

    assert report.records == 0
    assert report.any_within[1] == 0.0


# --- The configuration ------------------------------------------------------


def test_a_negative_weight_is_refused():
    """A negative weight would boost the groups the prior finds unlikely."""
    with pytest.raises(ConfigError, match="weight"):
        GroupPriorConfig(enabled=True, weight=-1.0)


def test_a_non_positive_regularization_is_refused():
    with pytest.raises(ConfigError, match="regularization"):
        GroupPriorConfig(enabled=True, regularization=0.0)


# --- Persistence ------------------------------------------------------------


def test_a_fitted_head_round_trips_through_the_artifact_store(tmp_path):
    """The path every boosted run reads: `predict` loads the head, never fits it."""
    from llms4subjects.artifacts import (
        ArtifactStore,
        MissingGroupPrior,
        load_group_prior,
        save_group_prior,
    )
    from llms4subjects.config import load_experiment_text

    config = load_experiment_text(
        "name: fixture\nencoder: {name: fixture/fake}\n"
        "group_prior: {enabled: true, weight: 0.5}\n"
    )
    store = ArtifactStore(tmp_path / "artifacts", data_revision="fixture")

    with pytest.raises(MissingGroupPrior, match="train_group_prior"):
        load_group_prior(store, config)

    vectors, truth = separable()
    fitted = group_prior.train(vectors, truth, GROUPS, CONFIG)
    save_group_prior(store, config, fitted)
    restored = load_group_prior(store, config)

    assert restored.groups == fitted.groups
    assert restored.distributions(vectors) == fitted.distributions(vectors)
