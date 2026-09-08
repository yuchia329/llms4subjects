"""Reciprocal rank fusion: what combining the retrievers is supposed to mean.

Like the lexical matcher, fusion is arithmetic this repository owns rather than
a model wrapper, so it is testable on its own terms. What is asserted here is
never a particular fused float — docs/spec.md rules that out, and the weights
are retuned whenever a retriever changes — but the four properties the design
rests on: rank is what is combined rather than score, agreement between
retrievers is worth something, a weight moves a retriever's influence, and
every surviving candidate remembers where it came from.

The pipeline-level contract — 100 candidates per record, provenance surviving
the whole call, removing a retriever changing the output — lives in
test_pipeline.py.
"""

import pytest

from llms4subjects.config import FusionConfig
from llms4subjects.contracts import Candidate, CandidateList
from llms4subjects.stages.fusion import fuse

CONFIG = FusionConfig(candidates=100, rrf_k=60)


def ranked(record_id: str, name: str, *codes: str) -> CandidateList:
    """One retriever's list for one record: codes best first, scores descending.

    The scores descend steeply on purpose. Fusion reads rank and not score, so
    a retriever whose scores are two orders of magnitude larger than another's
    must not thereby win, and every list built here has scale a fused-by-score
    implementation would be visibly wrong about.
    """
    return CandidateList(
        record_id=record_id,
        candidates=tuple(
            Candidate(code=code, score=1000.0 / (rank + 1), sources={name: rank})
            for rank, code in enumerate(codes)
        ),
    )


def fused(lists: dict[str, list[CandidateList]], weights=None, config=CONFIG):
    weights = weights or {name: 1.0 for name in lists}
    return fuse(lists, weights, config)


# --- Combining ranked lists -------------------------------------------------


def test_a_code_both_retrievers_found_outranks_one_only_one_found():
    """Agreement is the whole reason to fuse rather than concatenate."""
    result = fused(
        {
            "knn": [ranked("r1", "knn", "a", "b", "c")],
            "dense": [ranked("r1", "dense", "c", "d", "e")],
        }
    )[0]

    assert result.codes[0] == "c"


def test_the_fused_list_holds_every_code_any_retriever_proposed():
    result = fused(
        {
            "knn": [ranked("r1", "knn", "a", "b")],
            "dense": [ranked("r1", "dense", "c")],
        }
    )[0]

    assert set(result.codes) == {"a", "b", "c"}
    assert len(result.codes) == len(set(result.codes))


def test_scores_are_non_increasing():
    result = fused(
        {
            "knn": [ranked("r1", "knn", "a", "b", "c", "d")],
            "dense": [ranked("r1", "dense", "d", "c", "e", "f")],
        }
    )[0]

    scores = [candidate.score for candidate in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_a_retrievers_own_order_survives_when_it_is_the_only_one():
    """With one list in, fusion re-scores but must not re-order."""
    result = fused({"knn": [ranked("r1", "knn", "a", "b", "c")]})[0]

    assert result.codes == ("a", "b", "c")


def test_ties_are_broken_by_code_rather_than_by_retriever_order():
    """Two codes at the same rank in different lists score identically.

    Breaking that tie by whichever retriever the mapping happened to iterate
    first would make a ranking depend on dict order, which is the same defect
    the dense retriever's cutoff handling exists to avoid.
    """
    lists = {
        "knn": [ranked("r1", "knn", "b")],
        "dense": [ranked("r1", "dense", "a")],
    }

    forward = fused(lists)[0]
    reversed_order = fused({name: lists[name] for name in reversed(list(lists))})[0]

    assert forward.codes == ("a", "b") == reversed_order.codes


# --- rrf_k, which sets what a rank is worth ---------------------------------


def test_a_large_rrf_k_lets_agreement_beat_a_single_retrievers_top_rank():
    """`rrf_k` is the knob between "rank matters" and "agreement matters"."""
    lists = {
        "knn": [ranked("r1", "knn", "top", *[f"c{i}" for i in range(19)], "agreed")],
        "dense": [ranked("r1", "dense", *[f"d{i}" for i in range(19)], "agreed")],
    }

    flat = fused(lists, config=FusionConfig(rrf_k=60))[0].codes
    steep = fused(lists, config=FusionConfig(rrf_k=0))[0].codes

    # `top` is one retriever's first place; `agreed` is twentieth in both.
    assert flat.index("agreed") < flat.index("top")
    assert steep.index("top") < steep.index("agreed")


# --- Weights ----------------------------------------------------------------


def test_a_weight_of_zero_removes_a_retrievers_influence_on_the_ranking():
    lists = {
        "knn": [ranked("r1", "knn", "a", "b", "c")],
        "dense": [ranked("r1", "dense", "c", "b", "a")],
    }

    result = fused(lists, weights={"knn": 1.0, "dense": 0.0})[0]

    assert result.codes == ("a", "b", "c")


def test_raising_a_retrievers_weight_promotes_the_codes_it_ranks_first():
    lists = {
        "knn": [ranked("r1", "knn", "a", "b")],
        "dense": [ranked("r1", "dense", "b", "a")],
    }

    knn_led = fused(lists, weights={"knn": 2.0, "dense": 1.0})[0]
    dense_led = fused(lists, weights={"knn": 1.0, "dense": 2.0})[0]

    assert knn_led.codes[0] == "a"
    assert dense_led.codes[0] == "b"


def test_a_missing_weight_is_refused_by_name():
    """A weight silently defaulted to 1.0 would be reported as a tuned one."""
    with pytest.raises(KeyError, match="dense"):
        fuse(
            {
                "knn": [ranked("r1", "knn", "a")],
                "dense": [ranked("r1", "dense", "b")],
            },
            {"knn": 1.0},
            CONFIG,
        )


# --- Provenance -------------------------------------------------------------


def test_the_rank_fused_in_is_the_position_in_the_list_fusion_was_handed():
    """Not the rank the candidate remembers, which predates the restriction.

    A retriever stamps its own position before the pipeline cuts its candidates
    to the tib-core vocabulary, so by the time fusion sees the list those
    numbers can have gaps in them. Scoring by the stale number would penalise a
    candidate for a neighbour that was filtered out from under it, and writing
    it into the fused provenance would report a rank the list does not have.
    """
    stale = CandidateList(
        "r1",
        (
            Candidate(code="a", score=0.9, sources={"knn": 7}),
            Candidate(code="b", score=0.8, sources={"knn": 30}),
        ),
    )

    result = fused({"knn": [stale]})[0]

    assert [candidate.sources for candidate in result.candidates] == [
        {"knn": 0},
        {"knn": 1},
    ]


def test_a_candidate_that_names_no_retriever_is_still_ranked_by_its_position():
    """No fabricated rank 0: an unstamped candidate is where the list puts it."""
    unstamped = CandidateList(
        "r1",
        (
            Candidate(code="a", score=0.9),
            Candidate(code="b", score=0.8),
        ),
    )

    result = fused({"knn": [unstamped]})[0]

    assert result.codes == ("a", "b")
    assert [candidate.sources for candidate in result.candidates] == [
        {"knn": 0},
        {"knn": 1},
    ]


def test_every_fused_candidate_carries_the_rank_each_retriever_gave_it():
    """Attribution is the project's central claim, so it survives the fusion."""
    result = fused(
        {
            "knn": [ranked("r1", "knn", "a", "shared")],
            "dense": [ranked("r1", "dense", "shared", "b")],
        }
    )[0]

    sources = {candidate.code: candidate.sources for candidate in result.candidates}
    assert sources["shared"] == {"knn": 1, "dense": 0}
    assert sources["a"] == {"knn": 0}
    assert sources["b"] == {"dense": 1}


# --- Shape of the output ----------------------------------------------------


def test_the_fused_list_is_cut_to_the_configured_candidate_count():
    codes = [f"c{i}" for i in range(200)]
    result = fused(
        {"knn": [ranked("r1", "knn", *codes)]}, config=FusionConfig(candidates=100)
    )[0]

    assert len(result.candidates) == 100
    assert result.codes == tuple(codes[:100])


def test_records_come_back_in_the_order_the_retrievers_returned_them():
    lists = {
        "knn": [ranked("r1", "knn", "a"), ranked("r2", "knn", "b")],
        "dense": [ranked("r1", "dense", "c"), ranked("r2", "dense", "d")],
    }

    assert [result.record_id for result in fused(lists)] == ["r1", "r2"]


def test_a_record_one_retriever_ranked_nothing_for_still_gets_the_others():
    """The lexical retriever returns nothing for a record with no heading in it."""
    lists = {
        "knn": [ranked("r1", "knn", "a", "b")],
        "lexical": [CandidateList("r1", ())],
    }

    assert fused(lists)[0].codes == ("a", "b")


def test_no_retrievers_at_all_fuses_to_nothing():
    """The all-off ablation is a row in the table; the pipeline supplies its ids."""
    assert fuse({}, {}, CONFIG) == []


def test_retrievers_that_disagree_about_the_records_are_refused():
    """Mismatched lists would otherwise fuse one record's codes into another's."""
    with pytest.raises(ValueError, match="r2"):
        fused(
            {
                "knn": [ranked("r1", "knn", "a")],
                "dense": [ranked("r2", "dense", "b")],
            }
        )
