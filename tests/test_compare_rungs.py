"""Comparing two encoder screens: what makes the ranking claim honest.

Ticket 10 runs the same four encoders at the full 32,043-document index and asks
one question — did the rung-1 ranking hold? The answer is a claim about two
measurements taken months and hours apart, so the comparator's job is to refuse
every pair of screens where the answer would not mean what it says: a pair whose
encoders differ, whose label text or fusion moved alongside the index, or which
were scored on different splits. Only index size may move between the rungs.

Nothing here loads a model. A screen is a JSON document by then, which is the
whole point of persisting it: the comparison is arithmetic over two files, so it
can be re-run, tested, and re-read after the fact without four GPU-free hours.
"""

import importlib.util
import json
from pathlib import Path

import pytest


def _script(name: str):
    """`scripts/` is not importable as a package, so it is loaded by path."""
    import sys

    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compare = _script("compare_rungs")


HELD_EQUAL = {
    "label_text": {"bilingual": True},
    "retrievers": {"knn": {"weight": 1.5}},
    "fusion": {"candidates": 100, "rrf_k": 60},
    "group_prior": {"enabled": False},
    "reranker": {"enabled": False},
    "adjudication": {"enabled": False},
    "encoder.max_length": 512,
}


def row(encoder: str, recall: float, bands: dict[str, float] | None = None, **over):
    """One encoder's row of a screen document, as the screen writes it."""
    document = {
        "encoder": encoder,
        "config": f"rung-{encoder}",
        "dimensions": 768,
        "parameters": 305_000_000,
        "load_seconds": 1.0,
        "retrieve_seconds": 10.0,
        "wrote": 3,
        "micro": {"10": recall, "50": recall + 0.1},
        "official_macro": {"10": recall + 0.1},
        "bands": {
            band: {"10": value} for band, value in (bands or {"head": recall}).items()
        },
        "per_retriever": {"lexical": {"10": 0.1781}},
        "held_equal": dict(HELD_EQUAL),
    }
    document.update(over)
    return document


def screen(documents: int, rows, **over):
    """A whole screen document at one index size."""
    payload = {
        "schema": compare.SCHEMA,
        "split": "core_dev",
        "records": 5354,
        "data_revision": "abcdef123456",
        "selection_k": 10,
        "device": "mps",
        "index": {
            "documents": documents,
            "corpus": 32043,
            "corpora": ["core_train"],
            "stratify": documents < 32043,
            "seed": 42,
        },
        "rows": list(rows),
    }
    payload.update(over)
    return payload


def written(tmp_path: Path, name: str, payload) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def loaded(tmp_path: Path, documents: int, rows, **over):
    payload = screen(documents, rows, **over)
    return compare.load_screen(written(tmp_path, f"screen-{documents}.json", payload))


# --- Only index size may move ------------------------------------------------


def test_two_screens_differing_only_in_index_size_are_comparable(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(tmp_path, 32043, [row("lab/a", 0.5), row("lab/b", 0.4)])

    assert compare.differences([small, full]) == {}


def test_a_retuned_fusion_weight_between_rungs_is_refused(tmp_path):
    """Rung 2 with retuned weights measures the tuning, not the index."""
    moved = dict(HELD_EQUAL, fusion={"candidates": 100, "rrf_k": 10})
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(
        tmp_path,
        32043,
        [row("lab/a", 0.5, held_equal=moved), row("lab/b", 0.4, held_equal=moved)],
    )

    assert "fusion" in compare.differences([small, full])

    with pytest.raises(compare.NotAComparison, match="fusion"):
        compare.check([small, full])


def test_a_different_label_rendering_between_rungs_is_refused(tmp_path):
    moved = dict(HELD_EQUAL, label_text={"bilingual": False})
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(
        tmp_path,
        32043,
        [row("lab/a", 0.5, held_equal=moved), row("lab/b", 0.4, held_equal=moved)],
    )

    assert "label_text" in compare.differences([small, full])


def test_a_row_out_of_step_with_its_own_screen_is_refused(tmp_path):
    """One encoder retuned inside a rung would rank the tuning within it, too."""
    moved = dict(HELD_EQUAL, fusion={"candidates": 100, "rrf_k": 10})
    full = loaded(
        tmp_path, 32043, [row("lab/a", 0.5), row("lab/b", 0.4, held_equal=moved)]
    )

    assert "fusion" in compare.differences([full])


def test_screens_over_different_encoders_are_not_a_comparison(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(tmp_path, 32043, [row("lab/a", 0.5), row("lab/c", 0.4)])

    with pytest.raises(compare.NotAComparison, match="lab/b"):
        compare.check([small, full])


def test_screens_over_different_splits_are_not_a_comparison(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(
        tmp_path, 32043, [row("lab/a", 0.5), row("lab/b", 0.4)], split="core_train"
    )

    with pytest.raises(compare.NotAComparison, match="core_train"):
        compare.check([small, full])


def test_two_screens_at_the_same_index_size_are_not_two_rungs(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    again = loaded(tmp_path, 8000, [row("lab/a", 0.5), row("lab/b", 0.4)])

    with pytest.raises(compare.NotAComparison, match="8000"):
        compare.check([small, again])


def test_one_screen_is_not_a_comparison(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])

    with pytest.raises(compare.NotAComparison, match="at least two"):
        compare.check([small])


def test_a_screen_of_one_encoder_has_no_ranking_to_compare(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4)])
    full = loaded(tmp_path, 32043, [row("lab/a", 0.5)])

    with pytest.raises(compare.NotAComparison, match="one encoder"):
        compare.check([small, full])


def test_a_foreign_document_is_rejected_by_its_schema(tmp_path):
    path = written(tmp_path, "other.json", {"schema": "something-else", "rows": []})

    with pytest.raises(compare.NotAComparison, match="something-else"):
        compare.load_screen(path)


# --- The screens are ordered by index size, never by argument order ----------


def test_screens_are_read_smallest_index_first(tmp_path):
    full = loaded(tmp_path, 32043, [row("lab/a", 0.5), row("lab/b", 0.4)])
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])

    ordered = compare.by_index_size([full, small])

    assert [each.documents for each in ordered] == [8000, 32043]


# --- Did the ranking hold? ---------------------------------------------------


def test_an_identical_order_at_both_index_sizes_held(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(tmp_path, 32043, [row("lab/a", 0.5), row("lab/b", 0.35)])

    verdict = compare.held([small, full], 10)

    assert verdict.stable
    assert verdict.moves == ()


def test_a_swap_at_the_top_is_a_flip_and_names_what_moved(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(tmp_path, 32043, [row("lab/a", 0.3), row("lab/b", 0.5)])

    verdict = compare.held([small, full], 10)

    assert not verdict.stable
    assert dict(verdict.moves) == {"lab/a": (1, 2), "lab/b": (2, 1)}


def test_the_ranking_is_by_the_selection_metric_best_first(tmp_path):
    full = loaded(
        tmp_path, 32043, [row("lab/a", 0.31), row("lab/b", 0.42), row("lab/c", 0.07)]
    )

    assert compare.ranking(full, 10) == ["lab/b", "lab/a", "lab/c"]


def test_a_tie_is_broken_by_name_rather_than_by_arrival(tmp_path):
    full = loaded(tmp_path, 32043, [row("lab/b", 0.4), row("lab/a", 0.4)])

    assert compare.ranking(full, 10) == ["lab/a", "lab/b"]


# --- Rank correlation --------------------------------------------------------


def test_an_unchanged_order_correlates_perfectly():
    assert compare.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert compare.kendall([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_a_reversed_order_correlates_perfectly_negatively():
    assert compare.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert compare.kendall([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_one_adjacent_swap_of_four_is_measured_not_rounded():
    """The textbook values: rho = 1 - 6*2/(4*15), tau = (5-1)/6."""
    assert compare.spearman([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(0.8)
    assert compare.kendall([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(2 / 3)


def test_tied_scores_share_a_rank_rather_than_taking_arrival_order():
    """Two encoders at the same recall are not evidence of an order."""
    assert compare.ranks([0.4, 0.4, 0.2]) == [1.5, 1.5, 3.0]


def test_kendall_is_tau_b_so_a_tie_cannot_read_as_agreement():
    """Tau-a would divide by all three pairs and read 0.67; tau-b discounts the
    tied one from the x side alone, so agreement is not credited to a tie."""
    assert compare.kendall([1, 1, 3], [1, 2, 3]) == pytest.approx(2 / 6**0.5)


def test_a_correlation_of_two_points_is_undefined_rather_than_one(tmp_path):
    """Two encoders can only agree or reverse, so rho carries no information."""
    assert compare.spearman([1, 2], [1, 2]) is None
    assert compare.kendall([1, 2], [1, 2]) is None


def test_the_correlation_is_computed_between_the_two_rungs(tmp_path):
    small = loaded(
        tmp_path,
        8000,
        [row("lab/a", 0.4), row("lab/b", 0.3), row("lab/c", 0.2), row("lab/d", 0.1)],
    )
    full = loaded(
        tmp_path,
        32043,
        [row("lab/a", 0.5), row("lab/b", 0.4), row("lab/d", 0.3), row("lab/c", 0.2)],
    )

    verdict = compare.held([small, full], 10)

    assert verdict.spearman == pytest.approx(0.8)
    assert verdict.kendall == pytest.approx(2 / 3)


# --- The tables --------------------------------------------------------------


def test_the_ranking_table_puts_both_index_sizes_side_by_side(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(tmp_path, 32043, [row("lab/a", 0.5), row("lab/b", 0.35)])

    table = compare.render_ranking([small, full], 10)

    assert "8,000" in table and "32,043" in table
    assert "0.4000" in table and "0.5000" in table
    # Ordered by the larger index, which is the rung the decision rests on.
    body = [line for line in table.splitlines() if line.startswith("| `lab/")]
    assert [line.split(" | ")[0] for line in body] == ["| `lab/a`", "| `lab/b`"]


def test_the_band_table_reports_every_band_at_both_index_sizes(tmp_path):
    bands = {"head": 0.7, "torso": 0.5, "tail": 0.3, "zero": 0.2}
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4, bands), row("lab/b", 0.3, bands)])
    full = loaded(tmp_path, 32043, [row("lab/a", 0.5, bands), row("lab/b", 0.4, bands)])

    table = compare.render_bands([small, full], 10)

    for band in ("head", "torso", "tail", "zero"):
        assert band in table
    assert table.count("| 8,000 |") == 2
    assert table.count("| 32,043 |") == 2


def test_the_verdict_is_stated_in_words_not_left_to_the_table(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(tmp_path, 32043, [row("lab/a", 0.5), row("lab/b", 0.35)])

    stated = compare.render_verdict(compare.held([small, full], 10))

    assert "held" in stated.lower()


def test_a_flip_is_stated_as_a_flip(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(tmp_path, 32043, [row("lab/a", 0.3), row("lab/b", 0.5)])

    stated = compare.render_verdict(compare.held([small, full], 10))

    assert "flip" in stated.lower()
    assert "lab/b" in stated


def test_the_shortlist_names_two_encoders_from_the_larger_index(tmp_path):
    small = loaded(
        tmp_path,
        8000,
        [row("lab/a", 0.4), row("lab/b", 0.3), row("lab/c", 0.2), row("lab/d", 0.1)],
    )
    full = loaded(
        tmp_path,
        32043,
        [row("lab/a", 0.5), row("lab/b", 0.4), row("lab/d", 0.45), row("lab/c", 0.2)],
    )

    shortlisted = compare.shortlist([small, full], 10)

    assert shortlisted == ("lab/a", "lab/d")


def test_the_shortlist_is_rendered_with_the_evidence_for_each_name(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(tmp_path, 32043, [row("lab/a", 0.5), row("lab/b", 0.35)])

    stated = compare.render_shortlist([small, full], 10)

    assert "lab/a" in stated and "lab/b" in stated
    assert "0.5000" in stated and "0.4000" in stated


# --- The committed screens ---------------------------------------------------


SCREENS = Path(__file__).resolve().parent.parent / "reference" / "screens"


def test_the_committed_screens_are_a_comparison():
    """The two rungs in `reference/screens/` must still compare.

    They are a measurement each, four hours of Apple Silicon apart, and the
    claim they support is read off them by `compare_rungs`. If a config edit
    ever makes them incomparable — a retuned weight, a relabelled index — the
    failure belongs here, at the point where the claim stops holding, rather
    than in a reader's terminal months later.
    """
    paths = sorted(SCREENS.glob("*.json"))
    if len(paths) < 2:
        pytest.skip(f"fewer than two screens in {SCREENS}")

    compare.check([compare.load_screen(path) for path in paths])


def test_the_committed_screens_hold_every_encoder_at_every_index_size():
    paths = sorted(SCREENS.glob("*.json"))
    if len(paths) < 2:
        pytest.skip(f"fewer than two screens in {SCREENS}")

    screens = [compare.load_screen(path) for path in paths]

    assert len({screen.encoders for screen in screens}) == 1
    assert len(screens[0].encoders) == 4


# --- The control: the retrievers that do not read the index ------------------


def with_retrievers(encoder: str, recall: float, per_retriever, **over):
    return row(encoder, recall, per_retriever=per_retriever, **over)


def test_the_per_retriever_table_holds_a_row_per_encoder_and_index_size(tmp_path):
    each = {"knn": {"10": 0.30}, "dense": {"10": 0.20}, "lexical": {"10": 0.1781}}
    small = loaded(
        tmp_path,
        8000,
        [with_retrievers("lab/a", 0.4, each), with_retrievers("lab/b", 0.3, each)],
    )
    full = loaded(
        tmp_path,
        32043,
        [with_retrievers("lab/a", 0.5, each), with_retrievers("lab/b", 0.4, each)],
    )

    table = compare.render_per_retriever([small, full], 10)

    for name in ("knn", "dense", "lexical"):
        assert name in table
    assert table.count("| 8,000 |") == 2
    assert table.count("| 32,043 |") == 2


def test_an_index_independent_retriever_that_did_not_move_is_the_control(tmp_path):
    """Only kNN reads the index, so dense and lexical must be identical."""
    each = {"knn": {"10": 0.30}, "dense": {"10": 0.20}, "lexical": {"10": 0.1781}}
    grew = {**each, "knn": {"10": 0.42}}
    small = loaded(tmp_path, 8000, [with_retrievers("lab/a", 0.4, each)] + [
        with_retrievers("lab/b", 0.3, each)
    ])
    full = loaded(tmp_path, 32043, [with_retrievers("lab/a", 0.5, grew)] + [
        with_retrievers("lab/b", 0.4, grew)
    ])

    assert compare.control_drift([small, full], 10) == {}
    assert "held" in compare.render_per_retriever([small, full], 10).lower()


def test_an_index_independent_retriever_that_moved_is_named(tmp_path):
    """A dense column that moved means something other than the index changed."""
    each = {"knn": {"10": 0.30}, "dense": {"10": 0.20}, "lexical": {"10": 0.1781}}
    moved = {**each, "dense": {"10": 0.25}}
    small = loaded(
        tmp_path,
        8000,
        [with_retrievers("lab/a", 0.4, each), with_retrievers("lab/b", 0.3, each)],
    )
    full = loaded(
        tmp_path,
        32043,
        [with_retrievers("lab/a", 0.5, moved), with_retrievers("lab/b", 0.4, moved)],
    )

    drift = compare.control_drift([small, full], 10)

    assert set(drift) == {"dense"}
    assert "lab/a" in dict(drift["dense"])
    assert "dense" in compare.render_per_retriever([small, full], 10)


def test_a_retriever_missing_from_one_rung_is_not_read_as_drift(tmp_path):
    """A rung that ran kNN alone has no dense column to compare."""
    small = loaded(
        tmp_path,
        8000,
        [
            with_retrievers("lab/a", 0.4, {"knn": {"10": 0.3}}),
            with_retrievers("lab/b", 0.3, {"knn": {"10": 0.2}}),
        ],
    )
    full = loaded(
        tmp_path,
        32043,
        [
            with_retrievers("lab/a", 0.5, {"knn": {"10": 0.4}, "dense": {"10": 0.2}}),
            with_retrievers("lab/b", 0.4, {"knn": {"10": 0.3}, "dense": {"10": 0.2}}),
        ],
    )

    assert compare.control_drift([small, full], 10) == {}


# --- What else may not move between the rungs --------------------------------


def test_a_limited_screen_is_not_comparable_to_a_whole_split(tmp_path):
    """`--limit 500` is a documented flag, and 500 records is another variable."""
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)], records=500)
    full = loaded(tmp_path, 32043, [row("lab/a", 0.5), row("lab/b", 0.4)])

    with pytest.raises(compare.NotAComparison, match="500"):
        compare.check([small, full])


def test_screens_over_different_corpora_are_not_index_size(tmp_path):
    """rung 3 indexes all-subjects train; that is a different corpus, not a size."""
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(tmp_path, 32043, [row("lab/a", 0.5), row("lab/b", 0.4)])
    full = compare.load_screen(
        written(
            tmp_path,
            "other-corpus.json",
            screen(
                70588,
                [row("lab/a", 0.5), row("lab/b", 0.4)],
                index={
                    "documents": 70588,
                    "corpus": 70588,
                    "corpora": ["all_train"],
                    "stratify": False,
                    "seed": 42,
                },
            ),
        )
    )

    with pytest.raises(compare.NotAComparison, match="all_train"):
        compare.check([small, full])


def test_an_empty_shortlist_is_refused_at_the_flag(tmp_path):
    small = written(tmp_path, "a.json", screen(8000, [row("lab/a", 0.4), row("lab/b", 0.3)]))
    full = written(tmp_path, "b.json", screen(32043, [row("lab/a", 0.5), row("lab/b", 0.4)]))

    with pytest.raises(SystemExit):
        compare.main([str(small), str(full), "--shortlist", "0"])


def test_an_undefined_correlation_says_which_of_its_two_causes_it_was(tmp_path):
    """Four encoders tied at one rung is not "too few encoders"."""
    small = loaded(
        tmp_path,
        8000,
        [row("lab/a", 0.4), row("lab/b", 0.4), row("lab/c", 0.4), row("lab/d", 0.4)],
    )
    full = loaded(
        tmp_path,
        32043,
        [row("lab/a", 0.5), row("lab/b", 0.4), row("lab/c", 0.3), row("lab/d", 0.2)],
    )

    stated = compare.render_verdict(compare.held([small, full], 10))

    assert "every encoder scored the same" in stated
    assert "below the" not in stated


def test_two_encoders_are_reported_as_too_few_for_a_coefficient(tmp_path):
    small = loaded(tmp_path, 8000, [row("lab/a", 0.4), row("lab/b", 0.3)])
    full = loaded(tmp_path, 32043, [row("lab/a", 0.5), row("lab/b", 0.35)])

    stated = compare.render_verdict(compare.held([small, full], 10))

    assert "below the 3" in stated
