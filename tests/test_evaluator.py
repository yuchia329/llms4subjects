"""Seam 2: `evaluate(gold, predictions) -> metrics`.

The only test that protects the headline number. Correctness is established by
equivalence against the organizers' own script rather than by expectations
typed in here, because a hand-picked expectation only proves that two people
made the same arithmetic choice.
"""

import official_scorer as scorer
import pytest

from llms4subjects.stages.evaluator import BANDS, OFFICIAL_KS, evaluate, render

METRICS = ("precision", "recall", "f1")
TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def entries():
    return scorer.fixture()["records"]


@pytest.fixture(scope="module")
def bands(entries):
    """Bands for the fixture's labels, spanning all four of them.

    Assigned here rather than read from the frozen reference so that this file
    tests aggregation; that the real assignment is read and never recomputed is
    tested in test_frequency_bands.py.
    """
    codes = sorted({code for entry in entries for code in entry["gold"]})
    return {code: BANDS[index % len(BANDS)] for index, code in enumerate(codes)}


@pytest.fixture(scope="module")
def report(entries, bands):
    return evaluate(
        scorer.gold(entries),
        scorer.predictions(entries),
        bands,
        scorer.cells(entries),
    )


@pytest.fixture(scope="module")
def official(entries, tmp_path_factory):
    module = scorer.official()
    return scorer.official_sheets(
        module, entries, tmp_path_factory.mktemp("official")
    )


# --- Equivalence with the official scorer -----------------------------------


@pytest.mark.parametrize("k", OFFICIAL_KS)
@pytest.mark.parametrize("metric", METRICS)
def test_official_macro_matches_the_scorers_overall_row(official, report, k, metric):
    combined = official["combined"]
    overall = combined[combined["Record Type"] == "Overall"].iloc[0]

    assert report.official_macro.at_k[k][metric] == pytest.approx(
        overall[f"{metric}_{k}"], abs=TOLERANCE
    )


@pytest.mark.parametrize("k", OFFICIAL_KS)
@pytest.mark.parametrize("metric", METRICS)
def test_per_cell_metrics_match_the_scorers_cell_rows(official, report, k, metric):
    combined = official["combined"]
    rows = combined[combined["Record Type"] != "Overall"]
    assert len(rows) == len(report.by_cell)

    for _, row in rows.iterrows():
        cell = (row["Record Type"], row["Language"])
        assert report.by_cell[cell].at_k[k][metric] == pytest.approx(
            row[f"{metric}_{k}"], abs=TOLERANCE
        )


@pytest.mark.parametrize("k", OFFICIAL_KS)
@pytest.mark.parametrize("metric", METRICS)
def test_record_type_slice_matches_the_scorers_record_type_sheet(
    official, report, k, metric
):
    rows = official["record_type"]
    assert len(rows) == len(report.by_type)

    for _, row in rows.iterrows():
        assert report.by_type[row["Record Type"]].at_k[k][metric] == pytest.approx(
            row[f"{metric}_{k}"], abs=TOLERANCE
        )


@pytest.mark.parametrize("k", OFFICIAL_KS)
@pytest.mark.parametrize("metric", METRICS)
def test_language_slice_matches_the_scorers_language_sheet(official, report, k, metric):
    rows = official["language"]
    assert len(rows) == len(report.by_language)

    for _, row in rows.iterrows():
        assert report.by_language[row["Language"]].at_k[k][metric] == pytest.approx(
            row[f"{metric}_{k}"], abs=TOLERANCE
        )


def test_the_fixture_covers_the_cases_that_separate_implementations(entries):
    profiles = {entry["profile"] for entry in entries}
    assert {"gold-larger-than-k", "no-hits", "perfect-within-k"} <= profiles

    cells = {(entry["type"], entry["lang"]) for entry in entries}
    assert len({record_type for record_type, _ in cells}) >= 3
    assert {"de", "en"} <= {language for _, language in cells}

    counts = {}
    for entry in entries:
        cell = (entry["type"], entry["lang"])
        counts[cell] = counts.get(cell, 0) + 1
    assert 1 in counts.values(), "no single-record cell, so macro/micro cannot diverge"


# --- Hand-computed edges ----------------------------------------------------


def score_one(gold, predicted, bands=None, cell=("Book", "en"), ks=(5,)):
    return evaluate(
        {"r": list(gold)},
        {"r": list(predicted)},
        bands or {},
        {"r": cell},
        ks=ks,
    )


def test_gold_set_larger_than_k_caps_recall_not_precision():
    gold = [f"gnd:{n}" for n in range(8)]
    report = score_one(gold, gold)

    assert report.micro.precision(5) == pytest.approx(1.0)
    assert report.micro.recall(5) == pytest.approx(5 / 8)


def test_no_hits_scores_zero_everywhere_without_dividing_by_zero():
    report = score_one(["gnd:1", "gnd:2"], [f"gnd:x{n}" for n in range(5)])

    for metrics in (report.micro, report.official_macro):
        assert metrics.precision(5) == metrics.recall(5) == metrics.f1(5) == 0.0


def test_a_perfect_prediction_within_k_reaches_full_recall():
    gold = ["gnd:1", "gnd:2", "gnd:3"]
    report = score_one(gold, gold + ["gnd:x", "gnd:y"])

    assert report.micro.recall(5) == pytest.approx(1.0)
    assert report.micro.precision(5) == pytest.approx(3 / 5)


def test_a_missing_prediction_scores_zero_as_the_official_scorer_does():
    report = evaluate({"r": ["gnd:1"]}, {}, {}, {"r": ("Book", "en")}, ks=(5,))

    assert report.micro.recall(5) == 0.0
    assert report.scored == 1


def test_records_without_gold_are_excluded_and_named():
    report = evaluate(
        {"a": ["gnd:1"], "b": []},
        {"a": ["gnd:1"], "b": ["gnd:1"]},
        {},
        {"a": ("Book", "en"), "b": ("Book", "en")},
        ks=(5,),
    )

    assert report.scored == 1
    assert report.without_gold == ("b",)
    assert report.micro.recall(5) == pytest.approx(1.0)


def test_a_record_outside_the_cell_map_is_an_error_not_a_silent_omission():
    with pytest.raises(KeyError):
        evaluate({"r": ["gnd:1"]}, {"r": ["gnd:1"]}, {}, {}, ks=(5,))


def test_a_single_record_cell_carries_the_same_weight_as_a_large_one():
    gold = {"a": ["gnd:1"], "b": ["gnd:1"], "c": ["gnd:2"]}
    predictions = {"a": ["gnd:1"], "b": ["gnd:1"], "c": ["gnd:x"]}
    cells = {"a": ("Book", "en"), "b": ("Book", "en"), "c": ("Report", "de")}

    report = evaluate(gold, predictions, {}, cells, ks=(5,))

    # Micro sees two of three records right; the official aggregation averages
    # a perfect two-record cell with a failed one-record cell.
    assert report.micro.recall(5) == pytest.approx(2 / 3)
    assert report.official_macro.recall(5) == pytest.approx(0.5)


def test_ks_are_configurable_and_default_to_the_official_sweep(entries, bands, report):
    assert report.ks == OFFICIAL_KS
    assert tuple(report.micro.at_k) == OFFICIAL_KS

    narrow = evaluate(
        scorer.gold(entries), scorer.predictions(entries), bands, scorer.cells(entries), ks=(3,)
    )
    assert narrow.ks == (3,)


# --- Slices -----------------------------------------------------------------


def test_every_band_is_reported_even_when_it_holds_nothing():
    report = score_one(["gnd:1"], ["gnd:1"], bands={"gnd:1": "head"})

    assert tuple(report.by_band) == BANDS
    assert report.by_band["head"].recall(5) == pytest.approx(1.0)
    assert report.by_band["tail"].assignments == 0
    assert report.by_band["tail"].recall(5) == 0.0


def test_band_recall_counts_only_that_bands_gold_labels():
    gold = ["gnd:head", "gnd:tail1", "gnd:tail2"]
    bands = {"gnd:head": "head", "gnd:tail1": "tail", "gnd:tail2": "tail"}
    report = score_one(gold, ["gnd:head", "gnd:tail1", "gnd:x", "gnd:y", "gnd:z"], bands=bands)

    assert report.by_band["head"].recall(5) == pytest.approx(1.0)
    assert report.by_band["tail"].recall(5) == pytest.approx(0.5)


def test_a_label_the_frozen_reference_does_not_name_falls_in_the_zero_band():
    report = score_one(["gnd:never-seen"], ["gnd:never-seen"], bands={})

    assert report.by_band["zero"].assignments == 1
    assert report.by_band["zero"].recall(5) == pytest.approx(1.0)


def test_band_precisions_sum_to_the_overall_precision(report):
    """The claim `_by_band` makes about precision, held to.

    Recall is the number the writeup reads, but precision is reported too, and
    the only additive reading of a per-band precision is the share of the k
    slots that landed on that band. Four numbers that each look like a
    precision and do not sum to one are worse than not reporting them.
    """
    for k in OFFICIAL_KS:
        assert sum(
            report.by_band[band].precision(k) for band in BANDS
        ) == pytest.approx(report.micro.precision(k), abs=TOLERANCE)


def test_band_recall_ignores_records_holding_no_label_of_that_band():
    """Adding a record with no tail gold must not move tail recall."""
    gold = {"a": ["gnd:tail"], "b": ["gnd:head"]}
    bands = {"gnd:tail": "tail", "gnd:head": "head"}
    cells = {"a": ("Book", "en"), "b": ("Book", "en")}

    alone = evaluate({"a": gold["a"]}, {"a": ["gnd:tail"]}, bands, cells, ks=(5,))
    together = evaluate(gold, {"a": ["gnd:tail"], "b": ["gnd:x"]}, bands, cells, ks=(5,))

    assert together.by_band["tail"].recall(5) == alone.by_band["tail"].recall(5) == 1.0
    assert together.by_band["tail"].records == 1


def test_band_assignments_cover_every_gold_assignment_exactly_once(report, entries):
    assignments = sum(len(set(entry["gold"])) for entry in entries)

    assert sum(report.by_band[band].assignments for band in BANDS) == assignments


def test_slices_partition_the_scored_records(report, entries):
    assert sum(metrics.records for metrics in report.by_cell.values()) == report.scored
    assert sum(metrics.records for metrics in report.by_type.values()) == report.scored
    assert (
        sum(metrics.records for metrics in report.by_language.values()) == report.scored
    )


# --- Divergence -------------------------------------------------------------


def test_the_two_aggregations_diverge_and_the_gap_is_reported(report):
    delta = report.divergence.delta

    assert tuple(delta) == OFFICIAL_KS
    for k in OFFICIAL_KS:
        for metric in METRICS:
            assert delta[k][metric] == pytest.approx(
                report.official_macro.at_k[k][metric] - report.micro.at_k[k][metric],
                abs=TOLERANCE,
            )
    assert any(
        abs(delta[k]["recall"]) > TOLERANCE for k in OFFICIAL_KS
    ), "a fixture with a single-record cell must show some divergence"


def test_cell_contributions_are_ranked_and_sum_to_one(report):
    cells = report.divergence.cells

    assert len(cells) == len(report.by_cell)
    assert [cell.mean_recall_share for cell in cells] == sorted(
        (cell.mean_recall_share for cell in cells), reverse=True
    )
    for k in OFFICIAL_KS:
        total = sum(cell.recall_share[k] for cell in cells)
        assert total == pytest.approx(1.0, abs=1e-6)


def test_small_cells_are_over_weighted_relative_to_their_record_share(report):
    smallest = min(report.divergence.cells, key=lambda cell: cell.records)

    assert smallest.weight == pytest.approx(1 / len(report.by_cell))
    assert smallest.weight > smallest.record_share


def test_dominant_cells_are_the_shortest_list_reaching_the_requested_share(report):
    dominant = report.divergence.dominant(0.5)

    assert sum(cell.mean_recall_share for cell in dominant) >= 0.5
    assert sum(cell.mean_recall_share for cell in dominant[:-1]) < 0.5


# --- Reporting --------------------------------------------------------------


def test_render_puts_the_two_aggregations_side_by_side(report):
    text = render(report)

    assert "micro" in text
    assert "official" in text
    # All three metrics, because the leaderboard is quoted on precision as well.
    for metric in METRICS:
        assert metric in text
    for band in BANDS:
        assert band in text
    for k in report.ks:
        assert f"R@{k}" in text


def test_render_narrows_to_the_requested_ks_and_slice_metric(report):
    text = render(report, ks=(5, 10), slice_metric="precision")

    assert "R@50" not in text
    assert "band [micro]" in text
    # The slice tables switch metric; the headline block always shows all three.
    assert text.count("P@5") == 1 + 3

    with pytest.raises(ValueError, match="slice_metric"):
        render(report, slice_metric="accuracy")
