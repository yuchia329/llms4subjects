"""The final test run's harness, at the seams that need no model weights.

The run itself is four hours of Apple Silicon and happens once, so what is
tested here is everything around the prediction: that the official scorer really
is run over a submission tree and agrees with the local evaluator on the cells
both can see, that the committed plan still describes the committed configs, and
that the report carries the rows ticket 17 requires — the published leaderboard,
the duplicate caveat, the zero-shot band, and the cells the organizers' script
cannot read.
"""

import importlib.util
import json
from pathlib import Path

import official_scorer as scorer
import pytest

from llms4subjects.config import load_experiment
from llms4subjects.contracts import CandidateList, Candidate, Cell, Record
from llms4subjects.corpus import frequency_bands
from llms4subjects.stages.evaluator import evaluate
from llms4subjects.testset import configuration_digest, load_plan

REPO_ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    """`scripts/` is not importable as a package, so it is loaded by path."""
    import sys

    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def final_test():
    return _script("final_test")


@pytest.fixture(scope="module")
def entries():
    return scorer.fixture()["records"]


def records(entries):
    return [
        Record(
            id=entry["id"],
            type=entry["type"],
            lang=entry["lang"],
            title="t",
            abstract="a",
            subjects=tuple(entry["gold"]),
        )
        for entry in entries
    ]


def candidates(entries, length=50):
    """Fifty ranked codes per record: the fixture's, padded to the format."""
    lists = []
    for entry in entries:
        codes = list(entry["predictions"])[:length]
        padding = [f"gnd:pad-{entry['id']}-{n}" for n in range(length - len(codes))]
        lists.append(
            CandidateList(
                entry["id"],
                tuple(
                    Candidate(code=code, score=float(length - rank))
                    for rank, code in enumerate(codes + padding)
                ),
            )
        )
    return lists


# --- The configuration was fixed before the split was read --------------------


def test_the_committed_plan_still_describes_the_committed_configs(final_test):
    """The refusal that protects the headline, asserted against the real files.

    If this fails, either a config drifted after the plan was fixed — which the
    run itself refuses — or the plan was rewritten, which `--fix-plan` refuses
    once the receipt exists. Either way the headline is no longer a number
    chosen in advance, and that has to break a test rather than a habit.
    """
    plan = load_plan()

    for row, path in final_test.ROWS.items():
        assert row in plan["rows"], row
        assert plan["rows"][row]["config"] == path
        assert plan["rows"][row]["digest"] == configuration_digest(
            load_experiment(REPO_ROOT / path)
        ), row


def test_the_headline_row_is_the_one_without_adjudication(final_test):
    """docs/spec.md allows the appendix row a model the cutoff does not cover."""
    plan = load_plan()
    headline = load_experiment(REPO_ROOT / final_test.ROWS[plan["headline"]])

    assert not headline.adjudication.enabled


def test_every_planned_row_shares_the_headlines_candidate_generation(final_test):
    """A row that also moved the retrievers would not isolate its own stage."""
    plan = load_plan()
    headline = load_experiment(REPO_ROOT / final_test.ROWS[plan["headline"]])

    for row, path in final_test.ROWS.items():
        other = load_experiment(REPO_ROOT / path)
        for section in ("encoder", "index", "label_text", "retrievers", "fusion"):
            assert other.section(section) == headline.section(section), (row, section)


def test_fixing_the_plan_again_is_refused_once_the_split_has_been_read(
    final_test, tmp_path, monkeypatch, capsys
):
    """Otherwise a decision could be re-dated after its answer was seen."""
    monkeypatch.setattr(final_test, "TEST_RUN_FILE", tmp_path / "test_run.json")
    monkeypatch.setattr(final_test, "TEST_PLAN_FILE", tmp_path / "test_plan.json")
    (tmp_path / "test_run.json").write_text(json.dumps({"read": "2026-09-09"}))

    assert final_test.fix_plan({}) == 1
    assert "already been read" in capsys.readouterr().out
    assert not (tmp_path / "test_plan.json").exists()


# --- The official scorer, over a tree this run wrote --------------------------


def test_the_official_scorer_is_run_over_the_submission_tree(
    final_test, entries, tmp_path
):
    scorer.official()  # skip the module if pandas/openpyxl are absent
    official = final_test.score_officially(
        records(entries), candidates(entries), {}, tmp_path
    )

    assert (tmp_path / "results" / "final_test.xlsx").exists()
    assert official.records == len(entries)
    assert 0.0 <= official.recall(10) <= 1.0


def test_the_local_evaluator_agrees_with_the_scorer_on_the_same_cells(
    final_test, entries, tmp_path
):
    """The headline is quoted from their script; this says ours reproduces it."""
    scorer.official()
    official = final_test.score_officially(
        records(entries), candidates(entries), {}, tmp_path
    )
    local = evaluate(
        gold=scorer.gold(entries),
        predictions={
            result.record_id: result.codes for result in candidates(entries)
        },
        bands=frequency_bands(),
        cells=scorer.cells(entries),
    )

    for k in (5, 10, 50):
        assert official.recall(k) == pytest.approx(
            local.official_macro.recall(k), abs=1e-4
        )


def test_cells_the_scorer_cannot_read_are_left_out_of_its_trees(
    final_test, entries, tmp_path
):
    """Their reader raises on them, so the run scores them locally instead."""
    scorer.official()
    unreadable = {Cell("Book", "fr"): (entries[0]["id"],)}

    official = final_test.score_officially(
        records(entries), candidates(entries), unreadable, tmp_path
    )

    assert official.records == len(entries) - 1
    assert not list((tmp_path / "pred").glob("*/fr"))


# --- The report carries what the ticket asks for ------------------------------


@pytest.fixture(scope="module")
def scored(final_test, entries, tmp_path_factory):
    scorer.official()
    workspace = tmp_path_factory.mktemp("scored")
    gold = scorer.gold(entries)
    predictions = {
        result.record_id: result.codes for result in candidates(entries)
    }
    bands = frequency_bands()
    cells = scorer.cells(entries)
    duplicate = entries[0]["id"]

    return final_test.Scored(
        row="headline",
        config_path="configs/test.yaml",
        config=load_experiment(REPO_ROOT / "configs" / "test.yaml"),
        report=evaluate(gold, predictions, bands, cells),
        without_duplicates=evaluate(
            {id: codes for id, codes in gold.items() if id != duplicate},
            predictions,
            bands,
            cells,
        ),
        official=final_test.score_officially(
            records(entries), candidates(entries), {}, workspace
        ),
        submission=workspace / "pred",
        elapsed=1.0,
    )


def test_the_leaderboard_table_carries_the_four_published_rows(final_test, scored):
    rendered = final_test.render_leaderboard([scored], scored)

    for team, *_ in final_test.LEADERBOARD:
        assert team in rendered
    assert "**this run — headline**" in rendered


def test_the_leaderboard_row_is_the_scorers_own_figure(final_test, scored):
    rendered = final_test.render_leaderboard([scored], scored)

    assert f"{scored.official.recall(10):.4f}" in rendered
    assert f"{scored.official.mean_recall:.4f}" in rendered


def test_both_aggregations_are_reported_side_by_side(final_test, scored):
    rendered = final_test.render_aggregations(scored)

    assert "record-micro" in rendered
    assert "official-macro, every cell" in rendered
    assert "official scorer, de/en cells" in rendered


def test_the_duplicate_caveat_is_a_row_with_both_figures(final_test, scored):
    facts = final_test.Facts(
        records=4910,
        duplicates=("one-id",),
        train_duplicates=("one-id",),
        unreadable={},
    )

    rendered = final_test.render_duplicates([scored], facts)

    assert "without" in rendered
    assert f"{scored.report.micro.recall(10):.4f}" in rendered
    assert f"{scored.without_duplicates.micro.recall(10):.4f}" in rendered


def test_the_duplicate_row_reconciles_the_index_set_with_the_stated_163(
    final_test, scored
):
    """The spec's figure is against `core_train`; this run indexes more than that."""
    facts = final_test.Facts(
        records=4910,
        duplicates=tuple(f"id-{n}" for n in range(180)),
        train_duplicates=tuple(f"id-{n}" for n in range(163)),
        unreadable={},
    )

    rendered = final_test.render_duplicates([scored], facts)

    assert "180 records" in rendered
    assert "163" in rendered


def test_every_band_is_reported_including_zero_shot(final_test, scored):
    rendered = final_test.render_bands([scored])

    for band in ("head", "torso", "tail", "zero"):
        assert f"{band} R@10" in rendered


def test_the_cells_the_official_scorer_cannot_read_are_named(final_test):
    rendered = final_test.render_unreadable(
        final_test.Facts(
            records=4910,
            duplicates=(),
            train_duplicates=(),
            unreadable={
                Cell("Book", "fr"): ("a", "b"),
                Cell("Thesis", "ja"): ("c",),
            },
        )
    )

    assert "Book / fr" in rendered
    assert "Thesis / ja" in rendered
    assert "3 records" in rendered


def test_a_blocked_row_is_labelled_rather_than_scored_as_zero(final_test):
    blocked = final_test.Blocked(
        row="adjudicated",
        config_path="configs/test-adjudicate.yaml",
        reason="ANTHROPIC_API_KEY is not set, and the adjudication stage calls anthropic.",
    )

    rendered = final_test.render_blocked([blocked])
    assert "adjudicated" in rendered
    assert "ANTHROPIC_API_KEY" in rendered
    assert final_test.row_document(blocked)["blocked"].startswith("ANTHROPIC_API_KEY")


def test_the_run_document_carries_both_aggregations_and_every_band(
    final_test, scored
):
    document = final_test.row_document(scored)

    assert document["digest"] == configuration_digest(scored.config)
    assert set(document["bands"]) == {"head", "torso", "tail", "zero"}
    assert document["micro"]["10"]["recall"] == pytest.approx(
        scored.report.micro.recall(10)
    )
    assert document["official_scorer"]["at_k"]["10"]["recall"] == pytest.approx(
        scored.official.recall(10)
    )
    assert "without_duplicates" in document


def test_a_row_the_official_scorer_did_not_score_keeps_the_tables_shape(
    final_test, scored
):
    """A ragged row renders as a different table, which reads as a different run."""
    import dataclasses

    rendered = final_test.render_leaderboard(
        [dataclasses.replace(scored, official=None)], None
    )
    lines = rendered.splitlines()
    row = [line for line in lines if line.startswith("| this run")][0]
    separator = [line for line in lines if line.startswith("|---")][0]

    assert row.count("|") == separator.count("|")


def test_the_unregistered_appendix_row_is_named_rather_than_left_out(final_test):
    """A results table missing the row would read as a pipeline without the stage."""
    rendered = final_test.render_blocked(
        [final_test.Blocked("adjudicated", "configs/test-adjudicate.yaml", "no key")]
    )

    assert "appendix" in rendered
    assert "verify_model_releases.py" in rendered
