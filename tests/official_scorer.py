"""Access to the organizers' evaluation script, imported as shipped.

The script lives in `official_eval/` under its published name, hyphens and all,
so it cannot be imported by name; loading it by path is the point rather than a
workaround. It is vendored unmodified: the moment it is edited to make a test
pass, the test stops meaning anything.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from llms4subjects.contracts import Record
from llms4subjects.stages import submission

REPO_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_SCRIPT = REPO_ROOT / "official_eval" / "llms4subjects-evaluation.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "evaluation.json"


def official() -> ModuleType:
    """The organizers' script as a module, skipping if its deps are absent."""
    pytest.importorskip("pandas", reason="the official scorer is written against pandas")
    pytest.importorskip("openpyxl", reason="the official scorer writes .xlsx")
    if not OFFICIAL_SCRIPT.exists():
        pytest.skip(f"{OFFICIAL_SCRIPT} is not vendored in this checkout")

    spec = importlib.util.spec_from_file_location("official_eval", OFFICIAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture() -> dict:
    """The committed gold-and-predictions fixture."""
    return json.loads(FIXTURE.read_text())


def gold(entries: list[dict]) -> dict[str, list[str]]:
    return {entry["id"]: list(entry["gold"]) for entry in entries}


def predictions(entries: list[dict]) -> dict[str, list[str]]:
    return {entry["id"]: list(entry["predictions"]) for entry in entries}


def cells(entries: list[dict]) -> dict[str, tuple[str, str]]:
    return {entry["id"]: (entry["type"], entry["lang"]) for entry in entries}


def write_gold_tree(entries: list[dict], destination: Path) -> Path:
    """The gold side of the organizers' layout, through the writer the run uses.

    Ticket 17 needs the gold tree in production code — the official scorer reads
    two parallel trees and the release ships the test split with its annotations
    hidden — so this delegates rather than keeping a second implementation that
    could drift from it. Every equivalence test in `tests/test_evaluator.py`
    therefore runs against the same writer the final test run does.
    """
    submission.write_gold_tree(
        [
            Record(
                id=entry["id"],
                type=entry["type"],
                lang=entry["lang"],
                title=entry.get("title", ""),
                abstract=entry.get("abstract", ""),
                subjects=tuple(entry["gold"]),
            )
            for entry in entries
        ],
        destination,
    )
    return destination


def write_prediction_tree(entries: list[dict], destination: Path) -> Path:
    """The submission side, written the way a participant would by hand."""
    for entry in entries:
        directory = destination / entry["type"] / entry["lang"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{entry['id']}.json").write_text(
            json.dumps({"dcterms:subject": list(entry["predictions"])})
        )
    return destination


def official_sheets(module: ModuleType, entries: list[dict], workspace: Path) -> dict:
    """Run the official script end to end and return its three sheets.

    Going through the spreadsheet rather than calling the metric functions
    directly is deliberate: the `Overall` figure this project reports as its
    headline is computed in the spreadsheet-assembly code, not in the metric
    functions, so calling the functions would leave the headline untested.
    """
    import pandas as pd

    true_dict = module.read_gnd_files(str(write_gold_tree(entries, workspace / "gold")), True)
    pred_dict = module.read_gnd_files(
        str(write_prediction_tree(entries, workspace / "pred")), False
    )
    assert module.validate_directory_structure(true_dict, pred_dict)

    results = workspace / "results"
    module.evaluate_and_save_to_excel(
        str(results), "fixture.xlsx", true_dict, pred_dict, list(range(5, 55, 5))
    )
    workbook = results / "fixture.xlsx"
    return {
        "combined": pd.read_excel(workbook, sheet_name="Record Type and Language"),
        "record_type": pd.read_excel(workbook, sheet_name="Record Type"),
        "language": pd.read_excel(workbook, sheet_name="Language"),
    }
