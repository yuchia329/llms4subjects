"""Structural invariants of the stage package.

The pipeline used to be scripts that each re-read the CSVs; these tests keep it
from drifting back. Dataset reads belong to `llms4subjects.corpus`, cache reads
to `llms4subjects.artifacts`, and a stage receives its inputs as arguments.
"""

import ast
import importlib
from pathlib import Path

import pytest

# One module per stage boundary in docs/spec.md ("Module boundaries").
STAGE_MODULES = [
    "label_text",
    "encoders",
    "indexes",
    "retrievers",
    "fusion",
    "group_prior",
    "reranker",
    "adjudicator",
    "evaluator",
    "submission",
]

STAGE_DIR = Path(__file__).resolve().parent.parent / "llms4subjects" / "stages"

# Training is not one of those boundaries, which is why `llms4subjects.finetune`
# sits outside this package: it produces an adapter that `encoders` loads, and
# it never runs inside `predict`.

# What opens or reads a file. `json.loads` is deliberately not here: it parses a
# string, and a stage that parses an API response or a model's answer is not
# reading anything. The file half of `json.loads(path.read_text())` is caught by
# `read_text` and `open` instead.
READ_CALLS = {
    "open", "read_csv", "read_json", "read_parquet", "read_text", "read_bytes",
    "load",
}


def stage_files():
    return sorted(p for p in STAGE_DIR.glob("*.py") if p.name != "__init__.py")


def test_every_stage_boundary_has_exactly_one_module():
    assert [p.stem for p in stage_files()] == sorted(STAGE_MODULES)


@pytest.mark.parametrize("name", STAGE_MODULES)
def test_stage_module_imports(name):
    importlib.import_module(f"llms4subjects.stages.{name}")


@pytest.mark.parametrize("name", STAGE_MODULES)
def test_stage_module_does_not_read_files(name):
    tree = ast.parse((STAGE_DIR / f"{name}.py").read_text())

    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            called.add(func.id)
        elif isinstance(func, ast.Attribute):
            called.add(func.attr)

    forbidden = called & READ_CALLS
    assert not forbidden, f"{name} reads files directly: {sorted(forbidden)}"


@pytest.mark.parametrize("name", STAGE_MODULES)
def test_stage_module_does_not_load_the_dataset_itself(name):
    tree = ast.parse((STAGE_DIR / f"{name}.py").read_text())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{a.name}" for a in node.names)

    assert "llms4subjects.corpus" not in imported, f"{name} loads the dataset itself"


def test_the_predict_seam_exists():
    from llms4subjects.pipeline import predict

    assert callable(predict)
