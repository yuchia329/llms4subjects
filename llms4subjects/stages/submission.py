"""Writes the organizers' `<Type>/<lang>/<id>.json` tree, 50 ranked codes each.

This is the one stage that touches the filesystem as its purpose: its output is
the input to the official scorer, so the layout is the contract.

Their reader recovers the record type and language from the last two path
segments and pairs a prediction file with a gold file by basename, which makes
three things load-bearing and easy to get wrong: the directory nesting, the
`.json` extension, and the `dcterms:subject` key inside. Each is asserted here
rather than left to whoever writes the run script.

The writer refuses anything the format cannot represent — a ranking that is not
exactly 50 codes, a repeated code, a record with no cell, a blank cell segment,
the same record twice. All of those are producible by an upstream bug and none
of them fail loudly on their own: a blank language segment collapses two path
components into one, and a duplicate record silently overwrites its earlier
file. Failing here costs a run; failing quietly costs a wrong number.

Implemented by ticket 03.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ..contracts import CODES_PER_RECORD, Cell, Code, Prediction, cell_of

# The property the shared task is about, and the only key in a submission file.
SUBJECT_FIELD = "dcterms:subject"


def write_submission(
    predictions: Sequence[Prediction],
    cells: Mapping[str, tuple[str, str]],
    destination: str | Path,
) -> list[Path]:
    """Write one file per record, returning the paths written."""
    root = Path(destination)
    written: list[Path] = []
    seen: dict[str, Path] = {}

    for prediction in predictions:
        codes = tuple(prediction.codes)
        _check_ranking(prediction.record_id, codes)
        cell = _directory_cell(cells, prediction.record_id)

        directory = root / cell.record_type / cell.language
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{prediction.record_id}.json"

        if prediction.record_id in seen:
            raise ValueError(
                f"duplicate record {prediction.record_id!r}: already written to "
                f"{seen[prediction.record_id]}"
            )
        seen[prediction.record_id] = path

        path.write_text(
            json.dumps({SUBJECT_FIELD: list(codes)}, indent="\t", ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        written.append(path)

    return written


def _check_ranking(record_id: str, codes: tuple[Code, ...]) -> None:
    if len(codes) != CODES_PER_RECORD:
        raise ValueError(
            f"record {record_id!r} has {len(codes)} codes; the submission format "
            f"takes exactly {CODES_PER_RECORD} and has no representation for fewer"
        )
    if len(set(codes)) != len(codes):
        raise ValueError(
            f"record {record_id!r} has duplicate codes; the scorer takes the set "
            "of the top k, so a repeat spends a slot on nothing"
        )


def _directory_cell(
    cells: Mapping[str, tuple[str, str]], record_id: str
) -> Cell:
    """The record's cell, refused if either half cannot be a path segment.

    The evaluator scores a record with a blank language perfectly happily — the
    dev split has four — but there is no directory to write one to, so the
    stricter rule lives here rather than in the shared lookup.
    """
    cell = cell_of(cells, record_id)
    if not cell.record_type or not cell.language:
        raise ValueError(
            f"record {record_id!r} has the blank cell {tuple(cell)!r}; a blank "
            "segment would collapse the layout the scorer reads"
        )
    return cell
