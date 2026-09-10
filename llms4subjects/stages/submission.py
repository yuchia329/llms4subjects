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

`write_gold_tree` is the other side of the same contract, added by ticket 17:
the scorer pairs two parallel trees, and the gold one has to be written from
this project's own splits because the release ships the test set with its
annotations hidden.

Implemented by ticket 03.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ..contracts import CODES_PER_RECORD, Cell, Code, Prediction, Record, cell_of

# The property the shared task is about, and the only key in a submission file.
SUBJECT_FIELD = "dcterms:subject"

# The gold side is JSON-LD, and their reader takes the subjects from the last
# member of `@graph` — so that member's position is load-bearing where the rest
# of the document is decoration. The id URI is the one the release publishes.
GRAPH_FIELD = "@graph"
RECORD_URI = "https://www.tib.eu/en/suchen/id/TIBKAT:{id}"


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

        path = _reserve(root, prediction.record_id, cell, ".json", seen)
        path.write_text(
            json.dumps({SUBJECT_FIELD: list(codes)}, indent="\t", ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        written.append(path)

    return written


def write_gold_tree(
    records: Sequence[Record], destination: str | Path
) -> list[Path]:
    """The other half of the organizers' layout: the gold labels, as JSON-LD.

    Their scorer reads gold and predictions from two parallel trees, so running
    it against this project's own split means writing the gold side too — the
    release ships the test split with its annotations hidden, and the clean CSVs
    are where the answers live. Ticket 17 is the only caller: it is what makes
    "scored by the official scorer" a fact rather than a claim about a local
    evaluator that agrees with it on a fixture.

    The surrounding record is reproduced rather than reduced to the one field
    their reader looks at, so a change to what it reads fails here instead of
    quietly scoring zero. A record with no gold subjects is written anyway and
    their reader drops it, exactly as `evaluator.evaluate` does.
    """
    root = Path(destination)
    written: list[Path] = []
    seen: dict[str, Path] = {}
    cells = {record.id: (record.type, record.lang) for record in records}

    for record in records:
        path = _reserve(
            root, record.id, _directory_cell(cells, record.id), ".jsonld", seen
        )
        path.write_text(
            json.dumps(
                {
                    GRAPH_FIELD: [
                        {"@id": RECORD_URI.format(id=record.id)},
                        {
                            "@type": "bibo:Document",
                            SUBJECT_FIELD: [
                                {"@id": code} for code in record.subjects
                            ],
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        written.append(path)

    return written


def _reserve(
    root: Path, record_id: str, cell: Cell, suffix: str, seen: dict[str, Path]
) -> Path:
    """Where one record's file goes, refusing a record already written.

    Both trees need the same three things — the two directory levels, one file
    per record id, and a refusal rather than an overwrite when a record appears
    twice — and a duplicate that silently overwrote its earlier file would cost
    a record from whichever tree it landed in. Shared so the gold side cannot
    drift from the prediction side.
    """
    directory = root / cell.record_type / cell.language
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record_id}{suffix}"
    if record_id in seen:
        raise ValueError(
            f"duplicate record {record_id!r}: already written to {seen[record_id]}"
        )
    seen[record_id] = path
    return path


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
