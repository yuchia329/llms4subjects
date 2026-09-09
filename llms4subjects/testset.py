"""Opening the gold test split, once, under a configuration fixed beforehand.

Every other module in this package is built to keep `core_test` shut:
`IndexConfig` refuses to index it, `scripts/run_experiment.py` refuses to score
it, and `reference/split_alignment.json` names the nine documents an index drops
so that no run has to open the split to prove it is not in it. Ticket 17 is the
one run that opens it, and "once" needs machinery of its own, because a rule
that is only prose is a rule that gets an exception the first time a number
disappoints.

Two documents, both committed, and the order between them is the whole design:

- **`reference/test_plan.json`**, written and committed *before* the split is
  read. It names each row the run will score and digests the configuration
  sections that decide what that row predicts. A run whose configuration has
  moved since is refused, so "the chosen configuration was fixed before the test
  set was read" is checkable in git rather than asserted in a results section.
- **`reference/test_run.json`**, written *after*. A second run is refused unless
  it carries a justification, which is then recorded beside the read it
  supersedes. Nothing here prevents a second run; it makes one leave a mark.

The plan digests `label_text`, `encoder`, `index`, `retrievers`, `fusion`,
`group_prior`, `reranker` and `adjudication` — the sections the artifact store
already calls a prediction's dependencies — and nothing else. `name` and `notes`
are prose about a row, and a digest that moved when a comment did would fire on
a typo fix and be switched off by the next person who hit it.

This module also holds the two facts the report has to state about the split
itself: which of its records duplicate an indexed training document, and which
of its cells the organizers' own scorer cannot read.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import STAGE_DEPENDENCIES
from .config import ExperimentConfig
from .contracts import Cell, Record
from .paths import TEST_PLAN_FILE, TEST_RUN_FILE

PLAN_SCHEMA = "test-plan/1"
RECEIPT_SCHEMA = "test-receipt/1"

# The sections that decide what a row predicts, which the artifact store already
# names as a prediction's dependencies. Read from there rather than repeated, so
# a section added to the pipeline is covered by the plan on the same commit.
CONFIGURED = STAGE_DEPENDENCIES["predictions"]

DIGEST_LENGTH = 12

PLAN_HINT = (
    "Fix it, and commit it, before the split is read:\n"
    "  python scripts/final_test.py --fix-plan"
)

# What the organizers' reader can index into. It seeds its result with these
# five record types and these two languages and then subscripts that dictionary
# by directory name, so anything else raises a `KeyError` inside their code;
# `tests/test_submission.py` holds the failure itself. The test split contains
# French, Spanish, Czech, Turkish, Dutch and Japanese records, so this is a
# partition the run has to make rather than a case that cannot arise.
OFFICIAL_RECORD_TYPES = ("Article", "Book", "Conference", "Report", "Thesis")
OFFICIAL_LANGUAGES = ("de", "en")


class RunUnplanned(RuntimeError):
    """A row nothing fixed before the gold test split was opened."""


class SplitAlreadyRead(RuntimeError):
    """The gold test split has been read, and this run has no justification."""


def configuration_digest(config: ExperimentConfig) -> str:
    """A digest of everything about `config` that decides what it predicts."""
    payload = json.dumps(
        {name: config.section(name) for name in CONFIGURED},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:DIGEST_LENGTH]


def plan(
    *,
    headline: str,
    rows: Mapping[str, ExperimentConfig],
    fixed: date | None = None,
    paths: Mapping[str, str] | None = None,
    rationale: str = "",
) -> dict:
    """The configurations this run will score, as the document that gets committed.

    `headline` names the row the leaderboard comparison is read from. It has to
    be one of `rows`, because a headline chosen after the fact is the thing this
    file exists to prevent.
    """
    if headline not in rows:
        raise ValueError(
            f"the headline row {headline!r} is not one of the planned rows "
            f"({', '.join(sorted(rows))})"
        )
    paths = paths or {}
    return {
        "schema": PLAN_SCHEMA,
        "fixed": (fixed or date.today()).isoformat(),
        "headline": headline,
        "rationale": rationale,
        "rows": {
            name: {
                "config": paths.get(name, ""),
                "name": config.name,
                "digest": configuration_digest(config),
            }
            for name, config in sorted(rows.items())
        },
    }


def load_plan(path: str | Path | None = None) -> dict:
    """The committed plan, verbatim, or a refusal saying nothing fixed one."""
    source = Path(path) if path is not None else TEST_PLAN_FILE
    if not source.exists():
        raise RunUnplanned(
            f"{source} is missing, so no configuration was fixed before the "
            f"gold test split was read.\n{PLAN_HINT}"
        )
    return json.loads(source.read_text())


def require_plan(
    rows: Mapping[str, ExperimentConfig], path: str | Path | None = None
) -> dict:
    """The plan, if it fixed exactly these configurations. Two ways it refuses.

    A row the plan does not name is a configuration chosen after the split was
    opened, whatever order the files were actually written in. A row whose
    digest has moved is the same thing wearing the planned row's name.

    A planned row that is *not* being run is not a refusal: the adjudication
    rows need an API credential this environment does not have, and a blocked
    row should not make the rest of the run unscoreable.
    """
    document = load_plan(path)
    planned = document.get("rows", {})

    unplanned = sorted(set(rows) - set(planned))
    if unplanned:
        raise RunUnplanned(
            f"the plan does not name the row(s) {', '.join(unplanned)}, so "
            f"nothing fixed them before the split was read.\n{PLAN_HINT}"
        )

    for name in sorted(rows):
        current = configuration_digest(rows[name])
        if planned[name].get("digest") != current:
            raise RunUnplanned(
                f"row {name!r} was planned as {planned[name].get('digest')!r} "
                f"and is now {current!r}, so its configuration changed since "
                f"the plan was fixed.\n{PLAN_HINT}"
            )
    return document


def require_unread(
    path: str | Path | None = None, justification: str = ""
) -> tuple[dict, ...]:
    """The reads this run supersedes, or a refusal naming the one it would.

    Returns what the next receipt should carry as its `earlier` block: every
    previous read, each with the justification given for superseding it. An
    unread split returns nothing and needs no justification, which is the
    ordinary case exactly once.
    """
    source = Path(path) if path is not None else TEST_RUN_FILE
    if not source.exists():
        return ()

    written = json.loads(source.read_text())
    read = written.get("read", "an unrecorded date")
    if not justification:
        raise SplitAlreadyRead(
            f"the gold test split was read on {read} and {source} records it. "
            "Reading it again needs a recorded reason:\n"
            '  python scripts/final_test.py --justify "why this run is needed"'
        )
    return tuple(written.get("earlier", ())) + (
        {"read": read, "justification": justification},
    )


def receipt(
    *,
    read: date | None = None,
    plan: Mapping,
    rows: Mapping[str, Mapping],
    records: int = 0,
    earlier: Sequence[Mapping] = (),
) -> dict:
    """What the run scored, as the document that says the split has been opened."""
    return {
        "schema": RECEIPT_SCHEMA,
        "read": (read or date.today()).isoformat(),
        "records": records,
        "plan": dict(plan),
        "rows": dict(rows),
        "earlier": tuple(earlier),
    }


def duplicated_from(
    records: Sequence[Record], index_records: Sequence[Record]
) -> tuple[str, ...]:
    """Test records whose title and abstract exactly duplicate an indexed one.

    docs/spec.md commits to reporting the score with and without these, because
    a neighbour retriever is unusually strong on precisely them: the document it
    is closest to carries the answer under a different TIBKAT id. The match is
    exact rather than normalised, and on both fields together, because the
    caveat is about identical records — a looser rule would be a different and
    larger claim reported under this one's number.

    A record matching itself is not a duplicate: it is the same document, and
    the index drops it anyway (`splits.clear_index`).
    """
    indexed: dict[tuple[str, str], set[str]] = {}
    for indexed_record in index_records:
        indexed.setdefault(_document(indexed_record), set()).add(indexed_record.id)

    return tuple(
        sorted(
            record.id
            for record in records
            if indexed.get(_document(record), set()) - {record.id}
        )
    )


def unreadable_cells(
    cells: Mapping[str, tuple[str, str]]
) -> dict[Cell, tuple[str, ...]]:
    """The records the organizers' scorer would raise on, grouped by cell.

    Named rather than counted, and returned rather than dropped: the run scores
    them with the local evaluator and reports them as a cell the published table
    has no row for, which is one of the reasons the micro figure travels beside
    the official one.
    """
    grouped: dict[Cell, list[str]] = {}
    for record_id, (record_type, language) in cells.items():
        if record_type in OFFICIAL_RECORD_TYPES and language in OFFICIAL_LANGUAGES:
            continue
        grouped.setdefault(Cell(record_type, language), []).append(record_id)
    return {cell: tuple(sorted(ids)) for cell, ids in sorted(grouped.items())}


def _document(record: Record) -> tuple[str, str]:
    """The two fields the caveat is about, as the release wrote them."""
    return (record.title, record.abstract)
