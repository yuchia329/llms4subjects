"""What goes into an index, and the check that says it may.

Rung 3 grows the index from the tib-core training split to the all-subjects one,
which is a second shared-task track rather than more of the same corpus. Two
things have to hold before that is safe, and neither is obvious from the file:

- **No held-out record is in it.** The tracks are documented as split-aligned,
  and for the gold test split they are: none of its 4,910 records is anywhere in
  `all_train`. Nine `core_dev` records are, which is why this is a check rather
  than a citation. Those nine are named in the attestation and dropped from the
  index by every run, so the alignment claim holds by construction instead of by
  assertion. Checking it at run time instead would mean opening the gold test
  split on every run to prove that it is not being opened, which is the wrong
  trade (docs/spec.md, ticket 17).

- **No document is indexed twice.** `all_train` holds 70,633 rows under 70,588
  ids: 45 documents the release files twice, 20 of them with differing gold.
  Indexed as they come, those 45 vote twice in every neighbour harvest, and the
  "70,588 documents" every table reports would be 70,633.

Both are here rather than in `corpus`, which reads files and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence

from .contracts import Record
from .corpus import data_revision
from .paths import SPLIT_ALIGNMENT_FILE, SPLIT_FILES

ATTESTATION_SCHEMA = 1

# Splits that must not appear in any index: the gold test set, and the dev split
# every number before ticket 17 is measured on.
HELD_OUT = ("core_dev", "core_test")

WRITE_HINT = "Write it with:\n  python scripts/verify_split_alignment.py"


def corpus_revision(corpus: str) -> str:
    """A digest of one index corpus and the held-out splits it is checked against.

    Per corpus rather than per dataset, and this is the whole of why: the
    dataset revision digests every split that exists, so building the
    all-subjects file changes it — and an attestation keyed on it would refuse
    a checkout that has only the tib-core splits and only ever indexes
    `core_train`, over a file that checkout does not have. What makes a check
    stale is a change to the corpus it checked or to the splits it checked that
    corpus against, and that is exactly what this digests.
    """
    return data_revision(
        [SPLIT_FILES[corpus], *(SPLIT_FILES[name] for name in HELD_OUT)]
    )


class ConflictingRecords(ValueError):
    """One record id carrying two different documents, which is a rebuild bug."""


class SplitAlignmentUnverified(RuntimeError):
    """An index corpus nothing has checked against the held-out splits."""


@dataclass(frozen=True)
class Merged:
    """An index corpus with its duplicates resolved, and what that took."""

    records: tuple[Record, ...]
    # Rows read, against `len(records)` documents kept.
    rows: int
    duplicate_ids: int
    # Gold assignments recovered from a second copy of a document, which would
    # have been dropped by keeping the first row and lost by keeping the last.
    merged_subjects: int

    def __len__(self) -> int:
        return len(self.records)


def merge(records: Iterable[Record]) -> Merged:
    """One record per id, in first-seen order, keeping every gold assignment.

    A document the release files twice is one document: its text is indexed
    once, and its subjects are the union of what the copies carry. The union
    rather than either copy because both rows are the release's own gold for
    that document, and a rule that kept one of them would silently drop
    assignments — 25 of them in `all_train` — on the strength of file order.

    Copies that disagree on anything but their subjects are refused. Two
    different documents under one id is not a duplicate, and merging them would
    index one document's text under the other's assignments.
    """
    kept: dict[str, Record] = {}
    duplicated: set[str] = set()
    rows = 0
    recovered = 0

    for record in records:
        rows += 1
        first = kept.get(record.id)
        if first is None:
            kept[record.id] = record
            continue

        duplicated.add(record.id)
        if _document(first) != _document(record):
            raise ConflictingRecords(
                f"record {record.id!r} appears twice with different content; "
                "that is a rebuild bug rather than a duplicate filing"
            )
        added = [code for code in record.subjects if code not in first.subjects]
        recovered += len(added)
        if added:
            kept[record.id] = _with_subjects(first, first.subjects + tuple(added))

    return Merged(
        records=tuple(kept.values()),
        rows=rows,
        duplicate_ids=len(duplicated),
        merged_subjects=recovered,
    )


def _document(record: Record) -> tuple[str, str, str, str]:
    """Everything about a record except the assignments made against it."""
    return (record.type, record.lang, record.title, record.abstract)


def _with_subjects(record: Record, subjects: tuple[str, ...]) -> Record:
    import dataclasses

    return dataclasses.replace(record, subjects=subjects)


def overlap(
    index_records: Sequence[Record], held_out: Mapping[str, set[str]]
) -> dict[str, int]:
    """How many indexed documents are also in each held-out split."""
    indexed = {record.id for record in index_records}
    return {name: len(indexed & ids) for name, ids in sorted(held_out.items())}


def excluded_ids(
    index_records: Sequence[Record], held_out: Mapping[str, set[str]]
) -> dict[str, list[str]]:
    """Which indexed documents each held-out split claims, by id.

    Named rather than counted, because the count is a finding and the ids are
    the fix: a run drops exactly these from its index, and drops them without
    opening the split that named them.
    """
    indexed = {record.id for record in index_records}
    return {
        name: sorted(indexed & ids) for name, ids in sorted(held_out.items())
    }


def attest(
    *,
    revision: str,
    corpora: Mapping[str, Merged],
    held_out: Mapping[str, set[str]],
    revisions: Mapping[str, str] | None = None,
    verified: date | None = None,
) -> dict:
    """The alignment check as the document that gets committed.

    `revision` is the whole dataset as it stood when the check ran, recorded for
    provenance and read by nobody. What a later run compares against is the
    per-corpus digest beside each entry; see `corpus_revision`.
    """
    revisions = revisions or {name: corpus_revision(name) for name in corpora}
    return {
        "schema": ATTESTATION_SCHEMA,
        "data_revision": revision,
        "verified": (verified or date.today()).isoformat(),
        "corpora": {
            name: {
                "revision": revisions[name],
                "rows": merged.rows,
                "records": len(merged.records),
                "duplicate_ids": merged.duplicate_ids,
                "merged_subjects": merged.merged_subjects,
            }
            for name, merged in sorted(corpora.items())
        },
        "held_out": {
            name: {"records": len(ids)} for name, ids in sorted(held_out.items())
        },
        "overlaps": {
            name: overlap(merged.records, held_out)
            for name, merged in sorted(corpora.items())
        },
        # The ids behind those counts. Committing them is what lets a run
        # exclude a held-out record from its index without reading the split
        # that holds it, which for `core_test` is the whole point.
        "excluded": {
            name: excluded_ids(merged.records, held_out)
            for name, merged in sorted(corpora.items())
        },
    }


def load_attestation(path: str | Path | None = None) -> dict:
    """The committed alignment check, verbatim."""
    source = Path(path) if path is not None else SPLIT_ALIGNMENT_FILE
    if not source.exists():
        raise SplitAlignmentUnverified(f"{source} is missing.\n{WRITE_HINT}")
    return json.loads(source.read_text())


def require_alignment(
    corpora: Sequence[str],
    path: str | Path | None = None,
    revisions: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """The ids this index must drop, or a refusal saying nothing checked it.

    Two ways it refuses: the attestation never checked this corpus, or it
    checked a different version of it. Both mean the same thing — the corpus
    about to be indexed is one nothing has compared against the held-out splits
    — and both are fatal, because a contaminated index produces numbers that
    look fine.

    Only the corpora actually being indexed are checked. A checkout that has
    never built the all-subjects split runs rung 1 and rung 2 without ever
    consulting the row that describes it.

    An overlap that *has* been checked is not a refusal. It is nine `core_dev`
    records inside `all_train`, and the answer to it is to index the other
    70,579, which is what the returned ids are for. The caller drops them and
    says how many it dropped; see `scripts/run_experiment.py`.
    """
    written = load_attestation(path)

    excluded: set[str] = set()
    for corpus in corpora:
        entry = written.get("corpora", {}).get(corpus)
        checked = written.get("excluded", {}).get(corpus)
        if entry is None or checked is None:
            raise SplitAlignmentUnverified(
                f"nothing has checked {corpus!r} against the held-out splits.\n"
                f"{WRITE_HINT}"
            )
        current = (
            revisions.get(corpus) if revisions is not None else None
        ) or corpus_revision(corpus)
        if entry.get("revision") != current:
            raise SplitAlignmentUnverified(
                f"the alignment of {corpus!r} was checked against "
                f"{entry.get('revision')!r} and it is now {current!r}, so the "
                f"corpus or a held-out split has changed since.\n{WRITE_HINT}"
            )
        for ids in checked.values():
            excluded.update(ids)
    return frozenset(excluded)


def clear_index(
    records: Sequence[Record], excluded: Collection[str]
) -> list[Record]:
    """The index corpus with every held-out document dropped."""
    return [record for record in records if record.id not in excluded]
