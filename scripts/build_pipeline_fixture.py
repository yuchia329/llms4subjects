"""Build the committed corpus fixture the pipeline seam is tested over.

    python scripts/build_pipeline_fixture.py --force

Two disjoint record sets and a vocabulary slice, shaped so that the invariants
docs/spec.md asks of `predict` can be asserted at all:

- `index` records come from tib-core train and carry the gold subjects a
  neighbour retriever harvests;
- `queries` come from dev, span both languages and at least three record types,
  and carry their own gold subjects so a test can perturb them and watch the
  candidate set not move;
- `vocabulary` is a release-shaped slice holding every code the fixture may
  return, plus `unseen_code`, which appears in no fixture record and is the
  property the label tower exists to reach, minus `withheld_codes`, which appear
  in index records and must never be returned because they are outside the
  vocabulary.

Abstracts are truncated: the fixture has to stay small enough to commit and to
run on CPU in seconds, and no assertion here depends on the tail of an abstract.

Dev and train only — the gold test split is opened once, at the end of the
project.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llms4subjects.contracts import Record  # noqa: E402
from llms4subjects.corpus import (  # noqa: E402
    MissingDataset,
    load_split,
    load_vocabulary,
)
from llms4subjects.paths import REPO_ROOT, VOCABULARY_FILES  # noqa: E402

FIXTURE_FILE = REPO_ROOT / "tests" / "fixtures" / "corpus.json"

SEED = 20250131

INDEX_RECORDS = 150
QUERY_RECORDS = 40

# Long enough to look like the real thing to an encoder, short enough to commit.
ABSTRACT_CHARS = 600

# Query records span both languages and four record types, so the by-language
# and by-type slices of a fixture run are not degenerate.
QUERY_CELLS = (
    ("Book", "en", 14),
    ("Book", "de", 12),
    ("Thesis", "de", 8),
    ("Conference", "en", 6),
)

# Index records skewed the same way, so a query has plausible neighbours.
INDEX_CELLS = (
    ("Book", "en", 60),
    ("Book", "de", 50),
    ("Thesis", "de", 25),
    ("Conference", "en", 15),
)

WITHHELD = 3


def build() -> dict:
    rng = random.Random(SEED)
    vocabulary = load_vocabulary("tib-core")
    raw_entries = _raw_entries()

    queries = _choose(load_split("core_dev"), QUERY_CELLS, vocabulary, rng)
    index = _choose(load_split("core_train"), INDEX_CELLS, vocabulary, rng)

    query_codes = {code for record in queries for code in record.subjects}
    index_codes = {code for record in index for code in record.subjects}

    # Codes an index record carries and the vocabulary does not: the restriction
    # to the vocabulary has to hold whatever the index happens to contain.
    withheld = sorted(index_codes - query_codes)[:WITHHELD]
    if len(withheld) < WITHHELD:
        raise SystemExit("index records share every code with the queries")

    # A code no fixture record carries, which only the label tower can reach.
    unseen = sorted(set(vocabulary) - query_codes - index_codes)[0]

    slice_codes = sorted((query_codes | index_codes | {unseen}) - set(withheld))

    return {
        "source": {
            "index_split": "core_train",
            "query_split": "core_dev",
            "vocabulary": "tib-core",
            "seed": SEED,
            "abstract_chars": ABSTRACT_CHARS,
        },
        "index": [_entry(record) for record in index],
        "queries": [_entry(record) for record in queries],
        "vocabulary": [raw_entries[code] for code in slice_codes],
        "unseen_code": unseen,
        "withheld_codes": withheld,
    }


def _raw_entries() -> dict[str, dict]:
    """The release entries verbatim, so the fixture slice reads like the file."""
    raw = json.loads(VOCABULARY_FILES["tib-core"].read_text(encoding="utf-8"))
    entries = raw.values() if isinstance(raw, dict) else raw
    return {entry["Code"]: entry for entry in entries}


def _choose(records, cells, vocabulary, rng) -> list[Record]:
    """A stable sample per cell, of records that have text and gold subjects."""
    by_cell: dict[tuple[str, str], list[Record]] = {}
    for record in records:
        if not record.subjects or not record.text.strip():
            continue
        if not all(code in vocabulary for code in record.subjects):
            continue
        by_cell.setdefault((record.type, record.lang), []).append(record)

    chosen: list[Record] = []
    for record_type, language, wanted in cells:
        available = sorted(by_cell.get((record_type, language), ()), key=lambda r: r.id)
        if len(available) < wanted:
            raise SystemExit(
                f"cell {(record_type, language)} holds {len(available)} usable "
                f"records, need {wanted}"
            )
        chosen.extend(rng.sample(available, wanted))
    return chosen


def _entry(record: Record) -> dict:
    return {
        "id": record.id,
        "type": record.type,
        "lang": record.lang,
        "title": record.title,
        "abstract": record.abstract[:ABSTRACT_CHARS],
        "subjects": list(dict.fromkeys(record.subjects)),
    }


def describe(fixture: dict) -> str:
    lines = [
        f"index records:   {len(fixture['index'])}",
        f"query records:   {len(fixture['queries'])}",
        f"vocabulary:      {len(fixture['vocabulary'])} entries",
        f"unseen code:     {fixture['unseen_code']}",
        f"withheld codes:  {', '.join(fixture['withheld_codes'])}",
    ]
    for name in ("index", "queries"):
        cells: dict[tuple[str, str], int] = {}
        for entry in fixture[name]:
            key = (entry["type"], entry["lang"])
            cells[key] = cells.get(key, 0) + 1
        lines.append(f"{name} cells:")
        lines += [
            f"  {record_type:<12} {language}  {count:>3}"
            for (record_type, language), count in sorted(cells.items())
        ]
    codes = {code for entry in fixture["index"] for code in entry["subjects"]}
    lines.append(f"distinct codes in the index: {len(codes)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite the fixture")
    args = parser.parse_args(argv)

    try:
        fixture = build()
    except MissingDataset as error:
        print(error)
        return 1

    print(describe(fixture))

    if FIXTURE_FILE.exists() and not args.force:
        print(f"\n{FIXTURE_FILE} exists; pass --force to rewrite.")
        return 1

    FIXTURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_FILE.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
    print(f"\nwrote {FIXTURE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
