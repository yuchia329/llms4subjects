"""Build the committed evaluation fixture that pins the evaluator to the official scorer.

    python scripts/build_eval_fixture.py --force

The fixture is a handful of dev records with fabricated 50-code rankings, chosen
so the equivalence test exercises the cases where two plausible implementations
disagree rather than a comfortable average:

- a record whose gold set is larger than k, where perfect ranking still caps recall
- a record with no hits at all
- a record whose whole gold set sits inside the top k, where recall reaches 1.0
- a scoring cell holding exactly one record, which is where micro and the
  official macro-over-cells aggregation come apart

Predictions are fabricated rather than produced by a model on purpose: this test
must keep working before any retriever exists, and must not change when one does.
Dev only — the gold test split is opened once, at the end of the project.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llms4subjects.contracts import CODES_PER_RECORD  # noqa: E402
from llms4subjects.corpus import (  # noqa: E402
    MissingDataset,
    load_split,
    load_vocabulary,
)
from llms4subjects.paths import REPO_ROOT  # noqa: E402

FIXTURE_FILE = REPO_ROOT / "tests" / "fixtures" / "evaluation.json"

SEED = 20250131

# The official scorer's reader seeds its result dictionary with `de` and `en`
# only, so a submission tree containing any other language raises a KeyError
# inside their script. The fixture stays inside what their reader can read; the
# limitation itself is recorded in test_submission.py.
LANGUAGES = ("de", "en")

# (record type, language, records). Six cells, four record types, both
# languages, and two cells holding a single record.
CELLS = (
    ("Book", "en", 16),
    ("Book", "de", 12),
    ("Thesis", "de", 10),
    ("Conference", "en", 8),
    ("Report", "de", 1),
    ("Article", "en", 1),
)


# Dev, never test: nothing before ticket 17 may read the gold test split.
SPLIT = "core_dev"


def build() -> dict:
    rng = random.Random(SEED)
    vocabulary = sorted(load_vocabulary("tib-core"))
    records = load_split(SPLIT)

    by_cell: dict[tuple[str, str], list] = {}
    for record in records:
        if record.lang not in LANGUAGES or not record.subjects:
            continue
        by_cell.setdefault((record.type, record.lang), []).append(record)

    chosen = []
    for record_type, language, wanted in CELLS:
        available = by_cell[(record_type, language)]
        # Largest gold sets first, so at least one record's gold exceeds k=5,
        # then id order, so the choice does not move when the CSV row order does.
        available = sorted(available, key=lambda r: (-len(r.subjects), r.id))
        chosen.extend(available[:wanted])

    # The three special cases are assigned by gold-set size rather than by
    # position, so "perfect" really means every gold label inside k rather than
    # a perfect ranking that k still truncates.
    by_gold = sorted(chosen, key=lambda r: (-len(r.subjects), r.id))
    profiles = {
        by_gold[0].id: "gold-larger-than-k",
        by_gold[-1].id: "perfect-within-k",
        by_gold[-2].id: "no-hits",
    }

    entries = []
    for position, record in enumerate(chosen):
        gold = list(dict.fromkeys(record.subjects))
        profile = profiles.get(record.id, "mixed")
        entries.append(
            {
                "id": record.id,
                "type": record.type,
                "lang": record.lang,
                "gold": gold,
                "predictions": _ranking(gold, vocabulary, profile, position, rng),
                "profile": profile,
            }
        )

    return {
        "source": {"split": SPLIT, "seed": SEED, "codes_per_record": CODES_PER_RECORD},
        "records": entries,
    }


def _ranking(
    gold: list[str],
    vocabulary: list[str],
    profile: str,
    position: int,
    rng: random.Random,
) -> list[str]:
    """A 50-code ranking for one record, shaped by the case it has to cover."""
    filler = _filler(gold, vocabulary, rng)

    if profile == "no-hits":
        return filler[:CODES_PER_RECORD]

    if profile in ("gold-larger-than-k", "perfect-within-k"):
        ranking = gold + filler
        return ranking[:CODES_PER_RECORD]

    # Mixed: place a shrinking share of the gold labels at scattered ranks, so
    # recall keeps climbing with k instead of saturating at k=5.
    keep = gold[: 1 + position % len(gold)]
    ranking = filler[:CODES_PER_RECORD]
    slots = rng.sample(range(CODES_PER_RECORD), len(keep))
    for slot, code in zip(sorted(slots), keep):
        ranking[slot] = code
    return ranking


def _filler(gold: list[str], vocabulary: list[str], rng: random.Random) -> list[str]:
    """Codes that are certainly wrong for this record, in a stable random order."""
    pool = rng.sample(vocabulary, CODES_PER_RECORD + len(gold) + 10)
    return [code for code in pool if code not in gold]


def describe(fixture: dict) -> str:
    cells: dict[tuple[str, str], int] = {}
    for entry in fixture["records"]:
        cells[(entry["type"], entry["lang"])] = (
            cells.get((entry["type"], entry["lang"]), 0) + 1
        )
    gold = [len(entry["gold"]) for entry in fixture["records"]]
    lines = [
        f"records: {len(fixture['records'])}",
        f"gold labels per record: min {min(gold)}, max {max(gold)}",
        "cells:",
    ]
    lines += [
        f"  {record_type:<12} {language}  {count:>3}"
        for (record_type, language), count in sorted(cells.items())
    ]
    lines.append("profiles:")
    for entry in fixture["records"]:
        if entry["profile"] != "mixed":
            hits = len(set(entry["gold"]) & set(entry["predictions"]))
            lines.append(
                f"  {entry['profile']:<20} {entry['id']}  "
                f"gold {len(entry['gold'])}, hits {hits}"
            )
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

    output = FIXTURE_FILE
    if output.exists() and not args.force:
        print(f"\n{output} exists; pass --force to rewrite.")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
