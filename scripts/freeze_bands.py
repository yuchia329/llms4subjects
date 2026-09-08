"""Freeze the label frequency bands against tib-core train counts.

Run once, deliberately:

    python scripts/freeze_bands.py            # report, refuse to overwrite
    python scripts/freeze_bands.py --force    # rewrite reference/frequency_bands.json

The bands exist so that "this label is tail" means the same thing in every
result table the project produces. Recomputing them at evaluation time would
break exactly that: adding the 38,545 all-subjects documents to the index in
rung 3 moves labels across the boundaries, and a tail number that quietly
covers fewer labels than last week's is not a measurement.

So the assignment is a committed artifact, and this script is the only thing
allowed to write it. `--force` in the command line is the record that a change
of reference was intended.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llms4subjects.contracts import Code  # noqa: E402
from llms4subjects.corpus import (  # noqa: E402
    MissingDataset,
    band_for_count,
    load_split,
)
from llms4subjects.paths import FREQUENCY_BANDS_FILE  # noqa: E402

# The reference split. Train counts, never dev or test: the bands describe how
# much training signal a label had, which is the quantity the breakdown is about.
REFERENCE_SPLIT = "core_train"

# Inclusive occurrence ranges, in descending order. `None` is unbounded above.
# This tuple is the definition of the bands; everything downstream reads the
# copy of it written into the reference file.
BOUNDARIES: tuple[dict[str, object], ...] = (
    {"band": "head", "min": 101, "max": None},
    {"band": "torso", "min": 10, "max": 100},
    {"band": "tail", "min": 1, "max": 9},
    # Written out so the reference states all four bands, even though no label
    # is listed under this one: it is every code the others do not name.
    {"band": "zero", "min": 0, "max": 0},
)

SCHEMA = 1


def build_reference() -> dict:
    records = load_split(REFERENCE_SPLIT)

    counts: Counter[Code] = Counter()
    for record in records:
        counts.update(record.subjects)

    labels: dict[str, dict[str, int]] = {
        str(boundary["band"]): {}
        for boundary in BOUNDARIES
        if boundary["band"] != "zero"
    }
    for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        labels[band_for_count(count, BOUNDARIES)][code] = count

    return {
        "schema": SCHEMA,
        "frozen_from": {
            "split": REFERENCE_SPLIT,
            "records": len(records),
            "assignments": sum(counts.values()),
            "distinct_labels": len(counts),
        },
        "boundaries": [dict(boundary) for boundary in BOUNDARIES],
        "labels": labels,
    }


def describe(reference: dict) -> str:
    lines = [
        f"reference split: {reference['frozen_from']['split']}",
        f"records:         {reference['frozen_from']['records']:,}",
        f"assignments:     {reference['frozen_from']['assignments']:,}",
        "",
    ]
    for boundary in reference["boundaries"]:
        band = boundary["band"]
        if band == "zero":
            lines.append(f"{band:<6} 0 occurrences            (every other label)")
            continue
        high = boundary["max"]
        span = f">{boundary['min'] - 1}" if high is None else f"{boundary['min']}-{high}"
        lines.append(f"{band:<6} {span:<24} {len(reference['labels'][band]):>6,} labels")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the committed reference; the bands are frozen without it",
    )
    args = parser.parse_args(argv)

    try:
        reference = build_reference()
    except MissingDataset as error:
        print(error)
        return 1

    print(describe(reference))

    # Not a flag: docs/artifacts.md says this script is the only thing that
    # writes the reference, and an `--output` would make that only a habit.
    output = FREQUENCY_BANDS_FILE
    if output.exists() and not args.force:
        print(f"\n{output} exists; the bands are frozen. Pass --force to rewrite.")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(reference, indent=1, sort_keys=False, ensure_ascii=False) + "\n"
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
