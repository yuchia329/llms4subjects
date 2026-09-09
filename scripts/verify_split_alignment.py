"""Check which corpora are clear to index, and commit the answer.

    python scripts/verify_split_alignment.py             # report only
    python scripts/verify_split_alignment.py --force     # rewrite the reference

Rung 3 indexes the all-subjects training split, which is a second shared-task
track rather than more of the tib-core corpus. The claim that makes that safe is
that the two tracks are split-aligned: a record held out of tib-core is held out
of all-subjects too, so 38,545 extra labelled documents come at no contamination
risk. This is the script that checks it.

It is the one place in the project that opens `core_test`, and it reads exactly
the record ids: no title, no abstract, no gold assignment leaves this script,
and the only thing it writes about the test split is how many records it has and
how many of them are in each index corpus. That is what lets every *run* verify
alignment for free — it reads the committed attestation, and never the test
split (docs/spec.md, ticket 17).

The reference it writes, `reference/split_alignment.json`, is keyed to the
dataset revision, so a rebuild of the data invalidates it and the next run says
so rather than indexing an unchecked corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llms4subjects.corpus import MissingDataset, data_revision, load_split  # noqa: E402
from llms4subjects.paths import SPLIT_ALIGNMENT_FILE  # noqa: E402
from llms4subjects.splits import (  # noqa: E402
    HELD_OUT,
    attest,
    corpus_revision,
    merge,
)

# The corpora a config may name in `index.corpora`, which is every split except
# the held-out ones. Checked together, so the attestation covers a config that
# has not been written yet.
INDEXABLE = ("core_train", "all_train")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="rewrite the committed attestation"
    )
    parser.add_argument("--out", default=str(SPLIT_ALIGNMENT_FILE))
    args = parser.parse_args(argv)

    try:
        corpora = {name: merge(load_split(name)) for name in INDEXABLE}
        # Ids alone. The gold test split is opened here and nowhere else, and
        # what it is asked is whether these ids are somewhere they should not
        # be — not what any of them is about.
        held_out = {
            name: {record.id for record in load_split(name)} for name in HELD_OUT
        }
    except MissingDataset as error:
        print(error)
        return 1

    revision = data_revision()
    document = attest(
        revision=revision,
        corpora=corpora,
        held_out=held_out,
        revisions={name: corpus_revision(name) for name in corpora},
    )

    print(f"data revision: {revision}\n")
    print(render(document))

    contaminated = sum(
        count
        for checked in document["overlaps"].values()
        for count in checked.values()
    )
    if contaminated:
        print(
            f"\n{contaminated} held-out record(s) are in an index corpus, so the "
            "tracks are not aligned as documented. They are named in the "
            "attestation and every run drops them from its index, which is what "
            "makes the alignment hold; see legacy/README.md for what indexing "
            "them instead would cost."
        )

    path = Path(args.out)
    if path.exists() and not args.force:
        existing = json.loads(path.read_text())
        stale = [
            name
            for name, entry in document["corpora"].items()
            if existing.get("corpora", {}).get(name, {}).get("revision")
            != entry["revision"]
        ]
        if not stale:
            print(f"\n{path} already attests every corpus as it stands; unchanged.")
            return 0
        print(
            f"\n{path} attests a different version of {', '.join(stale)}. "
            "Re-run with --force to replace it."
        )
        return 1

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(f"\nwrote {path}")
    return 0


def render(document: dict) -> str:
    """The two tables the attestation is: what is in each corpus, and overlaps."""
    lines = [
        "### Index corpora\n",
        "| corpus | rows | documents | duplicate ids | gold recovered by merging |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, counts in document["corpora"].items():
        lines.append(
            f"| {name} | {counts['rows']:,} | {counts['records']:,} | "
            f"{counts['duplicate_ids']:,} | {counts['merged_subjects']:,} |"
        )

    held_out = list(document["held_out"])
    lines += [
        "\n### Held-out records found in each corpus\n",
        "| corpus | " + " | ".join(
            f"{name} ({document['held_out'][name]['records']:,})"
            for name in held_out
        )
        + " |",
        "|---" + "|---:" * len(held_out) + "|",
    ]
    for name, checked in document["overlaps"].items():
        lines.append(
            f"| {name} | " + " | ".join(str(checked[held]) for held in held_out) + " |"
        )

    dropped = sorted(
        {
            record_id
            for corpus in document["excluded"].values()
            for ids in corpus.values()
            for record_id in ids
        }
    )
    if dropped:
        lines.append(
            f"\nDropped from every index: {len(dropped)} record(s) — "
            + ", ".join(dropped[:5])
            + (", ..." if len(dropped) > 5 else "")
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
