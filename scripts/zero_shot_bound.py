"""What no corpus of documents can reach: the figure the writeup leads with.

    python scripts/zero_shot_bound.py configs/test.yaml --split core_test \\
        --json reference/zero_shot_bound.json

Every system on this leaderboard proposes subject headings by document
similarity — the winning one harvests the subjects of a record's nearest
neighbours, and so does this project's strongest retriever. A heading no
document carries cannot be harvested from a neighbour at any k, from any index,
however good the encoder is. That is not a weakness of a model; it is a
property of the benchmark, and this script measures it.

The measurement is arithmetic over three committed things, which is why it is a
script of its own rather than a paragraph in a results section:

- **`reference/frequency_bands.json`**, the frozen bands, which say which
  labels no tib-core training record carries. Read, never recomputed — a bound
  derived from bands the corpus redrew would be a bound against itself.
- **the corpus a config indexes**, assembled by `run_experiment.load_inputs` so
  that it is the same 70,579 documents a run indexes and not 70,633 rows.
- **a split's gold**, which weights each label by the assignments it actually
  carries. The vocabulary-wide count would be dominated by the 64,820 codes
  nothing in this benchmark ever assigns.

`scripts/rung3_report.py` computes the same quantity on dev as one section of a
much larger report, and this script imports its arithmetic rather than
repeating it. What is here and not there is the gold test split, which is the
number the writeup quotes and which needs ticket 17's receipt: deriving
anything from `core_test`'s gold before the split has been opened would *be*
the opening, arriving through a script no plan was digested for.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms4subjects.config import load_experiment  # noqa: E402
from llms4subjects.contracts import CODES_PER_RECORD  # noqa: E402
from llms4subjects.corpus import (  # noqa: E402
    UNSEEN_BAND,
    band_for_count,
    frequency_band_reference,
    label_counts,
)
from llms4subjects.stages.evaluator import BANDS  # noqa: E402
from llms4subjects.testset import SplitUnread, require_read  # noqa: E402
from rung3_report import corpus_counts, migration, unreachable  # noqa: E402
from run_experiment import FORBIDDEN_SPLIT, INPUT_ERRORS, load_inputs  # noqa: E402

SCHEMA = "zero-shot-bound/1"


@dataclass(frozen=True)
class Bound:
    """The part of a split's gold that document similarity cannot propose.

    `labels` is counted over the split's gold rather than over the vocabulary,
    and every count here is of labels the *frozen* bands call zero-shot, so
    growing the corpus can move a label out of `unreachable` and can never move
    one in.
    """

    labels: int
    reached: int
    unreachable: int
    assignments: int
    assignments_reached: int
    assignments_unreachable: int
    total_assignments: int
    landed: Mapping[str, int]
    # The two label sets behind the counts, named rather than only tallied, so
    # that a run can be scored on exactly the headings the bound is about.
    unreachable_labels: tuple[str, ...] = ()
    reached_labels: tuple[str, ...] = ()

    @property
    def share(self) -> float:
        """The unreachable assignments as a fraction of all of the split's gold.

        The denominator is every gold assignment, not the zero-shot ones: the
        claim the writeup makes is about how much of the benchmark is out of
        reach, and a share of the unreachable band would be 1.0 by definition.
        """
        if not self.total_assignments:
            return 0.0
        return self.assignments_unreachable / self.total_assignments


def bound(
    *,
    gold_records: Sequence,
    index_records: Sequence,
    frozen: Mapping[str, int] | None = None,
    boundaries: Sequence[Mapping[str, object]] | None = None,
) -> Bound:
    """How much of `gold_records` no document in `index_records` carries."""
    if boundaries is None:
        boundaries = frequency_band_reference()["boundaries"]
    if frozen is None:
        frozen = label_counts()

    weights = corpus_counts(gold_records)
    counts = corpus_counts(index_records)
    labels = sorted(weights)

    by_label = migration(labels, frozen, counts, boundaries)
    by_assignment = migration(labels, frozen, counts, boundaries, weights=weights)

    # The same partition the tables count, kept as the codes themselves. One
    # pass over the split's gold labels, and the band of each is read exactly
    # as `migration` reads it.
    zero = [
        code
        for code in labels
        if band_for_count(frozen.get(code, 0), boundaries) == UNSEEN_BAND
    ]
    still = tuple(
        code
        for code in zero
        if band_for_count(counts.get(code, 0), boundaries) == UNSEEN_BAND
    )
    unreached = set(still)

    zero_labels = sum(by_label.get(UNSEEN_BAND, {}).values())
    zero_assignments = sum(by_assignment.get(UNSEEN_BAND, {}).values())
    still_labels = unreachable(by_label)
    still_assignments = unreachable(by_assignment)

    return Bound(
        labels=zero_labels,
        reached=zero_labels - still_labels,
        unreachable=still_labels,
        assignments=zero_assignments,
        assignments_reached=zero_assignments - still_assignments,
        assignments_unreachable=still_assignments,
        total_assignments=sum(weights.values()),
        landed={
            band: count
            for band, count in by_label.get(UNSEEN_BAND, {}).items()
            if band != UNSEEN_BAND
        },
        unreachable_labels=still,
        reached_labels=tuple(code for code in zero if code not in unreached),
    )


# The k the writeup quotes the bound at, and the two the band tables carry
# beside it. The submission is 50 codes, so nothing past it is measurable here.
REACHED_KS = (5, 10, CODES_PER_RECORD)


def read_submission(root: str | Path) -> dict[str, list[str]]:
    """A submission tree as one ranked list per record id.

    Read back rather than re-predicted: the ranking a run produced is a fact
    about that run, and re-deriving it for a table would mean re-opening the
    gold split under a configuration nothing planned.
    """
    return {
        path.stem: list(json.loads(path.read_text()).get("dcterms:subject", ()))
        for path in sorted(Path(root).rglob("*.json"))
    }


def recall_over(
    *,
    labels: Sequence[str],
    gold_records: Sequence,
    predictions: Mapping[str, Sequence[str]],
    k: int,
) -> tuple[int, int]:
    """Hits and gold assignments, counting only the assignments `labels` carry.

    A record the submission has no file for keeps its assignments in the
    denominator and scores nothing on them: the question is what share of the
    benchmark was reached, and a missing file is a heading not proposed.
    """
    wanted = set(labels)
    hits = assignments = 0
    for record in gold_records:
        top = set(predictions.get(record.id, ())[:k])
        for code in dict.fromkeys(record.subjects):
            if code in wanted:
                assignments += 1
                hits += code in top
    return hits, assignments


def _reached(
    measured: Bound, gold_records: Sequence, predictions: Mapping[str, Sequence[str]]
) -> dict[str, dict[int, tuple[int, int]]]:
    """What a run reached on each half of the zero-shot band, at each k."""
    return {
        name: {
            k: recall_over(
                labels=labels, gold_records=gold_records, predictions=predictions, k=k
            )
            for k in REACHED_KS
        }
        for name, labels in (
            ("unreachable", measured.unreachable_labels),
            ("reached", measured.reached_labels),
        )
    }


def require_readable(split: str, receipt: str | Path | None = None) -> dict:
    """`core_test` is derivable here only because ticket 17 already opened it."""
    if split != FORBIDDEN_SPLIT:
        return {}
    return require_read(receipt)


def render(
    measured: Bound,
    *,
    split: str,
    corpora: Sequence[str],
    documents: int,
    gold_records: Sequence = (),
    predictions: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """The block that goes into the results document, numbers and derivation.

    Given a run's submission, the block also carries what that run reached on
    each half of the zero-shot band — the bound as a measurement rather than
    only as a limit.
    """
    lines = [
        f"Of the {measured.total_assignments:,} gold assignments in `{split}`, "
        f"**{measured.assignments_unreachable:,} ({measured.share:.1%})** are "
        f"carried by labels that appear on none of the "
        f"{documents:,} documents in `{'`, `'.join(corpora)}` — the largest "
        f"corpus this benchmark makes available.\n",
        "| | labels | gold assignments |",
        "|---|---:|---:|",
        f"| zero-shot against tib-core train (frozen) | {measured.labels:,} | "
        f"{measured.assignments:,} |",
        f"| of those, carried by some indexed document | {measured.reached:,} | "
        f"{measured.assignments_reached:,} |",
        f"| **unreachable at any corpus size** | **{measured.unreachable:,}** | "
        f"**{measured.assignments_unreachable:,}** |",
    ]

    if measured.landed:
        lines += [
            "",
            "| where the reached labels land | labels |",
            "|---|---:|",
        ]
        for band in BANDS:
            if band in measured.landed:
                lines.append(f"| {band} | {measured.landed[band]:,} |")

    if predictions is not None:
        reached = _reached(measured, gold_records, predictions)
        lines += [
            "",
            "What the run reached on each half, micro recall over the "
            "assignments those labels carry:",
            "",
            "| labels | assignments | " + " | ".join(f"R@{k}" for k in REACHED_KS) + " |",
            "|---|---:|" + "---:|" * len(REACHED_KS),
        ]
        for name, caption in (
            ("unreachable", "carried by no indexed document"),
            ("reached", "carried by at least one"),
        ):
            row = reached[name]
            assignments = row[REACHED_KS[0]][1]
            scores = " | ".join(
                f"{hits / assignments:.3f}" if assignments else "—"
                for hits, _ in (row[k] for k in REACHED_KS)
            )
            lines.append(f"| {caption} | {assignments:,} | {scores} |")

    return "\n".join(lines)


def document(
    measured: Bound,
    *,
    split: str,
    config: str,
    corpora: Sequence[str],
    documents: int,
    revision: str,
    read: str,
    gold_records: Sequence = (),
    predictions: Mapping[str, Sequence[str]] | None = None,
    submission: str = "",
) -> dict:
    """The bound as plain data, so the writeup's headline cites a file.

    `read` is the date the receipt records the gold split was opened on, empty
    for any other split. It travels with the figure because a number derived
    from `core_test` is only as legitimate as the read it was derived after.

    A `reached` block appears only when a run's submission was given: the bound
    is a property of the benchmark and holds with no run at all, and a key
    reading 0.0 would say the opposite of "nothing was scored here".
    """
    written = {
        "schema": SCHEMA,
        "split": split,
        "config": config,
        "data_revision": revision,
        "read": read,
        "index": {"documents": documents, "corpora": list(corpora)},
        "bound": {
            "labels": measured.labels,
            "reached": measured.reached,
            "unreachable": measured.unreachable,
            "assignments": measured.assignments,
            "assignments_reached": measured.assignments_reached,
            "assignments_unreachable": measured.assignments_unreachable,
            "total_assignments": measured.total_assignments,
            "share_of_gold_assignments": measured.share,
            "landed": dict(measured.landed),
        },
    }

    if predictions is not None:
        reached = _reached(measured, gold_records, predictions)
        written["reached"] = {
            "submission": submission,
            **{
                name: {
                    str(k): (hits / assignments if assignments else 0.0)
                    for k, (hits, assignments) in row.items()
                }
                for name, row in reached.items()
            },
        }
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="the config whose index corpus bounds the split")
    parser.add_argument("--split", default="core_dev", help="the split to bound")
    parser.add_argument(
        "--predictions",
        help="a submission tree, to score a run on the labels the bound names",
    )
    parser.add_argument("--json", dest="json_path", help="write the bound here")
    args = parser.parse_args(argv)

    config = load_experiment(args.config)

    try:
        receipt = require_readable(args.split)
    except SplitUnread as error:
        print(error)
        return 1

    try:
        inputs = load_inputs(config, args.split)
    except INPUT_ERRORS as error:
        print(error.args[0] if isinstance(error, KeyError) else error)
        return 1

    predictions = None
    if args.predictions:
        predictions = read_submission(args.predictions)
        if not predictions:
            print(f"{args.predictions} holds no submission files")
            return 1

    measured = bound(
        gold_records=inputs.records, index_records=inputs.index_records
    )
    print(f"the bound on document similarity over {len(inputs.records)} "
          f"{args.split} records")
    print(f"data revision: {inputs.revision}")
    print(
        f"index: {len(inputs.index_records)} documents from "
        f"{', '.join(config.index.corpora)}"
    )
    if receipt:
        print(f"gold split read on {receipt.get('read', 'an unrecorded date')}")
    if predictions is not None:
        print(f"predictions: {len(predictions)} records from {args.predictions}")
    print()
    print(
        render(
            measured,
            split=args.split,
            corpora=config.index.corpora,
            documents=len(inputs.index_records),
            gold_records=inputs.records,
            predictions=predictions,
        )
    )

    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                document(
                    measured,
                    split=args.split,
                    config=args.config,
                    corpora=config.index.corpora,
                    documents=len(inputs.index_records),
                    revision=inputs.revision,
                    read=receipt.get("read", ""),
                    gold_records=inputs.records,
                    predictions=predictions,
                    submission=args.predictions or "",
                ),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
