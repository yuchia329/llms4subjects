"""The final test run: the gold split opened once, scored by the official scorer.

    python scripts/final_test.py --fix-plan          # before the split is read
    python scripts/final_test.py --rehearse          # the same harness, on dev
    python scripts/final_test.py                     # the run itself, once

Every configuration decision was made on dev (docs/results.md). This script runs
the chosen ones against `core_test` and reports what they score, and almost all
of its design is about the two ways that number could be dishonest.

**Chosen before, not after.** `reference/test_plan.json` digests each row's
configuration and is committed before the split is read; a row whose
configuration has moved since is refused rather than scored. `reference/
test_run.json` is written after, so a second read is a refusal that names the
first. Neither prevents anything — they make it leave a mark. See
`llms4subjects/testset.py`.

**Reported at both aggregations, and against the published table.** The official
`Overall` figure comes from the organizers' own script over a submission tree,
not from the local evaluator, which is checked against it in the same run on the
same cells. The micro figure stands beside it because they diverge materially:
eleven cells holding 0.4% of test records carry 55% of the official metric, and
the split contains French, Spanish, Czech, Turkish, Dutch and Japanese records
that each form a cell their reader cannot even index.

Two caveats are rows rather than footnotes: the score without the test records
that exactly duplicate an indexed training document, and the adjudication rows,
which are never merged into the headline.

Implemented by ticket 17.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms4subjects.artifacts import ArtifactStore  # noqa: E402
from llms4subjects.config import ExperimentConfig, load_experiment  # noqa: E402
from llms4subjects.contracts import CODES_PER_RECORD, Cell, Record  # noqa: E402
from llms4subjects.corpus import frequency_bands, load_split  # noqa: E402
from llms4subjects.hardware import describe_device, select_device  # noqa: E402
from llms4subjects.paths import TEST_PLAN_FILE, TEST_RUN_FILE  # noqa: E402
from llms4subjects.pipeline import predict  # noqa: E402
from llms4subjects.stages.adjudicator import MissingApiKey  # noqa: E402
from llms4subjects.stages.evaluator import (  # noqa: E402
    BANDS,
    OFFICIAL_KS,
    EvaluationReport,
    evaluate,
    render,
)
from llms4subjects.stages.submission import (  # noqa: E402
    write_gold_tree,
    write_submission,
)
from llms4subjects.testset import (  # noqa: E402
    RunUnplanned,
    SplitAlreadyRead,
    configuration_digest,
    duplicated_from,
    plan,
    receipt,
    require_plan,
    require_unread,
    unreadable_cells,
)
from run_experiment import INPUT_ERRORS, load_inputs  # noqa: E402

SCHEMA = "test-run/1"

TEST_SPLIT = "core_test"
# The rehearsal runs the whole harness against the split every decision was made
# on, so that a bug in this script is found before the one run that matters.
REHEARSAL_SPLIT = "core_dev"

# The rows, and the config each is read from. The headline is the one the
# leaderboard comparison quotes; the adjudication rows are separate by
# construction, because docs/spec.md allows the appendix row a model the cutoff
# does not cover and ticket 17 requires it never be merged into the headline.
HEADLINE = "headline"
ROWS: dict[str, str] = {
    HEADLINE: "configs/test.yaml",
    "adjudicated": "configs/test-adjudicate.yaml",
}

RATIONALE = (
    "Rung 3's encoder, fine-tuned, at the all-subjects index (docs/results.md: "
    "0.6044 dev micro R@10, the best measured), with the screened cross-encoder "
    "in `fuse` mode, which ticket 12 recommended for exactly this run: +0.0337 "
    "dev micro R@10 for one pass paid once. The group prior stays off (+0.005 "
    "for a stage with its own training step and a measurable cost to the "
    "zero-shot band), and adjudication is a separate labelled row rather than "
    "part of the headline, because no dev measurement of it exists."
)

# The published tib-core leaderboard (docs/spec.md, "Reference points"). The
# table this run's headline has to sit beside, quoted rather than recomputed.
LEADERBOARD = (
    ("RUC Team", 0.25, 0.48, 0.16, 0.57, 0.66),
    ("Annif", 0.23, 0.48, 0.14, 0.54, 0.59),
    ("LA2I2F", 0.20, 0.41, 0.13, 0.49, 0.58),
    ("DUTIR831", 0.23, 0.49, 0.13, 0.54, 0.56),
)


class OfficialScorerUnavailable(RuntimeError):
    """The organizers' script is not runnable in this checkout."""


@dataclass(frozen=True)
class Official:
    """The organizers' script's own numbers, over the cells its reader accepts."""

    at_k: Mapping[int, Mapping[str, float]]
    cells: int
    records: int

    def recall(self, k: int) -> float:
        return self.at_k[k]["recall"]

    def precision(self, k: int) -> float:
        return self.at_k[k]["precision"]

    @property
    def mean_recall(self) -> float:
        """The leaderboard's `Avg R@k` column: recall averaged over every k."""
        return sum(m["recall"] for m in self.at_k.values()) / len(self.at_k)


@dataclass(frozen=True)
class Scored:
    """One row of the run: what it predicted, and every way it was scored."""

    row: str
    config_path: str
    config: ExperimentConfig
    report: EvaluationReport
    without_duplicates: EvaluationReport
    official: Official | None
    submission: Path
    elapsed: float


@dataclass(frozen=True)
class Blocked:
    """A planned row that could not run, and the sentence saying why."""

    row: str
    config_path: str
    reason: str


@dataclass(frozen=True)
class Facts:
    """What is true of the split itself, whichever configuration scores it."""

    records: int
    # Test records duplicating a document *this run indexed*, which is the set
    # the caveat is operationally about: those are the ones whose nearest
    # neighbour carries the answer.
    duplicates: tuple[str, ...]
    # The same against `core_train` alone, which is the training split
    # docs/spec.md's "163 records" was stated over. Reported beside the first so
    # the two can be reconciled rather than confused.
    train_duplicates: tuple[str, ...]
    unreadable: Mapping[Cell, tuple[str, ...]]

    @property
    def summary(self) -> str:
        unreadable = sum(len(ids) for ids in self.unreadable.values())
        return (
            f"{self.records} records; {len(self.duplicates)} duplicate an "
            f"indexed document ({len(self.train_duplicates)} duplicate a "
            f"core_train one); {unreadable} in {len(self.unreadable)} cell(s) "
            "the official scorer cannot read"
        )


def split_facts(inputs, config: ExperimentConfig) -> Facts:
    """The duplicate sets and the unscoreable cells, computed once for the run."""
    cells = {record.id: (record.type, record.lang) for record in inputs.records}
    if "core_train" in config.index.corpora:
        train = inputs.index_records
    else:
        train = load_split("core_train")
    return Facts(
        records=len(inputs.records),
        duplicates=duplicated_from(inputs.records, inputs.index_records),
        train_duplicates=duplicated_from(inputs.records, train),
        unreadable=unreadable_cells(cells),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix-plan",
        action="store_true",
        help="write reference/test_plan.json and stop; commit it before running",
    )
    parser.add_argument(
        "--rehearse",
        action="store_true",
        help=f"run the whole harness against {REHEARSAL_SPLIT} and write no receipt",
    )
    parser.add_argument(
        "--justify",
        default="",
        help="the recorded reason for reading the gold split again",
    )
    parser.add_argument("--row", action="append", default=[], help="score only these rows")
    parser.add_argument("--out", default="artifacts/test", help="where trees are written")
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    parser.add_argument("--limit", type=int, help="rehearsal only: first N records")
    parser.add_argument("--json", help="also write the run document here")
    args = parser.parse_args(argv)

    wanted = args.row or list(ROWS)
    unknown = [row for row in wanted if row not in ROWS]
    if unknown:
        print(f"unknown row(s): {', '.join(unknown)} (known: {', '.join(ROWS)})")
        return 1

    configs = {row: load_experiment(ROWS[row]) for row in wanted}

    if args.fix_plan:
        return fix_plan(configs)

    split = REHEARSAL_SPLIT if args.rehearse else TEST_SPLIT

    try:
        planned = require_plan(configs)
        earlier = (
            () if args.rehearse else require_unread(justification=args.justify)
        )
    except (RunUnplanned, SplitAlreadyRead) as error:
        print(error)
        return 1

    print(f"plan:          {TEST_PLAN_FILE}, fixed {planned['fixed']}")
    print(f"headline row:  {planned['headline']}")
    if args.rehearse:
        print(f"rehearsal:     {REHEARSAL_SPLIT}, no receipt is written\n")
    else:
        print(f"reading:       {TEST_SPLIT}, {_ordinal(len(earlier) + 1)} time\n")

    device = select_device(args.device)
    print(describe_device(device))

    results: list[Scored | Blocked] = []
    records: Sequence[Record] = ()
    facts: Facts | None = None

    for row in wanted:
        config = configs[row]
        print(f"\n--- {row}: {config.name}  ({ROWS[row]})")
        try:
            inputs = load_inputs(config, split, args.limit if args.rehearse else None)
        except INPUT_ERRORS as error:
            print(error.args[0] if isinstance(error, KeyError) else error)
            return 1

        records = inputs.records
        if facts is None:
            # Properties of the split and the index corpus rather than of a
            # configuration, so they are computed once and shared by every row.
            facts = split_facts(inputs, config)
            print(facts.summary)

        results.append(
            score(row, config, inputs, split, args, device, facts)
        )

    print()
    print(report(results, records, facts, planned))

    document = {
        "schema": SCHEMA,
        "split": split,
        "records": len(records),
        "duplicates": {
            "indexed": list(facts.duplicates),
            "core_train": list(facts.train_duplicates),
        },
        "unreadable": {
            f"{cell.record_type}/{cell.language}": list(ids)
            for cell, ids in facts.unreadable.items()
        },
        "rows": {result.row: row_document(result) for result in results},
    }
    if args.json:
        Path(args.json).write_text(json.dumps(document, indent=2) + "\n")
        print(f"\nwrote {args.json}")

    if args.rehearse:
        print("\nrehearsal: no receipt written, the gold split was not read")
        return 0

    written = receipt(
        read=date.today(),
        plan=planned,
        rows=document["rows"],
        records=len(records),
        earlier=earlier,
    )
    TEST_RUN_FILE.write_text(json.dumps(written, indent=2, default=list) + "\n")
    print(f"\nwrote {TEST_RUN_FILE}; the gold test split is now read")
    return 0


def fix_plan(configs: Mapping[str, ExperimentConfig]) -> int:
    """Write the plan, refusing to move it once the split has been read."""
    if TEST_RUN_FILE.exists():
        print(
            f"{TEST_RUN_FILE} says the gold test split has already been read, so "
            "re-fixing the plan would date a decision after its answer. Record a "
            "justification with --justify instead."
        )
        return 1

    document = plan(
        headline=HEADLINE,
        rows=configs,
        paths={row: ROWS[row] for row in configs},
        rationale=RATIONALE,
    )
    TEST_PLAN_FILE.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {TEST_PLAN_FILE}")
    for row, entry in document["rows"].items():
        print(f"  {row:<12} {entry['digest']}  {entry['config']}")
    print("\nCommit it before reading the split; the run refuses a row it "
          "does not already name.")
    return 0


def score(
    row: str,
    config: ExperimentConfig,
    inputs,
    split: str,
    args,
    device: str,
    facts: Facts,
) -> Scored | Blocked:
    """Predict one row, write its trees, and score it every way the ticket asks."""
    store = ArtifactStore(args.artifacts, data_revision=inputs.revision)
    started = time.perf_counter()
    try:
        candidates = predict(
            inputs.records,
            config,
            inputs.vocabulary,
            inputs.index_records,
            store,
            device=device,
            name_qualifiers=inputs.name_qualifiers,
        )
    except MissingApiKey as error:
        # A labelled row rather than a failed run: the adjudication stage is
        # what ticket 13 left owed, and the rest of the report does not depend
        # on it. Reported as blocked, never as a zero.
        print(f"blocked: {error}")
        return Blocked(row=row, config_path=ROWS[row], reason=str(error))
    elapsed = time.perf_counter() - started
    print(f"predicted {len(candidates)} records in {elapsed:.1f}s")

    gold = {record.id: record.subjects for record in inputs.records}
    cells = {record.id: (record.type, record.lang) for record in inputs.records}
    predictions = {result.record_id: result.codes for result in candidates}
    bands = frequency_bands()

    report_all = evaluate(gold, predictions, bands, cells)
    kept = {id for id in gold if id not in set(facts.duplicates)}
    report_kept = evaluate(
        {id: codes for id, codes in gold.items() if id in kept},
        predictions,
        bands,
        cells,
    )

    destination = Path(args.out) / row
    submission = destination / "submission"
    # Two things the submission format cannot represent, both named rather than
    # dropped quietly: a ranking shorter than 50, and a record whose cell has a
    # blank half — the writer refuses each, and a run that let either vanish
    # would report a figure over a smaller split than the one it opened.
    short = [
        result.record_id
        for result in candidates
        if len(result.codes) < CODES_PER_RECORD
    ]
    cellless = [record.id for record in inputs.records if not all(cells[record.id])]
    writeable = [
        result
        for result in candidates
        if result.record_id not in set(short) | set(cellless)
    ]
    write_submission(
        [result.as_prediction(CODES_PER_RECORD) for result in writeable],
        cells,
        submission,
    )
    print(f"wrote {len(writeable)} submission files under {submission}")
    if short or cellless:
        print(
            f"warning: {len(short)} record(s) got fewer than {CODES_PER_RECORD} "
            f"codes and {len(cellless)} have a blank cell; both are in every "
            "local figure and in neither tree"
        )

    official = None
    try:
        official = score_officially(
            inputs.records, candidates, facts.unreadable, destination / "official"
        )
    except OfficialScorerUnavailable as error:
        print(f"official scorer not run: {error}")

    return Scored(
        row=row,
        config_path=ROWS[row],
        config=config,
        report=report_all,
        without_duplicates=report_kept,
        official=official,
        submission=submission,
        elapsed=elapsed,
    )


def score_officially(
    records: Sequence[Record],
    candidates,
    unreadable: Mapping[Cell, Sequence[str]],
    destination: Path,
) -> Official:
    """Run the organizers' script over a gold and a prediction tree.

    Restricted to the cells their reader can index into: it seeds its result
    with five record types and two languages and then subscripts that dictionary
    by directory name, so a French record raises inside their code. The excluded
    records are named in the report and scored by the local evaluator, which is
    checked against this figure on the cells both can see.
    """
    module = official_module()
    excluded = {id for ids in unreadable.values() for id in ids}
    readable = [record for record in records if record.id not in excluded]
    by_id = {result.record_id: result for result in candidates}
    scoreable = [
        record
        for record in readable
        if len(by_id[record.id].codes) >= CODES_PER_RECORD
    ]
    if len(scoreable) != len(readable):
        # Their reader pairs the trees by basename and their validator refuses a
        # gold file with no prediction beside it, so a short ranking cannot just
        # be left out — but nor can it be padded, since the padding would be
        # scored. Said out loud, because it would otherwise shrink the split the
        # headline is quoted over.
        print(
            f"warning: {len(readable) - len(scoreable)} readable record(s) have "
            f"fewer than {CODES_PER_RECORD} codes and are outside the official "
            "trees; the local figures still cover them"
        )

    write_gold_tree(scoreable, destination / "gold")
    write_submission(
        [by_id[record.id].as_prediction(CODES_PER_RECORD) for record in scoreable],
        {record.id: (record.type, record.lang) for record in scoreable},
        destination / "pred",
    )

    true_dict = module.read_gnd_files(str(destination / "gold"), True)
    pred_dict = module.read_gnd_files(str(destination / "pred"), False)
    if not module.validate_directory_structure(true_dict, pred_dict):
        raise OfficialScorerUnavailable(
            "the organizers' own validator rejected the tree this run wrote"
        )

    module.evaluate_and_save_to_excel(
        str(destination / "results"),
        "final_test.xlsx",
        true_dict,
        pred_dict,
        list(OFFICIAL_KS),
    )

    import pandas as pd

    combined = pd.read_excel(
        destination / "results" / "final_test.xlsx",
        sheet_name="Record Type and Language",
    )
    overall = combined[combined["Record Type"] == "Overall"].iloc[0]
    return Official(
        at_k={
            k: {
                "precision": float(overall[f"precision_{k}"]),
                "recall": float(overall[f"recall_{k}"]),
                "f1": float(overall[f"f1_{k}"]),
            }
            for k in OFFICIAL_KS
        },
        cells=len(combined) - 1,
        records=len(scoreable),
    )


def official_module():
    """The organizers' script, loaded by path because of its hyphenated name."""
    import importlib.util

    script = Path(__file__).resolve().parent.parent / "official_eval" / (
        "llms4subjects-evaluation.py"
    )
    if not script.exists():
        raise OfficialScorerUnavailable(f"{script} is not vendored in this checkout")
    try:
        import pandas  # noqa: F401
        import openpyxl  # noqa: F401
    except ImportError as error:  # pragma: no cover - environment failure path
        raise OfficialScorerUnavailable(
            f"the official scorer needs pandas and openpyxl: {error}"
        ) from error

    spec = importlib.util.spec_from_file_location("official_eval", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Rendering ----------------------------------------------------------------


def report(
    results: Sequence[Scored | Blocked],
    records: Sequence[Record],
    facts: Facts,
    planned: Mapping,
) -> str:
    """The whole report, in the order the ticket asks for it."""
    scored = [result for result in results if isinstance(result, Scored)]
    blocked = [result for result in results if isinstance(result, Blocked)]
    headline = next(
        (row for row in scored if row.row == planned.get("headline")), None
    )

    parts = [render_leaderboard(scored, headline)]
    if headline is not None:
        parts += [
            render_aggregations(headline),
            render_duplicates(scored, facts),
            render_bands(scored),
            render_cells(headline),
        ]
    parts.append(render_unreadable(facts))
    if blocked:
        parts.append(render_blocked(blocked))
    return "\n\n".join(parts)


def render_leaderboard(
    scored: Sequence[Scored], headline: Scored | None
) -> str:
    """This run's rows beside the four published ones, at the official figures.

    Precision reads low against intuition and not against the task: at 2.40 gold
    labels per test record a perfect system scores P@5 = 0.48, so RUC's 0.25 is
    roughly half of what is achievable rather than a quarter.
    """
    lines = [
        "The published tib-core leaderboard, and this run beside it "
        "(official aggregation)",
        "",
        "| system | P@5 | R@5 | P@10 | R@10 | Avg R@k |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for team, p5, r5, p10, r10, avg in LEADERBOARD:
        lines.append(
            f"| {team} | {p5:.2f} | {r5:.2f} | {p10:.2f} | {r10:.2f} | {avg:.2f} |"
        )
    for row in scored:
        if row.official is None:
            lines.append(f"| **this run — {row.row}** | (official scorer not run) ||||")
            continue
        mark = "**" if row is headline else ""
        lines.append(
            f"| {mark}this run — {row.row}{mark} | "
            f"{row.official.precision(5):.4f} | {row.official.recall(5):.4f} | "
            f"{row.official.precision(10):.4f} | {row.official.recall(10):.4f} | "
            f"{row.official.mean_recall:.4f} |"
        )
    if headline is not None and headline.official is not None:
        lines += [
            "",
            f"The four published rows are quoted to two decimals, as published. "
            f"This run's are the organizers' script's own `Overall` row over the "
            f"{headline.official.cells} cells its reader can index into "
            f"({headline.official.records} records).",
        ]
    return "\n".join(lines)


def render_aggregations(row: Scored) -> str:
    """Micro against official-macro, and the local evaluator against the script."""
    lines = [
        f"The two aggregations, headline row `{row.row}` "
        f"({row.config.name}, {row.config_path})",
        "",
        "| aggregation | " + " | ".join(f"R@{k}" for k in OFFICIAL_KS) + " |",
        "|---" * (len(OFFICIAL_KS) + 1) + "|",
        "| record-micro | "
        + " | ".join(f"{row.report.micro.recall(k):.4f}" for k in OFFICIAL_KS)
        + " |",
        "| official-macro, every cell | "
        + " | ".join(
            f"{row.report.official_macro.recall(k):.4f}" for k in OFFICIAL_KS
        )
        + " |",
    ]
    if row.official is not None:
        lines.append(
            "| official scorer, de/en cells | "
            + " | ".join(f"{row.official.recall(k):.4f}" for k in OFFICIAL_KS)
            + " |"
        )
    lines += ["", render(row.report, ks=OFFICIAL_KS)]
    return "\n".join(lines)


def render_duplicates(scored: Sequence[Scored], facts: Facts) -> str:
    """The caveat docs/spec.md commits to: the score with and without them."""
    share = len(facts.duplicates) / facts.records if facts.records else 0.0
    lines = [
        f"With and without the {len(facts.duplicates)} records ({share:.1%}) "
        "whose title and abstract exactly duplicate an indexed training document",
        "",
        "| row | micro R@10 | without | change | official-macro R@10 | without | change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scored:
        micro = row.report.micro.recall(10)
        micro_kept = row.without_duplicates.micro.recall(10)
        macro = row.report.official_macro.recall(10)
        macro_kept = row.without_duplicates.official_macro.recall(10)
        lines.append(
            f"| {row.row} | {micro:.4f} | {micro_kept:.4f} | "
            f"{micro_kept - micro:+.4f} | {macro:.4f} | {macro_kept:.4f} | "
            f"{macro_kept - macro:+.4f} |"
        )
    lines += [
        "",
        "A neighbour retriever is unusually strong on precisely these records: "
        "the document it ranks first carries the answer under a different "
        "TIBKAT id. The `without` columns are the honest figure for a record "
        "the index has never seen.",
        "",
        f"The set is taken against the {len(facts.duplicates)} documents this "
        f"run indexed. Against `core_train` alone — the training split "
        f"docs/spec.md's figure of 163 was stated over — it is "
        f"{len(facts.train_duplicates)}.",
    ]
    return "\n".join(lines)


def render_bands(scored: Sequence[Scored]) -> str:
    """Micro recall by frozen band, zero-shot included, with its support."""
    lines = [
        "By frozen frequency band (micro recall, gold assignments in brackets)",
        "",
        "| row | " + " | ".join(f"{band} R@10" for band in BANDS) + " |",
        "|---" * (len(BANDS) + 1) + "|",
    ]
    for row in scored:
        cells = [
            f"{row.report.by_band[band].recall(10):.4f} "
            f"({row.report.by_band[band].assignments})"
            for band in BANDS
        ]
        lines.append(f"| {row.row} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_cells(row: Scored) -> str:
    """Which cells carry the official figure, which is why both are reported."""
    dominant = row.report.divergence.dominant()
    share = sum(cell.mean_recall_share for cell in dominant)
    lines = [
        f"{len(dominant)} of {len(row.report.divergence.cells)} scoring cells "
        f"carry {share:.1%} of the official recall figure",
        "",
        "| cell | records | share of records | share of the figure |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {cell.record_type} / {cell.language} | {cell.records} | "
        f"{cell.record_share:.2%} | {cell.mean_recall_share:.2%} |"
        for cell in dominant
    ]
    return "\n".join(lines)


def render_unreadable(facts: Facts) -> str:
    """The cells the organizers' script cannot score, named rather than dropped."""
    unreadable = facts.unreadable
    total = sum(len(ids) for ids in unreadable.values())
    if not unreadable:
        return "Every cell in this split is one the official scorer can read."
    share = total / facts.records if facts.records else 0.0
    lines = [
        f"{total} records ({share:.1%}) in {len(unreadable)} cell(s) "
        "the official scorer's reader cannot index into",
        "",
        "| cell | records |",
        "|---|---:|",
    ]
    lines += [
        f"| {cell.record_type} / {cell.language} | {len(ids)} |"
        for cell, ids in unreadable.items()
    ]
    lines += [
        "",
        "Their reader seeds its result with `de` and `en` and then subscripts "
        "it by directory name, so these raise inside their code. They are "
        "scored by the local evaluator and are in every micro figure above, "
        "and they are one more reason the micro figure travels beside the "
        "official one.",
    ]
    return "\n".join(lines)


def render_blocked(blocked: Sequence[Blocked]) -> str:
    lines = ["Rows that did not run", "", "| row | config | why |", "|---|---|---|"]
    lines += [
        f"| {row.row} | `{row.config_path}` | {row.reason.splitlines()[0]} |"
        for row in blocked
    ]
    return "\n".join(lines)


def row_document(result: Scored | Blocked) -> dict:
    """One row as the receipt records it."""
    if isinstance(result, Blocked):
        return {"config": result.config_path, "blocked": result.reason}

    return {
        "config": result.config_path,
        "name": result.config.name,
        "digest": configuration_digest(result.config),
        "seconds": round(result.elapsed, 1),
        "micro": _at_k(result.report.micro),
        "official_macro": _at_k(result.report.official_macro),
        "official_scorer": (
            {
                "cells": result.official.cells,
                "records": result.official.records,
                "at_k": {
                    str(k): result.official.at_k[k] for k in OFFICIAL_KS
                },
            }
            if result.official is not None
            else None
        ),
        "without_duplicates": {
            "micro": _at_k(result.without_duplicates.micro),
            "official_macro": _at_k(result.without_duplicates.official_macro),
        },
        "bands": {
            band: {
                "assignments": result.report.by_band[band].assignments,
                "at_k": _at_k(result.report.by_band[band]),
            }
            for band in BANDS
        },
        "by_language_micro": {
            language: _at_k(metrics)
            for language, metrics in result.report.by_language_micro.items()
        },
        "by_type": {
            record_type: _at_k(metrics)
            for record_type, metrics in result.report.by_type.items()
        },
    }


def _at_k(metrics) -> dict[str, dict[str, float]]:
    return {str(k): dict(metrics.at_k[k]) for k in OFFICIAL_KS}


def _ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third"}.get(n, f"{n}th")


if __name__ == "__main__":
    raise SystemExit(main())
