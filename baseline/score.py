"""Score a baseline prediction file locally, through the shared evaluator.

    python -m baseline.score artifacts/baseline/mbert-dense/predictions-core_dev.json
    python -m baseline.score .../predictions-core_dev.json --markdown

Training happens on `nlp2`; scoring happens here, on purpose. The baseline row
has to be comparable to every pipeline row, and the only way to guarantee that
is for both to come out of `llms4subjects.stages.evaluator` with the same frozen
bands, rather than out of a metrics function that shipped with the model.

`--markdown` prints the row and the band table as they go into docs/results.md,
so the results document is a paste rather than a transcription.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llms4subjects.corpus import MissingDataset, frequency_bands, load_split
from llms4subjects.stages.evaluator import BANDS, evaluate, render

from .labels import load_labels, unreachable_assignments


def load_predictions(path: str | Path) -> dict:
    """A prediction file as written by `baseline.train`."""
    payload = json.loads(Path(path).read_text())
    for key in ("split", "predictions"):
        if key not in payload:
            raise ValueError(f"{path} has no {key!r}; it is not a baseline prediction file")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="a predictions-<split>.json written by baseline.train")
    parser.add_argument("--markdown", action="store_true", help="print the results-document rows")
    args = parser.parse_args(argv)

    path = Path(args.predictions)
    payload = load_predictions(path)
    predictions = payload["predictions"]

    try:
        records = load_split(payload["split"])
    except MissingDataset as error:
        print(error)
        return 1

    gold = {record.id: record.subjects for record in records}
    cells = {record.id: (record.type, record.lang) for record in records}
    missing = sorted(set(gold) - set(predictions))

    report = evaluate(gold, predictions, frequency_bands(), cells)

    print(f"{path}")
    print(f"run: {payload.get('run', '?')}  split: {payload['split']}  model: {payload.get('model', '?')}")
    if missing:
        print(f"records with no prediction (scored as zero): {len(missing)}")
    print()
    print(render(report))

    zero = report.by_band["zero"]
    leaking = any(zero.recall(k) for k in report.ks)
    print()
    print(
        f"zero-shot band: {zero.assignments} gold assignments, recall "
        + ("NOT zero — the output layer is leaking" if leaking else "exactly zero at every k")
    )
    print(
        "  The output layer has one column per label seen in core_train, so a "
        "label that never occurs there has no column and cannot be emitted at "
        "any score. This is a property of the approach, not of this run."
    )

    labels_file = path.parent / "labels.json"
    if labels_file.exists():
        codes = load_labels(labels_file)
        total, unreachable = unreachable_assignments(records, codes)
        share = f" ({unreachable / total:.1%})" if total else ""
        print(
            f"  {unreachable} of {total} gold assignments on this split"
            f"{share} have no column at all."
        )

    if args.markdown:
        print()
        print(markdown(report, payload, path))
    return 0


def markdown(report, payload: dict, path: Path) -> str:
    """The rows this run contributes to docs/results.md."""
    run_file = path.parent / "run.json"
    cost = ""
    if run_file.exists():
        run = json.loads(run_file.read_text())
        hours = run["seconds"] / 3600
        cost = (
            f"{run['config']['epochs']} epochs, batch {run['config']['batch_size']}, "
            f"lr {run['config']['lr']}, {run['device']} on {run['host']}, "
            f"{hours:.2f} h wall clock"
        )

    micro = report.micro
    official = report.official_macro
    lines = [
        "| aggregation | P@5 | R@5 | P@10 | R@10 | R@50 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| micro | {micro.precision(5):.4f} | {micro.recall(5):.4f} | "
        f"{micro.precision(10):.4f} | {micro.recall(10):.4f} | {micro.recall(50):.4f} |",
        f"| official macro-over-cells | {official.precision(5):.4f} | {official.recall(5):.4f} | "
        f"{official.precision(10):.4f} | {official.recall(10):.4f} | {official.recall(50):.4f} |",
        "",
        "| band | gold assignments | micro R@10 | micro R@50 |",
        "|---|---:|---:|---:|",
    ]
    for band in BANDS:
        metrics = report.by_band[band]
        lines.append(
            f"| {band} | {metrics.assignments} | {metrics.recall(10):.4f} | "
            f"{metrics.recall(50):.4f} |"
        )
    if cost:
        lines += ["", f"Training: {cost}."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
