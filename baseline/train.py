"""Re-run the rejected classifier on the clean splits (ticket 14).

Trains `bert-base-multilingual-cased` with a dense output layer over the 14,607
`core_train` labels, validates on `core_dev`, and writes ranked predictions for
scoring **locally** through `llms4subjects.stages.evaluator`:

    python -m baseline.train --smoke                  # first training step, laptop
    python -m baseline.train --epochs 15              # the real run, on nlp2
    python -m baseline.train --predict-only ...        # more predictions, same checkpoint
    python -m baseline.score artifacts/baseline/mbert-dense/predictions-core_dev.json

The full run is a GPU-host workload — 32,043 documents against a 14,607-way head
— so it goes to `nlp2`, and `--smoke` takes a prefix of the split so the loop can
be exercised on MPS first. Nothing here scores anything: predictions come back
from the host as JSON and the numbers come from the shared evaluator, so the
baseline row is produced by the same code path as every pipeline result.

Written outputs, under `artifacts/baseline/<run>/`:

| file | what it is |
|---|---|
| `labels.json` | the output layer in column order; part of the checkpoint |
| `checkpoint.pt` | the trained weights (the original never saved any) |
| `run.json` | configuration, per-epoch losses, wall clock, device |
| `predictions-<split>.json` | 50 ranked codes per record |

The gold test set stays closed: `--predict core_test` refuses to run without
`--open-test-set`, which is ticket 17's to pass, once.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from llms4subjects.contracts import CODES_PER_RECORD, Code
from llms4subjects.corpus import MissingDataset, load_split
from llms4subjects.hardware import describe_device, select_device
from llms4subjects.paths import ARTIFACT_DIR

from .classifier import (
    MAX_LENGTH,
    MODEL_NAME,
    SubjectDataset,
    build_model,
    build_tokenizer,
    rank,
    train as train_model,
)
from .labels import load_labels, output_layer, save_labels, unreachable_assignments

# docs/spec.md excludes reviving it, and `classifier.py` records what it did.
GRAPH_EXCLUSION = (
    "The GCN label-graph refiner is not revived. It modelled a hierarchy the "
    "vocabulary does not contain (zero skos:broader triples), its edges came "
    "from a mapping that collapsed each of 65 classification groups to one "
    "label — 130 of 14,607 labels connected, the rest isolated — and its "
    "output was a batch-constant vector that could only shift the head's bias."
)

TEST_SPLIT = "core_test"

# The splits a run may be asked to predict. `core_train` is absent on purpose:
# ranking the split the head memorised measures memorisation, not the approach.
PREDICTABLE = ("core_dev", TEST_SPLIT)

TEST_SET_CLOSED = (
    f"{TEST_SPLIT} is the gold test set, opened once at the end of the project "
    "(ticket 17). Pass --open-test-set only as part of that run."
)


def training_summary(
    *,
    run: str,
    device: str,
    codes: Sequence[Code],
    records: int,
    args_dict: dict,
    epochs: list[dict],
    seconds: float,
) -> dict:
    """Everything needed to read the resulting row a year from now.

    The ticket asks for the training configuration and its wall-clock cost, so
    they are one JSON file written beside the checkpoint rather than a number
    remembered from a terminal.
    """
    return {
        "run": run,
        "model": args_dict.get("model", MODEL_NAME),
        "approach": "dense output layer over the core_train label set",
        "device": device,
        "host": platform.node(),
        "labels": len(codes),
        "records": records,
        "config": {
            "epochs": args_dict.get("epochs"),
            "batch_size": args_dict.get("batch_size"),
            "lr": args_dict.get("lr"),
            "max_length": args_dict.get("max_length"),
            "optimizer": "AdamW",
            "loss": "BCEWithLogitsLoss",
            "pooling": "CLS",
        },
        "epochs_log": epochs,
        "seconds": seconds,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_component": {"revived": False, "reason": GRAPH_EXCLUSION},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="train on a small prefix of the split, to reach the first "
        "training step on a laptop",
    )
    parser.add_argument(
        "--records", type=int, default=None, help="use only the first N training records"
    )
    parser.add_argument(
        "--eval-records",
        type=int,
        default=None,
        help="use only the first N records of every scored split, for smoke runs",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--run", default=None, help="name of the artifact directory")
    parser.add_argument("--artifacts", default=str(ARTIFACT_DIR), help="artifact root")
    parser.add_argument(
        "--predict",
        nargs="*",
        default=["core_dev"],
        help="splits to write ranked predictions for",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="skip training: load the run directory's checkpoint and write "
        "predictions from it",
    )
    parser.add_argument(
        "--open-test-set",
        action="store_true",
        help=f"required to predict {TEST_SPLIT}; ticket 17 only",
    )
    args = parser.parse_args(argv)
    unknown = [split for split in args.predict if split not in PREDICTABLE]
    if unknown:
        # Checked before the epochs rather than after them: a typo that only
        # surfaced in the prediction loop would discard a finished GPU run.
        parser.error(
            f"unknown --predict split(s): {', '.join(unknown)} "
            f"(known: {', '.join(PREDICTABLE)})"
        )
    if args.smoke:
        args.records = args.records or 256
        args.eval_records = args.eval_records or 64
        args.epochs = 1
        args.batch_size = min(args.batch_size, 8)
    if args.run is None:
        args.run = "smoke" if args.smoke else "mbert-dense"
    if TEST_SPLIT in args.predict and not args.open_test_set:
        parser.error(TEST_SET_CLOSED)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(select_device(args.device))
    print(describe_device(str(device)))

    try:
        train_records = load_split("core_train")
        dev_records = load_split("core_dev")
    except MissingDataset as error:
        print(error)
        return 1
    if args.records is not None:
        train_records = train_records[: args.records]
    if args.eval_records is not None:
        dev_records = dev_records[: args.eval_records]

    codes = output_layer(train_records)
    print(f"output layer: {len(codes)} labels over {len(train_records)} records")

    total, unreachable = unreachable_assignments(dev_records, codes)
    # Named with the record count, because a `--smoke` prefix is a different
    # measurement from the split's own ceiling and the two look alike printed.
    print(
        f"gold assignments this layer has no column for, over "
        f"{len(dev_records)} core_dev records: {unreachable} of {total}"
        + (f" ({unreachable / total:.1%})" if total else "")
    )

    directory = Path(args.artifacts) / "baseline" / args.run
    directory.mkdir(parents=True, exist_ok=True)

    if args.predict_only:
        # Ticket 17 needs a test row from *this* checkpoint, and a second
        # 1.25 h fine-tune would be a different model however identical the
        # configuration. The saved layer is read rather than rebuilt, because
        # it is the head's column order and not a derivation of the split.
        codes = load_labels(directory / "labels.json")
        print(f"loaded output layer: {len(codes)} labels from {directory}")
    else:
        save_labels(directory / "labels.json", codes)

    tokenizer = build_tokenizer(args.model)
    train_loader = dev_loader = None
    if not args.predict_only:
        train_loader = DataLoader(
            SubjectDataset(train_records, codes, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=True,
        )
        dev_loader = DataLoader(
            SubjectDataset(dev_records, codes, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=False,
        )

    model = build_model(len(codes), args.model).to(device)

    if args.predict_only:
        checkpoint = directory / "checkpoint.pt"
        if not checkpoint.exists():
            print(f"{checkpoint} does not exist; there is nothing to predict from")
            return 1
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"loaded {checkpoint}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        started = time.time()
        history = train_model(
            model, train_loader, dev_loader, optimizer, device, args.epochs
        )
        seconds = round(time.time() - started, 1)
        print(f"trained {args.epochs} epoch(s) in {seconds:.1f}s on {device}")

        torch.save(model.state_dict(), directory / "checkpoint.pt")
        summary = training_summary(
            run=args.run,
            device=str(device),
            codes=codes,
            records=len(train_records),
            args_dict=vars(args),
            epochs=history,
            seconds=seconds,
        )
        (directory / "run.json").write_text(json.dumps(summary, indent=2))

    for split in args.predict:
        write_predictions(
            split=split,
            directory=directory,
            run=args.run,
            model=model,
            codes=codes,
            tokenizer=tokenizer,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            model_name=args.model,
            allow_test=args.open_test_set,
            limit=args.eval_records,
        )

    print(f"wrote {directory}")
    return 0


def write_predictions(
    *,
    split: str,
    directory: Path,
    run: str,
    model,
    codes: Sequence[Code],
    tokenizer,
    device,
    max_length: int,
    batch_size: int,
    model_name: str = MODEL_NAME,
    allow_test: bool = False,
    limit: int | None = None,
) -> Path:
    """50 ranked codes per record of `split`, as JSON for local scoring."""
    if split == TEST_SPLIT and not allow_test:
        # Checked at parse time and again here, so a programmatic caller cannot
        # reach the gold test set by passing the split name alone.
        raise PermissionError(TEST_SET_CLOSED)
    records = load_split(split)
    if limit is not None:
        records = records[:limit]
    loader = DataLoader(
        SubjectDataset(records, codes, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
    )
    rankings = rank(
        model, loader, codes, device, [record.id for record in records], CODES_PER_RECORD
    )
    path = directory / f"predictions-{split}.json"
    path.write_text(
        json.dumps(
            {
                "run": run,
                "split": split,
                "model": model_name,
                "labels": len(codes),
                # What the rankings actually hold, which is `k` unless the layer
                # is smaller than it — as it is on a `--smoke` run.
                "codes_per_record": min(CODES_PER_RECORD, len(codes)),
                "predictions": {
                    record_id: list(ranking) for record_id, ranking in rankings.items()
                },
            },
            indent=0,
        )
    )
    print(f"wrote {path} ({len(rankings)} records)")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
