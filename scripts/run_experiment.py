"""Run one experiment config end to end and print its scored report.

    python scripts/run_experiment.py configs/rung1-knn.yaml
    python scripts/run_experiment.py configs/rung1-knn.yaml --limit 200
    python scripts/run_experiment.py configs/rung1-knn.yaml --submission runs/rung1

This is the harness every rung is measured through: it reads the dataset, calls
`llms4subjects.pipeline.predict`, and scores the result with the shared
evaluator, so a number in docs/results.md comes from one code path however the
model that produced it was built.

The split defaults to dev. Scoring `core_test` is refused: the gold test split
is opened once, at the end of the project (ticket 17), which is a decision this
script should not let a flag undo by accident.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llms4subjects.artifacts import ArtifactStore  # noqa: E402
from llms4subjects.config import load_experiment  # noqa: E402
from llms4subjects.contracts import CODES_PER_RECORD  # noqa: E402
from llms4subjects.corpus import (  # noqa: E402
    MissingDataset,
    data_revision,
    frequency_bands,
    load_name_qualifiers,
    load_split,
    load_vocabulary,
)
from llms4subjects.hardware import describe_device, select_device  # noqa: E402
from llms4subjects.pipeline import predict  # noqa: E402
from llms4subjects.stages.evaluator import OFFICIAL_KS, evaluate, render  # noqa: E402
from llms4subjects.stages.submission import write_submission  # noqa: E402

VOCABULARY = "tib-core"

# The one split this script will not score. See ticket 17.
FORBIDDEN_SPLIT = "core_test"


@dataclass(frozen=True)
class Inputs:
    """Everything a run needs from the dataset, loaded once.

    `scripts/ablate_retrievers.py` imports `load_inputs` rather than repeating
    it: the refusal to open the gold test split, the corpora an index is built
    from and the `--limit` truncation are decisions this project makes in one
    place, and a second harness that re-derived them could quietly disagree.
    """

    revision: str
    vocabulary: dict
    name_qualifiers: dict
    records: list
    index_records: list


def load_inputs(config, split: str, limit: int | None = None) -> Inputs:
    """The dataset a config asks for, or `MissingDataset` / `KeyError` saying why."""
    revision = data_revision()
    records = load_split(split)
    return Inputs(
        revision=revision,
        vocabulary=load_vocabulary(VOCABULARY),
        name_qualifiers=load_name_qualifiers(VOCABULARY),
        records=records[:limit] if limit else records,
        index_records=[
            record for corpus in config.index.corpora for record in load_split(corpus)
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="an experiment config, e.g. configs/rung1-knn.yaml")
    parser.add_argument("--split", default="core_dev", help="the split to predict")
    parser.add_argument("--limit", type=int, help="score only the first N records")
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    parser.add_argument("--submission", help="also write a submission tree here")
    args = parser.parse_args(argv)

    if args.split == FORBIDDEN_SPLIT:
        print(
            f"{FORBIDDEN_SPLIT} is the gold test split; it is opened once, at the "
            "end of the project (ticket 17)."
        )
        return 1

    config = load_experiment(args.config)

    try:
        inputs = load_inputs(config, args.split, args.limit)
    except MissingDataset as error:
        print(error)
        return 1
    except KeyError as error:
        # An unknown --split; `corpus` names the ones it knows.
        print(error.args[0] if error.args else error)
        return 1

    revision = inputs.revision
    vocabulary = inputs.vocabulary
    name_qualifiers = inputs.name_qualifiers
    records = inputs.records
    index_records = inputs.index_records

    device = select_device(args.device)
    print(f"experiment:    {config.name}  ({args.config})")
    print(f"data revision: {revision}")
    print(describe_device(device))
    indexed = min(config.index.size or len(index_records), len(index_records))
    print(
        f"predicting {len(records)} {args.split} records against {indexed} of the "
        f"{len(index_records)} documents in {', '.join(config.index.corpora)}"
    )

    store = ArtifactStore(args.artifacts, data_revision=revision)
    started = time.perf_counter()
    candidates = predict(
        records,
        config,
        vocabulary,
        index_records,
        store,
        device=device,
        name_qualifiers=name_qualifiers,
    )
    elapsed = time.perf_counter() - started
    print(f"predicted in {elapsed:.1f}s ({elapsed / max(len(records), 1):.3f}s/record)\n")

    short = [result.record_id for result in candidates if len(result.candidates) < CODES_PER_RECORD]
    if short:
        print(
            f"warning: {len(short)} record(s) got fewer than {CODES_PER_RECORD} "
            f"codes, e.g. {short[0]}; the submission format has no representation "
            "for fewer\n"
        )

    # Scored past the submission's 50 as well, because candidate generation
    # emits `fusion.candidates` and recall there is the ceiling every later
    # stage — reranking, adjudication — can only rank within.
    ceiling = config.fusion.candidates
    report = evaluate(
        gold={record.id: record.subjects for record in records},
        predictions={
            result.record_id: result.codes for result in candidates
        },
        bands=frequency_bands(),
        cells={record.id: (record.type, record.lang) for record in records},
        ks=tuple(OFFICIAL_KS) + (ceiling,),
    )
    print(render(report, ks=OFFICIAL_KS))
    print(f"\ncandidate ceiling at {ceiling}: "
          f"micro R@{ceiling} {report.micro.recall(ceiling):.4f}, "
          f"official R@{ceiling} {report.official_macro.recall(ceiling):.4f}")
    print("  by band  " + "  ".join(
        f"{band} {metrics.recall(ceiling):.4f}"
        for band, metrics in report.by_band.items()
    ))

    if args.submission:
        if short:
            # The writer refuses a ranking that is not exactly 50 codes, and it
            # is right to: the format has no representation for fewer. Say so
            # here rather than tracebacking out of it after a scored run.
            print(
                f"\nrefusing to write a submission: {len(short)} record(s) have "
                f"fewer than {CODES_PER_RECORD} codes"
            )
            return 1

        written = write_submission(
            [result.as_prediction(CODES_PER_RECORD) for result in candidates],
            {record.id: (record.type, record.lang) for record in records},
            args.submission,
        )
        print(f"\nwrote {len(written)} submission files under {args.submission}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
