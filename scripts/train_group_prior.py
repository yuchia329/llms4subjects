"""Fit the 66-way classification-group head for one config, and score it.

    python scripts/train_group_prior.py configs/rung1-prior.yaml
    python scripts/train_group_prior.py configs/rung1-prior.yaml --force
    python scripts/train_group_prior.py configs/rung1-prior.yaml --report

The head is one logistic regression per group over the configured encoder's
document vectors. Fitting it costs seconds once those vectors are cached, which
is why this is not one of the three tickets that need the `nlp2` host.

It trains on **every** document in the config's `index.corpora`, not on the
subset `index.size` samples for the retrieval index. Index size and training-set
size vary independently by design (docs/spec.md, story 20), so that "more
neighbours" and "more training signal" stay separately attributable — and it is
also the better classifier by a wide margin: on `configs/rung1-prior.yaml` the
8,000-document index sample gives 121 documents per group and dev top-1 accuracy
0.6137, while all 32,043 tib-core train records give the 485 per group story 28
quotes, and 0.7107.

It writes `artifacts/group_prior/<key>/head.npz`, keyed by encoder, index and
group-prior configuration, which is what `llms4subjects.pipeline.predict` reads
when `group_prior.enabled` is on. `--report` scores whatever is already there
and writes nothing; `--force` refits over an existing head.

The accuracy this prints is a deliverable of ticket 11 rather than a debugging
aid: a boost is only worth its weight if the groups it points at are the right
ones, and "the true groups are within the top two predicted" is the figure that
says so.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms4subjects.artifacts import (  # noqa: E402
    GROUP_PRIOR_FILE,
    GROUP_PRIOR_STAGE,
    ArtifactStore,
    CachedEncoder,
    MissingGroupPrior,
    load_group_prior,
    save_group_prior,
)
from llms4subjects.config import load_experiment  # noqa: E402
from llms4subjects.hardware import describe_device, select_device  # noqa: E402
from llms4subjects.stages import encoders, group_prior  # noqa: E402
from run_experiment import (  # noqa: E402
    FORBIDDEN_SPLIT,
    INPUT_ERRORS,
    load_inputs,
)

# How deep into the predicted ranking the report looks. Two is the one the
# ticket asks for; one and three are there to say whether two is the right cut.
DEPTHS = (1, 2, 3)

# Accelerate's BLAS raises the floating-point flags on matmuls whose inputs and
# results are finite (numpy 2.1 on Apple Silicon), and scikit-learn does one per
# lbfgs iteration. Narrowed to that message so a real numerical failure inside
# the fit still reaches the console.
warnings.filterwarnings(
    "ignore",
    message="(divide by zero|overflow|invalid value) encountered in matmul",
    category=RuntimeWarning,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="a config with group_prior.enabled true")
    parser.add_argument("--split", default="core_dev", help="the split to score on")
    parser.add_argument("--limit", type=int, help="score only the first N records")
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    parser.add_argument(
        "--force", action="store_true", help="refit over an existing head"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="score the fitted head and write nothing",
    )
    args = parser.parse_args(argv)

    if args.split == FORBIDDEN_SPLIT:
        print(f"{FORBIDDEN_SPLIT} is the gold test split; see ticket 17.")
        return 1

    config = load_experiment(args.config)
    if not config.group_prior.enabled:
        # The key includes the whole `group_prior` section, so a head fitted
        # from a config with the flag off would be filed under a key no run can
        # ask for.
        print(
            f"{args.config} has group_prior.enabled false; the head is keyed by "
            "that section, so fitting from it would write where no run reads"
        )
        return 1

    try:
        inputs = load_inputs(config, args.split, args.limit)
    except INPUT_ERRORS as error:
        # `KeyError` is an unknown --split, and quotes its argument; the
        # rest already read as sentences.
        print(error.args[0] if isinstance(error, KeyError) else error)
        return 1

    device = select_device(args.device)
    store = ArtifactStore(args.artifacts, data_revision=inputs.revision)
    print(f"experiment:    {config.name}  ({args.config})")
    print(f"data revision: {inputs.revision}")
    print(describe_device(device))

    group_of_code = group_prior.group_of_code(inputs.vocabulary.values())
    all_groups = group_prior.groups(inputs.vocabulary.values())
    print(
        f"{len(all_groups)} classification groups over "
        f"{len(group_of_code)} of {len(inputs.vocabulary)} vocabulary entries"
    )

    encoder = CachedEncoder(
        encoders.load(config.encoder, device), store, config
    )

    # Every document in the corpora, not the index sample. See the module
    # docstring: the retrieval index's size is not the head's training size.
    training = inputs.index_records
    path = store.path(GROUP_PRIOR_STAGE, config, GROUP_PRIOR_FILE)

    if args.report:
        try:
            prior = load_group_prior(store, config)
        except MissingGroupPrior as error:
            print(error)
            return 1
        print(f"reading {path}")
    elif path.exists() and not args.force:
        print(
            f"{path} already holds a head for this config; --force refits it, "
            "--report scores it"
        )
        return 1
    else:
        prior = _fit(encoder, training, all_groups, group_of_code, config)
        print(f"wrote {save_group_prior(store, config, prior)}")

    _report(
        prior,
        encoder,
        training,
        inputs.records,
        args.split,
        group_of_code,
    )
    return 0


def _fit(encoder, training, all_groups, group_of_code, config):
    """One logistic regression per group over the index selection's vectors."""
    targets = group_prior.record_groups(training, group_of_code)
    covered = sum(1 for groups in targets if groups)
    present = len({group for groups in targets for group in groups})
    print(
        f"fitting on all {len(training)} documents of the index corpora — "
        f"{covered} carry at least one group, and they cover {present} of the "
        f"{len(all_groups)} groups "
        f"({len(training) / max(len(all_groups), 1):.0f} documents per group)"
    )

    started = time.perf_counter()
    vectors = encoder.encode_documents([record.text for record in training])
    encoded = time.perf_counter() - started

    started = time.perf_counter()
    prior = group_prior.train(vectors, targets, all_groups, config.group_prior)
    print(
        f"encoded in {encoded:.1f}s, fitted {len(all_groups)} groups in "
        f"{time.perf_counter() - started:.1f}s "
        f"(regularization C={config.group_prior.regularization})"
    )
    return prior


def _report(prior, encoder, training, records, split, group_of_code):
    """Accuracy on the split, with the fit on its own training documents beside it."""
    print()
    rows = [
        ("training corpora", training),
        (split, records),
    ]
    reports = {}
    for name, subset in rows:
        vectors = encoder.encode_documents([record.text for record in subset])
        reports[name] = group_prior.accuracy(
            prior.distributions(vectors),
            group_prior.record_groups(subset, group_of_code),
            depths=DEPTHS,
        )

    print("### Group classifier accuracy\n")
    print(
        "| documents | scored | true groups per record | "
        + " | ".join(f"any in top {depth}" for depth in DEPTHS)
        + " | "
        + " | ".join(f"all in top {depth}" for depth in DEPTHS)
        + " |"
    )
    print("|---" + "|---:" * (2 + 2 * len(DEPTHS)) + "|")
    for name, report in reports.items():
        print(
            f"| {name} | {report.records} | {report.groups_per_record:.2f} | "
            + " | ".join(f"{report.any_within[d]:.4f}" for d in DEPTHS)
            + " | "
            + " | ".join(f"{report.all_within[d]:.4f}" for d in DEPTHS)
            + " |"
        )

    scored = reports[split]
    print(
        f"\nOn {split}: the prior's top group is one of the record's true "
        f"groups {scored.any_within[1]:.1%} of the time, and every true group "
        f"is within its top two {scored.all_within[2]:.1%} of the time."
    )


if __name__ == "__main__":
    raise SystemExit(main())
