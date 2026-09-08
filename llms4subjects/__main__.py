"""`python -m llms4subjects <config>` — show what a config resolves to.

Useful before a long run: it prints the artifact keys the run will read and
write, so a cache miss is visible before the GPU time is spent.
"""

from __future__ import annotations

import argparse

from .artifacts import STAGES, ArtifactStore
from .config import load_experiment
from .corpus import MissingDataset, data_revision
from .hardware import describe_device, select_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to an experiment config, e.g. configs/rung1.yaml")
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    args = parser.parse_args(argv)

    config = load_experiment(args.config)
    try:
        revision = data_revision()
    except MissingDataset as error:
        print(error)
        return 1

    store = ArtifactStore(args.artifacts, data_revision=revision)
    print(f"experiment: {config.name}")
    print(f"data revision: {revision}")
    print(describe_device(select_device()))
    for stage in STAGES:
        print(f"  {stage:<16} {store.key(stage, config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
