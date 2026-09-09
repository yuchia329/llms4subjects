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
    if config.encoder.adapter is not None:
        # A fine-tuned config is the one kind that depends on a file no run
        # produces locally, so whether it is here belongs in the same summary
        # as the cache keys.
        print(f"adapter: {_adapter(config)}")
    for stage in STAGES:
        print(f"  {stage:<16} {store.key(stage, config)}")
    return 0


def _adapter(config) -> str:
    """One line on the fine-tune this config reads, or on its absence."""
    from .finetune import AdapterMismatch, read_adapter
    from .stages.encoders import resolve

    try:
        manifest = read_adapter(
            config.encoder.adapter, config.encoder.name, resolve(config.encoder).revision
        )
    except AdapterMismatch as error:
        return f"{config.encoder.adapter} — unusable: {error}"
    losses = manifest.get("losses") or []
    return (
        f"{config.encoder.adapter} — {manifest.get('examples', 0):,} pairs on "
        f"{manifest.get('host', 'an unknown host')}, loss "
        + (" → ".join(f"{loss:.4f}" for loss in losses) or "unrecorded")
    )


if __name__ == "__main__":
    raise SystemExit(main())
