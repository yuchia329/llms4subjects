"""Contrastively fine-tune one shortlisted encoder. Runs on the GPU host.

    ssh nlp2
    cd ~/projects/llms4subjects
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    .venv/bin/python -u scripts/train_encoder.py configs/rung3.yaml \\
        --negatives-from configs/rung2-bge-m3.yaml \\
        --batch-size 8 --negatives 2 --gradient-checkpointing \\
        > artifacts/adapters/bge-m3-contrastive-v1/train.log 2>&1

The only thing in this project that needs CUDA, and the only one that writes
weights. Everything it produces is a LoRA adapter of a few megabytes, which is
copied back and applied on Apple Silicon for indexing and evaluation — so no
number in the results table depends on this host still being reachable when
training is over.

Two configs, because a fine-tune has two encoders in it. The first is the rung-3
config being trained *for*: it names the base model and where the adapter goes.
The second is the untrained rung-2 config the hard negatives were mined with,
which is a different `encoder` section — no adapter — and therefore a different
artifact key. Naming it here rather than deriving it keeps the mined artifact
addressable by the configuration that produced it, and makes the run fail on a
missing file rather than silently training with in-batch negatives alone.

The dataset is not copied to the host; it is rebuilt there with the same command
used locally, and the run refuses to start if that rebuild produced a different
dataset from the one the negatives were mined against.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms4subjects.artifacts import (  # noqa: E402
    HARD_NEGATIVE_STAGE,
    ArtifactStore,
    MissingHardNegatives,
    load_hard_negatives,
)
from llms4subjects.config import load_experiment  # noqa: E402
from llms4subjects.corpus import (  # noqa: E402
    MissingDataset,
    data_revision,
    load_label_translations,
    load_name_qualifiers,
    load_split,
    load_vocabulary,
)
from llms4subjects.hardware import HardwareUnavailable, require_cuda  # noqa: E402
from llms4subjects.finetune import (  # noqa: E402
    ADAPTER_MANIFEST,
    TrainingSettings,
    adapter_manifest,
    apply_lora,
    examples,
    save_adapter,
    train,
)
from llms4subjects.pipeline import label_texts  # noqa: E402
from llms4subjects.stages import encoders  # noqa: E402
from mine_hard_negatives import DEFAULT_DEPTH, DEFAULT_SPLIT  # noqa: E402
from run_experiment import VOCABULARY  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="the rung-3 config being trained for")
    parser.add_argument(
        "--negatives-from",
        required=True,
        help="the untrained config the hard negatives were mined with",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="records to train on")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision", default="bf16", choices=["bf16", "fp32"],
        help="forward-pass precision; bf16 halves activation memory on an A100",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="recompute activations in the backward pass; for a shared GPU",
    )
    parser.add_argument(
        "--limit", type=int, help="train on the first N records, for a smoke run"
    )
    parser.add_argument("--force", action="store_true", help="retrain over an adapter")
    args = parser.parse_args(argv)

    config = load_experiment(args.config)
    if config.encoder.adapter is None:
        print(
            f"{args.config} names no `encoder.adapter`, so there is nowhere to "
            "write the fine-tune. A rung-3 config declares its adapter path."
        )
        return 1

    mined_with = encoders.pinned(load_experiment(args.negatives_from))
    if mined_with.encoder.name != config.encoder.name:
        # Negatives are one encoder's confusions. Another's are a different
        # model's mistakes, and training against them is not the experiment.
        print(
            f"the negatives were mined with {mined_with.encoder.name} and this "
            f"config trains {config.encoder.name}."
        )
        return 1

    directory = Path(config.encoder.adapter)
    if (directory / ADAPTER_MANIFEST).exists() and not args.force:
        print(f"{directory} already holds an adapter; --force to retrain.")
        return 0

    try:
        device = require_cuda("contrastive fine-tuning")
    except HardwareUnavailable as error:
        print(error)
        return 1

    try:
        revision = data_revision()
        records = load_split(args.split)
        vocabulary = load_vocabulary(VOCABULARY)
        name_qualifiers = load_name_qualifiers(VOCABULARY)
        translations = (
            load_label_translations() if config.label_text.bilingual else {}
        )
    except MissingDataset as error:
        print(error)
        return 1

    store = ArtifactStore(args.artifacts, data_revision=revision)
    try:
        mined = load_hard_negatives(store, mined_with, args.split, args.depth)
    except MissingHardNegatives as error:
        print(error)
        print(
            "\nIf they were mined on another machine, the dataset revisions "
            f"have to match: this one is {revision}."
        )
        return 1

    if args.limit:
        records = records[: args.limit]

    settings = TrainingSettings(
        epochs=args.epochs,
        batch_size=args.batch_size,
        negatives=args.negatives,
        learning_rate=args.learning_rate,
        rank=args.rank,
        seed=args.seed,
        document_length=config.encoder.max_length,
        gradient_checkpointing=args.gradient_checkpointing,
        precision=args.precision,
    )

    texts = label_texts(config, vocabulary, name_qualifiers, translations)
    pairs = examples(
        records, mined, settings.negatives, vocabulary=set(texts)
    )
    gold = {record.id: set(record.subjects) for record in records}

    resolved = encoders.resolve(config.encoder)
    print(f"training:      {config.name}  ({args.config})")
    print(f"base:          {config.encoder.name} @ {resolved.revision[:12]}")
    print(f"data revision: {revision}")
    print(f"negatives:     {store.key(HARD_NEGATIVE_STAGE, mined_with)} "
          f"({args.split}, depth {args.depth})")
    print(f"pairs:         {len(pairs)} over {len(records)} records")
    print(f"settings:      {settings.as_dict()}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        config.encoder.name, device=device, **resolved.kwargs()
    )
    model = apply_lora(model, config.encoder.name, settings)

    run = train(
        model,
        pairs,
        texts,
        gold,
        settings,
        resolved.prefixes,
        device=device,
    )

    written = save_adapter(
        model,
        directory,
        adapter_manifest(
            base=config.encoder.name,
            revision=resolved.revision,
            data_revision=revision,
            negatives=store.key(HARD_NEGATIVE_STAGE, mined_with),
            hyperparameters=settings.as_dict(),
            losses=run.losses,
            seconds=run.seconds,
            host=platform.node(),
            examples=len(pairs),
            trainable=run.trainable,
            skipped=run.skipped,
        ),
    )

    print(
        f"\ntrained {run.steps} steps in {run.seconds / 60:.1f} min on "
        f"{run.trainable / 1e6:.1f}M trainable parameters"
        + (f", {run.skipped} batches skipped for memory" if run.skipped else "")
    )
    print(f"loss: {' -> '.join(f'{loss:.4f}' for loss in run.losses)}")
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
