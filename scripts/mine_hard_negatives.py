"""Mine the untrained retriever's own mistakes, for the rung-3 fine-tune.

    python scripts/mine_hard_negatives.py configs/rung2-bge-m3.yaml
    python scripts/mine_hard_negatives.py configs/rung2-gte-base.yaml --depth 50
    python scripts/mine_hard_negatives.py configs/rung2.yaml --limit 500

The hard negatives of a contrastive fine-tune are the codes a retriever ranks
highest and gets wrong. Nothing has to be trained to find them: this runs the
*rung-2* configuration — the untrained encoder at the full tib-core index — over
the split it will be trained on, which is the same retrieval the rung-2 screen
already made, over the same texts.

Which is the point of doing it here rather than on the GPU host. The host has no
cached vectors, no label tower and no reason to build either; a fine-tune that
mined its own negatives would spend an hour of rented hardware reproducing work
the Mac has already done.

**Warm is not guaranteed.** An artifact key is the configuration *and the
dataset revision*, and the dataset revision digests every split file that
exists — so building `all_train.csv` for rung 3 changes it, and the rung-2
vectors are then under a key nothing will ask for again. Mining after that build
recomputes them (about 40 minutes for `bge-m3` on an M4 Pro) unless the cached
matrices are re-keyed by hand. The vectors themselves are still correct: each
file is named by a digest of the exact texts it holds, and none of those texts
changed. See docs/artifacts.md.

The ranking is written as it comes, gold included: filtering is
`finetune.hard_negatives`, which drops the record's whole gold set rather
than one pair's positive, and doing it here would bake one definition of
"negative" into the artifact.

Copy the result to the host with the rsync in docs/artifacts.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms4subjects.artifacts import (  # noqa: E402
    ArtifactStore,
    negatives_filename,
    save_hard_negatives,
)
from llms4subjects.config import load_experiment  # noqa: E402
from llms4subjects.corpus import MissingDataset  # noqa: E402
from llms4subjects.hardware import describe_device, select_device  # noqa: E402
from llms4subjects.pipeline import combine, retrieve  # noqa: E402
from llms4subjects.splits import SplitAlignmentUnverified  # noqa: E402
from llms4subjects.stages import encoders  # noqa: E402
from run_experiment import FORBIDDEN_SPLIT, load_inputs  # noqa: E402

# How deep into each record's ranking is kept. Well past the four negatives a
# fine-tune uses, so that the same artifact serves a run that wants more of them
# and one that filters more of them out: a record's own gold occupies the top of
# this list wherever the retriever was right.
DEFAULT_DEPTH = 50

# The split the fine-tune trains on. The tib-core training records, not the
# all-subjects ones: index size and training-set size are separate knobs
# (docs/spec.md, story 20), and rung 3 moves the first.
DEFAULT_SPLIT = "core_train"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="the untrained config to mine with")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="records to mine for")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--limit", type=int, help="mine only the first N records")
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    parser.add_argument("--force", action="store_true", help="re-mine over an existing artifact")
    args = parser.parse_args(argv)

    if args.split == FORBIDDEN_SPLIT:
        print(f"{FORBIDDEN_SPLIT} is the gold test split; see ticket 17.")
        return 1

    config = load_experiment(args.config)
    if config.encoder.adapter is not None:
        # The negatives are supposed to be the *untrained* retriever's, and a
        # fine-tune trained against its own confusions is a different
        # experiment that nothing in this project has argued for.
        print(
            f"{args.config} names an adapter, so its retriever is already "
            "trained. Mine with the rung-2 config for the same encoder."
        )
        return 1

    # Pinned for the reason `scripts/screen_encoders.py` pins: the vector cache
    # is keyed on the encoder section, and the section in the file carries no
    # revision, so an unpinned config would compute its own copy of vectors the
    # rung-2 screen already paid for.
    config = encoders.pinned(config)

    try:
        inputs = load_inputs(config, args.split, args.limit)
    except (MissingDataset, SplitAlignmentUnverified) as error:
        print(error)
        return 1
    except KeyError as error:
        print(error.args[0] if error.args else error)
        return 1

    store = ArtifactStore(args.artifacts, data_revision=inputs.revision)
    path = store.path(
        "hard_negatives", config, negatives_filename(args.split, args.depth)
    )
    if path.exists() and not args.force:
        print(f"{path} already exists; --force to re-mine.")
        return 0

    device = select_device(args.device)
    print(f"mining with {config.encoder.name} ({config.name})")
    print(f"data revision: {inputs.revision}")
    print(describe_device(device))
    print(
        f"{len(inputs.records)} {args.split} records against "
        f"{len(inputs.index_records)} indexed documents, keeping the top "
        f"{args.depth}"
    )

    started = time.perf_counter()
    per_retriever = retrieve(
        inputs.records,
        config,
        inputs.vocabulary,
        inputs.index_records,
        store,
        device=device,
        name_qualifiers=inputs.name_qualifiers,
    )
    fused = combine(per_retriever, config)
    elapsed = time.perf_counter() - started

    mined = {
        result.record_id: list(result.codes[: args.depth]) for result in fused
    }
    written = save_hard_negatives(store, config, args.split, args.depth, mined)

    print(f"\nmined in {elapsed:.1f}s ({elapsed / max(len(mined), 1):.3f}s/record)")
    print(render(mined, inputs.records, args.depth))
    print(f"\nwrote {written}")
    return 0


def render(mined, records, depth: int) -> str:
    """How much of what was mined is usable, which is not all of it.

    A record whose ranking is mostly right yields few negatives, and that is the
    good case; the number is here so that "four hard negatives per pair" is a
    measured claim rather than a setting.
    """
    gold = {record.id: set(record.subjects) for record in records}
    wrong = [
        sum(1 for code in codes if code not in gold.get(record_id, ()))
        for record_id, codes in mined.items()
    ]
    exhausted = sum(1 for count in wrong if count < 4)
    return "\n".join(
        [
            "",
            "### What the mining yielded\n",
            "| records mined | mean wrong codes in the top "
            f"{depth} | fewer than 4 |",
            "|---:|---:|---:|",
            f"| {len(mined):,} | {sum(wrong) / max(len(wrong), 1):.1f} | "
            f"{exhausted:,} |",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
