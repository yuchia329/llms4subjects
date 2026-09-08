"""What the subject-area group prior is worth, on dev, with and without.

    python scripts/ablate_group_prior.py configs/rung1-prior.yaml
    python scripts/ablate_group_prior.py configs/rung1-prior.yaml --limit 500

The prior is an ablation flag (docs/spec.md, story 30), so its contribution has
to be a number rather than an assertion. This runs the retrievers **once**,
fuses them once, and then re-boosts that one fused list at every weight in the
grid — a boost is a re-scoring and a sort, so the whole sweep costs one
retrieval pass. Weight 0.0 is the row where the prior runs and contributes
nothing, and it must reproduce `configs/rung1.yaml` exactly; that equality is
what makes every other row a measurement of the boost alone.

The band breakdown is printed with the prior enabled because the failure mode
of a group prior is a better aggregate paid for by the tail: the head band is
65 broad headings that a group prediction can only help, and the zero-shot band
is the one the design exists to reach.

Reranking and adjudication are switched off for the sweep whatever the config
says, so the rows measure the boost rather than a cross-encoder reordering it.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms4subjects.artifacts import (  # noqa: E402
    ArtifactStore,
    CachedEncoder,
    MissingGroupPrior,
    load_group_prior,
)
from llms4subjects.config import load_experiment  # noqa: E402
from llms4subjects.contracts import CandidateList, Record  # noqa: E402
from llms4subjects.corpus import MissingDataset, frequency_bands  # noqa: E402
from llms4subjects.hardware import describe_device, select_device  # noqa: E402
from llms4subjects.pipeline import combine, retrieve  # noqa: E402
from llms4subjects.stages import encoders, group_prior  # noqa: E402
from llms4subjects.stages.evaluator import BANDS, evaluate  # noqa: E402
from run_experiment import FORBIDDEN_SPLIT, load_inputs  # noqa: E402

# The k values the table reports. 10 is what model selection reads and 50 is
# the submission; the candidate ceiling is not among them because the boost
# reorders the candidate list and cannot change its membership, so recall at
# the ceiling is the same number in every row by construction.
TABLE_KS = (5, 10, 25, 50)

SELECTION_K = 10

# Coarse, and deliberately so: the weight is in units of a record's own score
# range, so the interesting question is its order of magnitude rather than its
# third decimal.
WEIGHT_GRID = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="a config with group_prior.enabled true")
    parser.add_argument("--split", default="core_dev", help="the split to predict")
    parser.add_argument("--limit", type=int, help="score only the first N records")
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    args = parser.parse_args(argv)

    if args.split == FORBIDDEN_SPLIT:
        print(f"{FORBIDDEN_SPLIT} is the gold test split; see ticket 17.")
        return 1

    config = load_experiment(args.config)
    if not config.group_prior.enabled:
        print(
            f"{args.config} has group_prior.enabled false; this table needs the "
            "config the head was fitted for"
        )
        return 1

    # The sweep is over the boost, so nothing downstream of it may reorder the
    # rows. The prior stays on: its key is what the fitted head is filed under.
    config = dataclasses.replace(
        config,
        reranker=dataclasses.replace(config.reranker, enabled=False),
        adjudication=dataclasses.replace(config.adjudication, enabled=False),
    )

    try:
        inputs = load_inputs(config, args.split, args.limit)
    except MissingDataset as error:
        print(error)
        return 1
    except KeyError as error:
        print(error.args[0] if error.args else error)
        return 1

    device = select_device(args.device)
    store = ArtifactStore(args.artifacts, data_revision=inputs.revision)
    print(f"experiment:    {config.name}  ({args.config})")
    print(f"data revision: {inputs.revision}")
    print(describe_device(device))

    try:
        prior = load_group_prior(store, config)
    except MissingGroupPrior as error:
        print(error)
        return 1

    records = inputs.records
    enabled = [
        name for name, settings in sorted(config.retrievers.items()) if settings.enabled
    ]
    print(
        f"sweeping the group prior over {len(records)} {args.split} records, "
        f"on candidates from {', '.join(enabled) or 'no retriever'}\n"
    )

    encoder = encoders.load(config.encoder, device)

    started = time.perf_counter()
    per_retriever = retrieve(
        records,
        config,
        inputs.vocabulary,
        inputs.index_records,
        store,
        device=device,
        encoder=encoder,
        name_qualifiers=inputs.name_qualifiers,
    )
    fused = (
        combine(per_retriever, config)
        if per_retriever
        else [CandidateList(record.id, ()) for record in records]
    )
    print(f"retrieved and fused in {time.perf_counter() - started:.1f}s")

    cached = CachedEncoder(encoder, store, config)
    distributions = prior.distributions(
        cached.encode_documents([record.text for record in records])
    )
    group_of_code = group_prior.group_of_code(inputs.vocabulary.values())

    scorer = _Scorer(records)
    rows = {
        weight: scorer.score(
            group_prior.apply_prior(
                fused,
                distributions,
                group_of_code,
                dataclasses.replace(config.group_prior, weight=weight),
            )
        )
        for weight in WEIGHT_GRID
    }
    _check_identity(fused, rows, scorer)

    _table(rows)
    _band_table(rows, config.group_prior.weight)
    _verdict(rows, config)
    return 0


class _Scorer:
    """Fixed gold, bands and cells, so a row is one call over one ranking."""

    def __init__(self, records: Sequence[Record]):
        self._gold = {record.id: record.subjects for record in records}
        self._cells = {record.id: (record.type, record.lang) for record in records}
        self._bands = frequency_bands()
        self.ks = TABLE_KS

    def score(self, candidates: Sequence[CandidateList]):
        return evaluate(
            gold=self._gold,
            predictions={result.record_id: result.codes for result in candidates},
            bands=self._bands,
            cells=self._cells,
            ks=self.ks,
        )


def _check_identity(fused, rows, scorer) -> None:
    """Weight 0 must be the unboosted ranking, or the sweep is not a sweep."""
    unboosted = scorer.score(fused)
    if rows[0.0].micro.recall(SELECTION_K) != unboosted.micro.recall(SELECTION_K):
        raise AssertionError(
            "a zero-weight boost changed the ranking; the ablation's own "
            "baseline row is not the unboosted one"
        )


def _table(rows) -> None:
    print("\n### The group prior, swept on dev (micro recall)\n")
    print(
        "| weight | "
        + " | ".join(f"R@{k}" for k in TABLE_KS)
        + f" | official R@{SELECTION_K} |"
    )
    print("|---" + "|---:" * (len(TABLE_KS) + 1) + "|")
    for weight, report in rows.items():
        label = f"{weight:g}" + (" (off)" if weight == 0.0 else "")
        cells = " | ".join(f"{report.micro.recall(k):.4f}" for k in TABLE_KS)
        print(
            f"| {label} | {cells} | "
            f"{report.official_macro.recall(SELECTION_K):.4f} |"
        )


def _band_table(rows, configured: float) -> None:
    print(f"\n### By frequency band (micro R@{SELECTION_K})\n")
    print("| weight | " + " | ".join(BANDS) + " |")
    print("|---" + "|---:" * len(BANDS) + "|")
    for weight, report in rows.items():
        label = f"{weight:g}" + (" (off)" if weight == 0.0 else "")
        if weight == configured:
            label += " ←"
        cells = " | ".join(
            f"{report.by_band[band].recall(SELECTION_K):.4f}"
            if band in report.by_band
            else "—"
            for band in BANDS
        )
        print(f"| {label} | {cells} |")


def _verdict(rows, config) -> None:
    off = rows[0.0].micro.recall(SELECTION_K)
    best = max(rows, key=lambda weight: rows[weight].micro.recall(SELECTION_K))
    print(
        f"\nbest weight on micro R@{SELECTION_K}: {best:g} at "
        f"{rows[best].micro.recall(SELECTION_K):.4f} against {off:.4f} "
        f"unboosted ({rows[best].micro.recall(SELECTION_K) - off:+.4f})"
    )
    configured = config.group_prior.weight
    if configured in rows:
        print(
            f"the committed weight is {configured:g}, at "
            f"{rows[configured].micro.recall(SELECTION_K):.4f} "
            f"({rows[configured].micro.recall(SELECTION_K) - off:+.4f})"
        )


if __name__ == "__main__":
    raise SystemExit(main())
