"""The per-retriever ablation table, and the dev tuning of the fusion weights.

    python scripts/ablate_retrievers.py configs/rung1.yaml
    python scripts/ablate_retrievers.py configs/rung1.yaml --tune-weights
    python scripts/ablate_retrievers.py configs/rung1.yaml --limit 500

Every retriever alone, every pair, and all three, each scored through the same
evaluator every other row in docs/results.md comes from. The table is the
deliverable of ticket 07 as much as the fusion is: it is what says whether the
pipeline's complexity earns its keep, and no published system on this benchmark
reports it.

The retrievers run **once** and every row fuses a subset of that one pass, which
is what makes seven rows affordable — a subset is a fusion over candidate lists
that are already in memory, so the seven rows cost one index build and one pass
over the label vocabulary between them. `pipeline.retrieve` and
`pipeline.combine` are the two halves of `pipeline.predict`, so no row here
comes from a code path the pipeline itself does not use.

`--tune-weights` sweeps the fusion weights and `rrf_k` on the same single pass
and prints the config block for the best of them. Reciprocal rank fusion is
invariant to a global scaling of the weights, so one retriever's weight is held
at 1.0 and the others are swept against it; the row where a retriever's weight
is 0 is not the same as the row where it is disabled — a weight of 0 still
contributes provenance and still costs its retrieval — and both are reported.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms4subjects.artifacts import ArtifactStore  # noqa: E402
from llms4subjects.config import ExperimentConfig, load_experiment  # noqa: E402
from llms4subjects.contracts import CandidateList, Record  # noqa: E402
from llms4subjects.corpus import MissingDataset, frequency_bands  # noqa: E402
from llms4subjects.hardware import describe_device, select_device  # noqa: E402
from llms4subjects.pipeline import combine, retrieve  # noqa: E402
from llms4subjects.stages.evaluator import BANDS, EvaluationReport, evaluate  # noqa: E402
from run_experiment import FORBIDDEN_SPLIT, load_inputs  # noqa: E402

# The k values the table reports. 10 is what model selection reads, 50 is the
# submission, and the ceiling is added from the config's `fusion.candidates`.
TABLE_KS = (5, 10, 25, 50)

# What `--tune-weights` sweeps. Coarse on purpose: the weights are a property of
# three retrievers whose relative strengths move with every encoder swap, so a
# finer grid would be fitting dev noise that the next rung invalidates anyway.
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
RRF_K_GRID = (10, 30, 60, 100)

# Model selection is micro Recall@10 on dev (docs/results.md), and the weights
# are selected on the same figure as everything else so that a tuned pipeline
# and an untuned one are comparable.
SELECTION_K = 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="a config with every retriever enabled")
    parser.add_argument("--split", default="core_dev", help="the split to predict")
    parser.add_argument("--limit", type=int, help="score only the first N records")
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    parser.add_argument(
        "--tune-weights",
        action="store_true",
        help="sweep the fusion weights and rrf_k, and print the tuned config block",
    )
    args = parser.parse_args(argv)

    if args.split == FORBIDDEN_SPLIT:
        print(f"{FORBIDDEN_SPLIT} is the gold test split; see ticket 17.")
        return 1

    config = load_experiment(args.config)
    enabled = [
        name for name, settings in sorted(config.retrievers.items()) if settings.enabled
    ]
    if len(enabled) < 2:
        print(
            f"{args.config} enables {len(enabled)} retriever(s); the ablation "
            "table needs the config with all of them on"
        )
        return 1

    # This table is `retrieve` followed by `combine`, which is candidate
    # generation and nothing after it. A config that also enables a stage
    # `predict` applies to the fused list would have every row printed under
    # its name with that stage never run — the silently ignored flag this
    # project refuses elsewhere.
    beyond = [
        name
        for name in ("group_prior", "reranker", "adjudication")
        if getattr(config, name).enabled
    ]
    if beyond:
        print(
            f"{args.config} enables {', '.join(beyond)}, which this table does "
            "not apply: it fuses retriever subsets and stops. Use the config "
            "with those off, or the ablation script for the stage itself."
        )
        return 1

    try:
        inputs = load_inputs(config, args.split, args.limit)
    except MissingDataset as error:
        print(error)
        return 1
    except KeyError as error:
        print(error.args[0] if error.args else error)
        return 1

    records = inputs.records
    device = select_device(args.device)
    print(f"experiment:    {config.name}  ({args.config})")
    print(f"data revision: {inputs.revision}")
    print(describe_device(device))
    print(f"ablating {', '.join(enabled)} over {len(records)} {args.split} records\n")

    store = ArtifactStore(args.artifacts, data_revision=inputs.revision)
    started = time.perf_counter()
    per_retriever = retrieve(
        records,
        config,
        inputs.vocabulary,
        inputs.index_records,
        store,
        device=device,
        name_qualifiers=inputs.name_qualifiers,
    )
    print(f"retrieved in {time.perf_counter() - started:.1f}s\n")

    scorer = _Scorer(records, config.fusion.candidates)

    if args.tune_weights:
        _tune(per_retriever, config, scorer)
        print()

    _table(per_retriever, config, scorer)
    return 0


class _Scorer:
    """Fixed gold, bands and cells, so a row is one call over one ranking."""

    def __init__(self, records: Sequence[Record], ceiling: int):
        self._gold = {record.id: record.subjects for record in records}
        self._cells = {record.id: (record.type, record.lang) for record in records}
        self._bands = frequency_bands()
        self.ceiling = ceiling
        self.ks = TABLE_KS + (ceiling,)

    def score(self, candidates: Sequence[CandidateList]) -> EvaluationReport:
        return evaluate(
            gold=self._gold,
            predictions={result.record_id: result.codes for result in candidates},
            bands=self._bands,
            cells=self._cells,
            ks=self.ks,
        )


def _subsets(names: Sequence[str]) -> list[tuple[str, ...]]:
    """Every retriever alone, then every pair, then all three."""
    return [
        subset
        for size in range(1, len(names) + 1)
        for subset in itertools.combinations(names, size)
    ]


def _fuse(
    per_retriever: Mapping[str, Sequence[CandidateList]],
    subset: Sequence[str],
    config: ExperimentConfig,
) -> list[CandidateList]:
    return combine({name: per_retriever[name] for name in subset}, config)


def _table(
    per_retriever: Mapping[str, Sequence[CandidateList]],
    config: ExperimentConfig,
    scorer: _Scorer,
) -> None:
    """The ablation table, as markdown, ready to paste into docs/results.md."""
    reports = {
        subset: scorer.score(_fuse(per_retriever, subset, config))
        for subset in _subsets(sorted(per_retriever))
    }
    ceiling = scorer.ceiling

    print("### Retriever ablation (micro recall)\n")
    print(
        "| retrievers | "
        + " | ".join(f"R@{k}" for k in TABLE_KS)
        + f" | R@{ceiling} | official R@{SELECTION_K} |"
    )
    print("|---" + "|---:" * (len(TABLE_KS) + 2) + "|")
    for subset, report in reports.items():
        cells = " | ".join(f"{report.micro.recall(k):.4f}" for k in TABLE_KS)
        print(
            f"| {' + '.join(subset)} | {cells} | {report.micro.recall(ceiling):.4f} "
            f"| {report.official_macro.recall(SELECTION_K):.4f} |"
        )

    _band_table(reports, SELECTION_K, "Retriever ablation by frequency band")
    _band_table(reports, ceiling, "The candidate ceiling, by frequency band")
    _fusion_contribution(reports, scorer)


def _band_table(
    reports: Mapping[tuple[str, ...], EvaluationReport], k: int, title: str
) -> None:
    """One row per subset, one column per frozen band, at one k."""
    print(f"\n### {title} (micro R@{k})\n")
    print("| retrievers | " + " | ".join(BANDS) + " |")
    print("|---" + "|---:" * len(BANDS) + "|")
    for subset, report in reports.items():
        cells = " | ".join(
            f"{report.by_band[band].recall(k):.4f}" if band in report.by_band else "—"
            for band in BANDS
        )
        print(f"| {' + '.join(subset)} | {cells} |")


def _fusion_contribution(
    reports: Mapping[tuple[str, ...], EvaluationReport], scorer: _Scorer
) -> None:
    """What fusion is worth, as a number rather than an assertion."""
    singles = {
        subset[0]: report for subset, report in reports.items() if len(subset) == 1
    }
    everything = max(reports, key=len)
    fused = reports[everything]

    print()
    for k in (SELECTION_K, scorer.ceiling):
        name, best = max(singles.items(), key=lambda item: item[1].micro.recall(k))
        print(
            f"fusion against the best single retriever, micro R@{k}: "
            f"{fused.micro.recall(k):.4f} against {name}'s "
            f"{best.micro.recall(k):.4f} "
            f"({fused.micro.recall(k) - best.micro.recall(k):+.4f})"
        )


def _tune(
    per_retriever: Mapping[str, Sequence[CandidateList]],
    config: ExperimentConfig,
    scorer: _Scorer,
) -> None:
    """Sweep the weights and `rrf_k` on dev, best-first, and print the block."""
    names = sorted(per_retriever)
    swept = names[1:]
    print(
        f"tuning {', '.join(swept)} against {names[0]} held at weight 1.0, over "
        f"{len(WEIGHT_GRID) ** len(swept) * len(RRF_K_GRID)} combinations, "
        f"selecting on micro R@{SELECTION_K}\n"
    )

    started = time.perf_counter()
    rows: list[tuple[float, float, dict[str, float], int]] = []
    for rrf_k in RRF_K_GRID:
        for values in itertools.product(WEIGHT_GRID, repeat=len(swept)):
            weights = {names[0]: 1.0, **dict(zip(swept, values))}
            tuned = _with_fusion(config, weights, rrf_k)
            report = scorer.score(_fuse(per_retriever, names, tuned))
            rows.append(
                (
                    report.micro.recall(SELECTION_K),
                    report.micro.recall(scorer.ceiling),
                    weights,
                    rrf_k,
                )
            )

    rows.sort(key=lambda row: (-row[0], row[3], sorted(row[2].items())))
    print(f"swept in {time.perf_counter() - started:.1f}s. Best ten:\n")
    print(
        "| "
        + " | ".join(f"{name} weight" for name in names)
        + f" | rrf_k | micro R@{SELECTION_K} | micro R@{scorer.ceiling} |"
    )
    print("|---" * len(names) + "|---:" * 3 + "|")
    for selected, ceiling, weights, rrf_k in rows[:10]:
        print(
            "| "
            + " | ".join(f"{weights[name]:.2f}" for name in names)
            + f" | {rrf_k} | {selected:.4f} | {ceiling:.4f} |"
        )

    _, _, best_weights, best_rrf_k = rows[0]
    print("\nthe tuned block, for the config:\n")
    print("retrievers:")
    for name in names:
        print(f"  {name}:\n    weight: {best_weights[name]}")
    print(f"fusion:\n  candidates: {config.fusion.candidates}\n  rrf_k: {best_rrf_k}")


def _with_fusion(
    config: ExperimentConfig, weights: Mapping[str, float], rrf_k: int
) -> ExperimentConfig:
    import dataclasses

    return dataclasses.replace(
        config,
        retrievers={
            name: dataclasses.replace(settings, weight=weights[name])
            if name in weights
            else settings
            for name, settings in config.retrievers.items()
        },
        fusion=dataclasses.replace(config.fusion, rrf_k=rrf_k),
    )


if __name__ == "__main__":
    raise SystemExit(main())
