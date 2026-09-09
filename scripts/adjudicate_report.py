"""What an LLM is worth on the records the retrievers were least sure of.

    python scripts/adjudicate_report.py configs/rung1-adjudicate.yaml --dry-run
    python scripts/adjudicate_report.py configs/rung1-adjudicate.yaml --sample 300
    python scripts/adjudicate_report.py configs/rung1-adjudicate.yaml

One pass, and it reports the stage twice: on the routed subset alone, which is
the only place adjudication can change anything, and on the whole split, which
is where it has to be paid for. A stage applied to a fifth of the records can
look strong on that fifth and be invisible on the split, and both numbers belong
in the table for the same reason the band breakdown does.

`--dry-run` routes, builds every prompt and calls nothing: it prints who would
be asked, what one prompt looks like and what the pass would cost, so the shape
of the run can be checked before an API key is spent on it. It is also the only
mode that runs without a key.

Responses are cached under the `adjudicated` artifact stage, keyed by the model,
the prompt revision and every knob that changes a prompt, and written through as
each one arrives — so an interrupted run keeps what it bought and a re-score
costs nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms4subjects.artifacts import (  # noqa: E402
    ArtifactStore,
    load_responses,
    log_rejections,
    read_rejections,
)
from llms4subjects.config import ExperimentConfig, IndexConfig, load_experiment  # noqa: E402
from llms4subjects.contracts import CandidateList, Record  # noqa: E402
from llms4subjects.corpus import frequency_bands  # noqa: E402
from llms4subjects.hardware import describe_device, select_device  # noqa: E402
from llms4subjects.models import MODEL_CUTOFF  # noqa: E402
from llms4subjects.pipeline import combine, label_texts, retrieve  # noqa: E402
from llms4subjects.stages import adjudicator, indexes  # noqa: E402
from llms4subjects.stages.evaluator import (  # noqa: E402
    BANDS,
    EvaluationReport,
    evaluate,
)
from rerank_report import achievable_precision  # noqa: E402
from run_experiment import (  # noqa: E402
    FORBIDDEN_SPLIT,
    INPUT_ERRORS,
    load_inputs,
)

TABLE_KS = (5, 10, 25, 50)

# Model selection is micro Recall@10 on dev, as everywhere else in this project.
SELECTION_K = 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="a config with adjudication.enabled true")
    parser.add_argument("--split", default="core_dev", help="the split to predict")
    parser.add_argument(
        "--sample",
        type=int,
        help="score a stratified sample of N records rather than the whole split",
    )
    parser.add_argument("--artifacts", default="artifacts", help="artifact root")
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    parser.add_argument(
        "--route-fraction",
        type=float,
        help="override adjudication.route_fraction for a screening run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "route and build the prompts, call nothing, and print what the pass "
            "would cost; the only mode that needs no API key"
        ),
    )
    args = parser.parse_args(argv)

    if args.split == FORBIDDEN_SPLIT:
        print(f"{FORBIDDEN_SPLIT} is the gold test split; see ticket 17.")
        return 1

    config = load_experiment(args.config)
    if args.route_fraction is not None:
        # The artifact key covers the whole adjudication section, so an
        # overridden fraction caches beside the config's rather than on top of
        # it — and a smaller routed set is a subset of the larger one's
        # responses only by luck, never by construction.
        config = dataclasses.replace(
            config,
            adjudication=dataclasses.replace(
                config.adjudication, route_fraction=args.route_fraction
            ),
        )
    if not config.adjudication.enabled:
        print(
            f"{args.config} has adjudication.enabled false; this harness "
            "measures what adjudication changes, so it needs the config that "
            "turns it on"
        )
        return 1
    if config.reranker.enabled or config.group_prior.enabled:
        # Both stages sit between fusion and this one, and this harness fuses
        # itself. Applying them here would be a second implementation of each;
        # not applying them would print every row below under the name of a
        # pipeline that never ran.
        print(
            f"{args.config} enables the reranker or the group prior, which "
            "this harness does not apply: it fuses and adjudicates. Use "
            "scripts/run_experiment.py for the whole pipeline, or the config "
            "with those stages off."
        )
        return 1

    resolved = adjudicator.resolve(config.adjudication)
    if not args.dry_run:
        # Before the retrieval pass rather than after it: a missing key should
        # cost a sentence, not an index. `--dry-run` calls nothing and needs no
        # key, which is what makes it the mode that checks a run's shape.
        try:
            adjudicator.check_credentials(config.adjudication)
        except adjudicator.MissingApiKey as error:
            print(f"{error}\nOr check the run's shape first with --dry-run.")
            return 1

    try:
        inputs = load_inputs(config, args.split)
    except INPUT_ERRORS as error:
        # `KeyError` is an unknown --split, and quotes its argument; the
        # rest already read as sentences.
        print(error.args[0] if isinstance(error, KeyError) else error)
        return 1

    records = _sampled(inputs.records, args.sample)
    device = select_device(args.device)
    print(f"experiment:    {config.name}  ({args.config})")
    print(f"data revision: {inputs.revision}")
    print(
        f"adjudicator:   {config.adjudication.model} via {resolved.provider}, "
        f"released {resolved.release.created} "
        f"({'APPENDIX, outside' if resolved.appendix else 'inside'} the "
        f"{MODEL_CUTOFF} cutoff)"
    )
    print(
        f"routing:       the least-confident "
        f"{config.adjudication.route_fraction:.0%} of records, top "
        f"{config.adjudication.candidates} offered, "
        f"{config.adjudication.select} chosen, prompt "
        f"{config.adjudication.prompt_revision}"
    )
    print(describe_device(device))
    print(f"predicting {len(records)} {args.split} records\n")

    store = ArtifactStore(args.artifacts, data_revision=inputs.revision)

    started = time.perf_counter()
    fused = combine(
        retrieve(
            records,
            config,
            inputs.vocabulary,
            inputs.index_records,
            store,
            device=device,
            name_qualifiers=inputs.name_qualifiers,
        ),
        config,
    )
    print(f"retrieved and fused in {time.perf_counter() - started:.1f}s")

    texts = label_texts(
        config, inputs.vocabulary, inputs.name_qualifiers, _translations(config)
    )
    routed = set(adjudicator.route(fused, config.adjudication))
    lists = [result for result in fused if result.record_id in routed]
    by_id = {record.id: record for record in records}
    subset = [by_id[result.record_id] for result in lists]
    _routing(fused, lists, config)

    if args.dry_run:
        # The "before" half of the report, which costs nothing and is what the
        # stage will be measured against: what the routed records score now,
        # and the ceiling a reordering of their candidates cannot pass.
        _headline(records, _Scorer(records, config.fusion.candidates).score(fused))
        print()
        _headline(
            subset,
            _Scorer(subset, config.fusion.candidates).score(lists),
            where="the routed subset",
        )
        _dry_run(subset, lists, texts, config)
        return 0

    keyed_as = adjudicator.pinned(config)
    responses = load_responses(store, keyed_as)
    cached = len(set(responses) & routed)
    started = time.perf_counter()
    results = adjudicator.adjudicate(
        subset,
        lists,
        texts,
        config.adjudication,
        model=adjudicator.load(config.adjudication),
        responses=responses,
    )
    seconds = time.perf_counter() - started
    log_rejections(store, keyed_as, results)
    after = adjudicator.apply(fused, results, config.adjudication)

    print(
        f"\nadjudicated {len(results)} records in {seconds:.1f}s: "
        f"{len(results) - cached} asked, {cached} replayed from the cache"
    )

    scorer = _Scorer(records, config.fusion.candidates)
    _headline(records, scorer.score(fused), scorer.score(after))

    subset_scorer = _Scorer(subset, config.fusion.candidates)
    print()
    _headline(
        subset,
        subset_scorer.score(lists),
        subset_scorer.score([result for result in after if result.record_id in routed]),
        where="the routed subset",
    )
    _bands(
        subset_scorer.score(lists),
        subset_scorer.score([result for result in after if result.record_id in routed]),
    )
    _rejections(results, store, keyed_as)
    return 0


def _sampled(records: Sequence[Record], size: int | None) -> list[Record]:
    """A stratified subsample of the split, or all of it.

    The sampler is the index selector, as in `scripts/rerank_report.py`: the dev
    CSV is grouped, so a prefix would make every language row a report on one
    language.
    """
    if not size or size >= len(records):
        return list(records)
    return indexes.select_documents(
        records, IndexConfig(size=size, stratify=True, seed=42)
    )


def _translations(config: ExperimentConfig) -> Mapping[str, str]:
    if not config.label_text.bilingual:
        return {}
    from llms4subjects.corpus import load_label_translations

    return load_label_translations()


# --- What the routing did ---------------------------------------------------


def _routing(
    fused: Sequence[CandidateList],
    routed: Sequence[CandidateList],
    config: ExperimentConfig,
) -> None:
    """Who was routed, and at what confidence. The fraction is a budget, so it
    is reported rather than assumed."""
    from llms4subjects.stages.reranker import confidence

    scores = sorted(confidence(result) for result in fused)
    threshold = (
        max(confidence(result) for result in routed) if routed else float("nan")
    )
    print(
        f"\nrouted {len(routed)} of {len(fused)} records "
        f"({len(routed) / max(len(fused), 1):.1%}, asked for "
        f"{config.adjudication.route_fraction:.0%}) at confidence "
        f"<= {threshold:.4f}"
    )
    print(
        f"confidence over the split: min {scores[0]:.4f}, median "
        f"{scores[len(scores) // 2]:.4f}, max {scores[-1]:.4f}"
    )


def _dry_run(
    records: Sequence[Record],
    lists: Sequence[CandidateList],
    texts: Mapping[str, str],
    config: ExperimentConfig,
) -> None:
    """What the pass would send, without sending it.

    Prompt length is the cost of this stage and the one thing a config can get
    badly wrong without failing, so it is measured here rather than discovered
    on an invoice.
    """
    prompts = [
        adjudicator.prompt(
            record, result.candidates[: config.adjudication.candidates], texts, config.adjudication
        )
        for record, result in zip(records, lists)
    ]
    if not prompts:
        print("\nnothing routed, so nothing would be sent.")
        return

    lengths = np.array([len(prompt) for prompt in prompts])
    print(
        f"\n{len(prompts)} prompts would be sent: {lengths.sum():,} characters "
        f"total, {lengths.mean():.0f} mean, {lengths.max():,} longest "
        f"(roughly {lengths.sum() / 4:,.0f} tokens in, at most "
        f"{len(prompts) * config.adjudication.max_output_tokens:,} out)"
    )
    print(f"\n--- one prompt, for {records[0].id} " + "-" * 40)
    print(prompts[0])
    print("-" * 72)


# --- Scoring ----------------------------------------------------------------


class _Scorer:
    """Fixed gold, bands and cells, so a row is one call over one ranking."""

    def __init__(self, records: Sequence[Record], ceiling: int):
        self.records = records
        self.gold = {record.id: record.subjects for record in records}
        self._cells = {record.id: (record.type, record.lang) for record in records}
        self._bands = frequency_bands()
        self.ks = TABLE_KS + (ceiling,)

    def score(self, candidates: Sequence[CandidateList]) -> EvaluationReport:
        return evaluate(
            gold=self.gold,
            predictions={result.record_id: result.codes for result in candidates},
            bands=self._bands,
            cells=self._cells,
            ks=self.ks,
        )


def _headline(
    records: Sequence[Record],
    before: EvaluationReport,
    after: EvaluationReport | None = None,
    where: str = "the whole split",
) -> None:
    """The table, with the adjudicated row where there is one.

    `after` is absent on a dry run, which still reports the row the stage will
    be measured against and the ceiling it works inside — both are properties of
    the candidates and cost nothing to know before the model is called.
    """
    gold = float(np.mean([len(record.subjects) for record in records]))
    rows = (("fused", before),) if after is None else (
        ("fused", before), ("adjudicated", after)
    )
    print(f"### Adjudication, before and after — {where} ({len(records)} records)\n")
    print(f"{gold:.2f} gold labels per record.\n")

    print("| | " + " | ".join(f"P@{k}" for k in TABLE_KS) + " | " +
          " | ".join(f"R@{k}" for k in TABLE_KS) + " |")
    print("|---" + "|---:" * (2 * len(TABLE_KS)) + "|")
    for name, report in rows:
        cells = [f"{report.micro.precision(k):.4f}" for k in TABLE_KS]
        cells += [f"{report.micro.recall(k):.4f}" for k in TABLE_KS]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print(
        "| achievable | "
        + " | ".join(f"{achievable_precision(records, k):.4f}" for k in TABLE_KS)
        + " | "
        + " | ".join("1.0000" for _ in TABLE_KS)
        + " |"
    )

    ceiling = before.ks[-1]
    print(
        f"\nthe candidate ceiling this reordering works inside: micro "
        f"R@{ceiling} {before.micro.recall(ceiling):.4f} — the stage cannot add "
        "a code, so that is what a perfect adjudicator would score at any k"
    )
    print("\nofficial macro-over-cells, the leaderboard's aggregation:")
    for name, report in rows:
        print(
            f"  {name}: P@5 {report.official_macro.precision(5):.4f}, "
            f"R@{SELECTION_K} {report.official_macro.recall(SELECTION_K):.4f}"
        )


def _bands(before: EvaluationReport, after: EvaluationReport) -> None:
    """Where the reordering lands, by how often a label was trained on."""
    print(f"\n### The routed subset by frequency band (micro R@{SELECTION_K})\n")
    print("| band | assignments | fused R@10 | adjudicated R@10 | difference |")
    print("|---" + "|---:" * 4 + "|")
    for band in BANDS:
        if band not in after.by_band:
            continue
        first, second = before.by_band[band], after.by_band[band]
        print(
            f"| {band} | {second.assignments} | "
            f"{first.recall(SELECTION_K):.4f} | {second.recall(SELECTION_K):.4f} | "
            f"{second.recall(SELECTION_K) - first.recall(SELECTION_K):+.4f} |"
        )


def _rejections(results, store: ArtifactStore, config: ExperimentConfig) -> None:
    """The constraint, as a number. Zero is the result this stage wants."""
    rejected = [result for result in results if result.rejected]
    print(
        f"\n### Constraint violations\n\n{len(rejected)} of {len(results)} "
        f"responses rejected ({len(rejected) / max(len(results), 1):.1%}); "
        "each kept the ranking it arrived with."
    )
    for result in rejected[:5]:
        print(f"  {result.record_id}: {result.reason}")
    if rejected:
        print(f"\nlogged to {store.path('adjudicated', config, 'rejections.jsonl')}")
        print(f"({len(read_rejections(store, config))} across every run of this config)")


if __name__ == "__main__":
    raise SystemExit(main())
