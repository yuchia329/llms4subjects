# 07 — Rank fusion and the per-retriever ablation table

**What to build:**

Stage one becomes complete here: three retrievers combine into one candidate list of 100 per record,
fused by reciprocal rank so no retriever's score scale can dominate by accident, with weights tuned on
dev.

Each surviving candidate carries provenance — which retrievers proposed it and at what rank — because the
project's central claim is about attribution, and attribution is impossible if the fused list forgets
where its entries came from.

The deliverable is as much the ablation table as the pipeline: every retriever alone, every pair, and all
three, each with band-level recall. That table is what tells you whether the pipeline's complexity earns
its keep, and it is the first result no published system on this benchmark reports.

**Blocked by:** 05, 06

**Status:** done — the fusion stage, the ablation harness and the dev-tuned weights.
Measured in a clean worktree at the previous commit, because a concurrent agent was
landing ticket 08's bilingual label text in the same checkout and it moves the two
label-reading retrievers; the numbers here are therefore the German-only rendering
every other row in docs/results.md was taken at.

- [x] Candidate generation emits the top 100 per record with scores and per-retriever provenance
- [x] Fusion weights are tuned on dev and recorded as committed configuration — `knn` 1.5, `dense` 1.0, `lexical` 1.0, `rrf_k` 60 in configs/rung1.yaml
- [x] An ablation table covers each retriever alone, each pair, and all three, with band-level recall for each
- [x] Recall at 100 is reported as the ceiling every later stage is measured against — micro 0.6242 fused
- [x] Removing any single retriever changes the output, verified by test
- [x] Fused recall is compared against the best single retriever, so fusion's contribution is a number — +0.0942 micro R@10 over kNN alone
