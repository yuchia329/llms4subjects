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

**Status:** ready-for-agent

- [ ] Candidate generation emits the top 100 per record with scores and per-retriever provenance
- [ ] Fusion weights are tuned on dev and recorded as committed configuration
- [ ] An ablation table covers each retriever alone, each pair, and all three, with band-level recall for each
- [ ] Recall at 100 is reported as the ceiling every later stage is measured against
- [ ] Removing any single retriever changes the output, verified by test
- [ ] Fused recall is compared against the best single retriever, so fusion's contribution is a number
