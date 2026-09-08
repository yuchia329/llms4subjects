# 12 — Off-the-shelf cross-encoder reranking

**What to build:**

Reranking is where precision is won: the fused candidate list is ordered by retrieval similarity, which
is a poor proxy for whether a subject genuinely applies. A cross-encoder reads the document and one label's
text jointly and can separate the near-duplicates that stage one confuses.

Run an off-the-shelf multilingual reranker first, on the same screening logic as the encoders — no GPU
budget is committed to a component that may already work untrained. This ticket runs entirely on Apple
Silicon.

Fine-tune it only if the untrained version demonstrably helps. That fine-tune would be a fourth run on the
`nlp2` GPU host (`ssh nlp2`) beyond the three already planned, so the recommendation this ticket produces
is a budget decision, not just a technical one.

This ticket also produces the per-record confidence measure that the adjudication and abstention tickets
both need, so its output contract matters beyond its own score.

When reading precision numbers here, note the ceiling: at 2.40 gold labels per test record, a perfect
system scores 0.48 at k=5 and 0.24 at k=10. The leaderboard leader's 0.25 at k=5 is roughly 52% of
achievable, not 25%.

**Blocked by:** 07

**Status:** done — the stage, the screening harness, and a recommendation
against the fine-tune. The off-the-shelf reranker loses 0.06 micro R@10 when its
order *replaces* the fused one (wrecking the head band, gaining the zero-shot
band) and wins 0.034 when `reranker.mix: fuse` combines the two orders instead,
which is the setting committed to `configs/rung1-rerank-base.yaml`. Measured on
a 300-record stratified dev sample, because a full dev pass is 3.9 hours on the
laptop. Its own relevance scores are anti-calibrated (Pearson −0.05, most
confident decile 66.7% no-hit), so the confidence measure ticket 13 routes on is
`confidence` read from the fused ranking (+0.41). `reranker.enabled` stays false
in the ladder.

- [x] The top 100 candidates are reranked to a final ranked 50 through the shared pipeline seam
- [x] The reranker model's release date precedes 2025-01-31 — both screened models are in
      `reference/model_releases.json` with pinned pre-cutoff revisions, and `reranker.resolve`
      refuses anything the registry does not vouch for
- [x] Dev Precision@5, Precision@10 and Recall@10 are reported before and after reranking — on a
      300-record stratified sample, with the wall-clock reason stated
- [x] Reported precision is accompanied by the achievable ceiling so the numbers are not misread
- [x] A per-record confidence measure is emitted and its calibration against correctness is shown —
      for all three rankings, which is how the anti-calibration was found
- [x] A recommendation on whether to fine-tune the reranker is recorded, with the evidence behind it
- [x] Reranking is toggleable, and the pipeline remains correct with it disabled
- [x] The off-the-shelf reranking pass completes on Apple Silicon without CUDA — 30,000 pairs in
      789s on MPS; length-sorted batching is what makes it affordable
- [x] If a reranker fine-tune is recommended, the recommendation states the expected `nlp2` cost and
      what it buys — it is not recommended, and the cost and the bounded upside are stated anyway
