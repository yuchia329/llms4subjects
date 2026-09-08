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

**Status:** ready-for-agent

- [ ] The top 100 candidates are reranked to a final ranked 50 through the shared pipeline seam
- [ ] The reranker model's release date precedes 2025-01-31
- [ ] Dev Precision@5, Precision@10 and Recall@10 are reported before and after reranking
- [ ] Reported precision is accompanied by the achievable ceiling so the numbers are not misread
- [ ] A per-record confidence measure is emitted and its calibration against correctness is shown
- [ ] A recommendation on whether to fine-tune the reranker is recorded, with the evidence behind it
- [ ] Reranking is toggleable, and the pipeline remains correct with it disabled
- [ ] The off-the-shelf reranking pass completes on Apple Silicon without CUDA
- [ ] If a reranker fine-tune is recommended, the recommendation states the expected `nlp2` cost and what it buys
