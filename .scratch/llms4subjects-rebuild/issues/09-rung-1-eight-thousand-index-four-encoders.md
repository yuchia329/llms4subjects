# 09 — Rung 1: 8,000-document index, four encoders screened

**What to build:**

The first rung of the scaling ladder, and the cheapest decision in the project: which encoder deserves
the GPU budget. All four candidates run off-the-shelf, no training, entirely on the Mac.

Screening untrained is deliberate. An off-the-shelf encoder will not beat a fine-tuned one, but the
ranking among encoders is expected to survive fine-tuning, and screening costs nothing while training all
four at every rung would cost nine to twelve GPU runs. This rung also establishes how much of the final
score is attributable to fine-tuning rather than to the pipeline.

Every candidate model must have been released on or before 2025-01-31, the close of the shared task's
evaluation window, so the eventual leaderboard comparison is not flattered by later model progress.

The index is stratified by record type and language so the subset is not accidentally an English-book
benchmark.

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] All four encoders run through the full stage-one pipeline at an 8,000-document index
- [ ] The index subset is stratified by record type and document language, and the stratification is reproducible
- [ ] Every model's release date is recorded and verified to precede 2025-01-31
- [ ] Encoders are ranked by dev micro Recall@10, with band breakdowns for each
- [ ] Total wall-clock time per encoder is recorded, so later cost claims are grounded
- [ ] The whole rung runs on Apple Silicon without CUDA and without touching `nlp2`
