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

**Status:** done — `gte-multilingual-base` wins at 0.4724 dev micro R@10 against
`multilingual-e5-base`'s 0.4149, with `bge-m3` 0.0022 behind it and
`multilingual-e5-large` below its own base model. Rungs 2 and 3 carry the top two.
The code landed in fae6f00, a concurrent agent's ticket-12 commit that swept the
whole working tree; the review fixes that followed are their own commit.

- [x] All four encoders run through the full stage-one pipeline at an 8,000-document index — `scripts/screen_encoders.py`, which refuses configs differing in anything but the encoder
- [x] The index subset is stratified by record type and document language, and the stratification is reproducible — seeded draw, every one of the 39 cells represented, shares matching the corpus to four decimals; the screen prints the table
- [x] Every model's release date is recorded and verified to precede 2025-01-31 — `reference/model_releases.json`, derived from the hub by `scripts/verify_model_releases.py`, and each model pinned to its newest pre-cutoff commit because both E5 checkpoints had 2026 commits landed on them
- [x] Encoders are ranked by dev micro Recall@10, with band breakdowns for each — plus each retriever alone per encoder, which is what shows the E5 flip is the label tower
- [x] Total wall-clock time per encoder is recorded, so later cost claims are grounded — load and stage one separately, with the cache state per row. The three cold rows were measured while two ticket-12 reranker passes shared the laptop, so they are reported as approximate; a clean `--cold` re-timing on an idle box is the one loose end
- [x] The whole rung runs on Apple Silicon without CUDA and without touching `nlp2` — MPS throughout, `requirements/mac.txt`
