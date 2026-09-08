# 15 — Rung 3: all-subjects index and contrastive fine-tuning

**What to build:**

The final rung, and the only one that spends GPU time. Two changes land together: the index grows to all
70,588 available training documents, and the two shortlisted encoders are contrastively fine-tuned with
in-batch negatives plus hard negatives mined from the untrained retriever.

Both fine-tunes run on the `nlp2` GPU host (`ssh nlp2`). Hard-negative mining happens locally first, since
it only needs the rung-2 indexes that already exist; the mined negatives travel to the host as an artifact.
Adapters come back and are applied locally for indexing and evaluation, so no result in this ticket depends
on the host remaining reachable after training finishes.

Using the larger corpus is safe and verified: the two shared-task tracks are split-aligned, so no
tib-core test record appears anywhere in the all-subjects training or dev splits. 38,545 additional
labelled records are available at no contamination risk.

Two reports come out of this ticket. First, whether the untrained encoder ranking survived fine-tuning,
and how many points training added — the answer to how much of the final score requires a GPU at all.
Second, band migration: the extra corpus converts only 180 of the 992 zero-shot test labels into seen
ones, leaving 812 labels carrying 7.2% of gold assignments reachable only through label text at any
corpus size. That number is the project's sharpest finding, because it bounds what document-similarity
methods can ever achieve on this benchmark.

Bands stay frozen against tib-core training counts throughout, so migration is reported as a separate
explicit quantity rather than silently shifting the tables.

**Blocked by:** 10, 11, 12

**Status:** ready-for-agent

- [ ] The index covers all 70,588 available training documents, with split alignment re-verified before use
- [ ] Both shortlisted encoders are contrastively fine-tuned with in-batch and mined hard negatives
- [ ] Fine-tuning runs as a parameter-efficient adaptation and each run completes within a few hours on `nlp2`
- [ ] Hard negatives are mined locally from existing rung-2 indexes and passed to the host as an artifact
- [ ] Adapters are pulled back and applied locally; indexing and evaluation run on Apple Silicon
- [ ] Adapters are saved as versioned artifacts so indexing and evaluation re-run without retraining
- [ ] Whether the untrained ranking held is stated, with the trained and untrained scores side by side
- [ ] The points added by fine-tuning are reported per band, not only in aggregate
- [ ] Band migration is reported: how many zero-shot labels became seen, and how many remain unreachable
- [ ] Frozen bands are used unchanged, verified by test
