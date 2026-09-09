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

**Status:** done

- [x] The index covers all 70,588 available training documents, with split alignment re-verified before use — 70,588 documents available (70,633 rows, 45 duplicate filings merged), **70,579 indexed**: the alignment holds exactly for `core_test` (0 of 4,910) and fails for dev (9 `core_dev` records), so those nine are named in `reference/split_alignment.json` and dropped from every index
- [x] Both shortlisted encoders are contrastively fine-tuned with in-batch and mined hard negatives — 78,037 pairs, InfoNCE over in-batch plus 2 mined negatives, with false negatives masked in three places
- [x] Fine-tuning runs as a parameter-efficient adaptation and each run completes within a few hours on `nlp2` — LoRA, 7.1M and 2.9M trainable parameters, 196.7 and 86.6 minutes
- [x] Hard negatives are mined locally from existing rung-2 indexes and passed to the host as an artifact — 7.5 minutes total on warm rung-2 vectors, `artifacts/hard_negatives/<key>/`
- [x] Adapters are pulled back and applied locally; indexing and evaluation run on Apple Silicon — deltas merged into the base weights at load, so a trained encoder is the same code path as an untrained one
- [x] Adapters are saved as versioned artifacts so indexing and evaluation re-run without retraining — `training.json` records base model, pinned revision, dataset revision, the negatives' artifact key, every hyperparameter, losses, host and wall clock; a mismatch raises before any weights are fetched
- [x] Whether the untrained ranking held is stated, with the trained and untrained scores side by side — **it did not**: `gte` 0.5376 → 0.6044 overtakes `bge-m3` 0.5411 → 0.5909. The official macro metric ranks them the other way, and both are reported
- [x] The points added by fine-tuning are reported per band, not only in aggregate — monotonic in label frequency: head +0.10/+0.13, torso +0.08/+0.09, tail +0.02/+0.03, **zero-shot −0.056/−0.023**
- [x] Band migration is reported: how many zero-shot labels became seen, and how many remain unreachable — 173 of 1,053 dev-gold zero-shot labels reached (all into the tail band), **880 unreachable, carrying 7.1% of gold assignments**; the ticket's test-side estimate was 7.2%
- [x] Frozen bands are used unchanged, verified by test — `tests/test_frequency_bands.py::test_the_rung_3_corpus_does_not_touch_the_frozen_bands` over the actual rung-3 corpus, and `tests/test_rung3_report.py` pins migration to the frozen boundaries

**Outcome:** rung 3 is `gte-multilingual-base` fine-tuned at 0.6044 dev micro
R@10, +0.0707 over rung 2. Of that, the corpus is worth +0.0079 and the
fine-tune +0.0668 — index size is spent, and the GPU is where the remaining
points are. The fusion weights are still `multilingual-e5-base`'s from rung 1,
deliberately not retuned inside this measurement, so 0.6044 is a floor.

**Not done here, and why:** retuning the fusion weights for the fine-tuned tower.
Rung 2 called it rung 3's first step, but the tower nearly doubled in strength
during this ticket, so tuning it against the untrained weights would have been
measuring the wrong thing, and tuning it inside the trained/untrained comparison
would have been reported as what fine-tuning bought. It is the cheapest
experiment left and belongs in a config of its own.
