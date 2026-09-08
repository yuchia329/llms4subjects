# 14 — Baseline: the previous classifier re-run on clean data

**What to build:**

The argument that a dense output layer cannot serve this problem is currently made from label statistics
alone. The project's earlier numbers were computed on a contaminated split and are unusable, so there is no
trustworthy measurement of the approach being rejected.

Re-run the previous classifier once on the clean splits and score it through the new evaluator with the
frozen bands. The expected result is the most vivid plot in the writeup: recall degrading across head,
torso and tail, and exactly zero on the zero-shot band — not as an argument, but as a measurement, since
a closed-vocabulary classifier cannot emit a label absent from its output layer.

Training runs on the `nlp2` GPU host (`ssh nlp2`), since a 14,607-way output layer over 32,043 documents is
not a laptop workload. Rebuild the dataset on the host with the standard command rather than copying it,
then pull the checkpoint and its predictions back so scoring happens locally through the shared evaluator —
the numbers must come from the same code path as every pipeline result.

This ticket is independent of the pipeline and can run in parallel with it. The graph component of the
original design is not revived; it modelled a hierarchy the vocabulary does not contain.

**Blocked by:** 02, 03

**Status:** done

- [x] The previous classifier trains on the clean tib-core training split and scores on dev — the test row is
      deferred to ticket 17, which opens the gold test set once; the path exists and is gated behind
      `--open-test-set` rather than being left to be written later
- [x] Results come through the shared evaluator with frozen bands, making them directly comparable to pipeline results
- [x] Zero-shot band recall is confirmed to be exactly zero, and the reason is stated
- [x] Training configuration and wall-clock cost are recorded
- [x] The graph component is explicitly excluded, with the reason recorded
- [x] The results row is added to the results document as the reference point
- [x] Training runs on `nlp2` and its wall-clock cost on that hardware is recorded
- [x] Predictions are pulled back and scored locally through the shared evaluator, not scored on the host

**Measured (dev, 5,354 records):** micro R@10 0.0667, official macro R@10 0.1623. By band, micro R@10:
head 0.4311, torso 0.0000, tail 0.0000, zero 0.0000. Torso and tail being zero as well was not predicted
and is the sharper finding: the trained head emits only 78 distinct codes across 267,700 ranked slots, 26
of them in any top-10, and 97.4% of slots go to head-band labels. 15 epochs, batch 64, lr 2e-5, 1.25 h on
one A100 80GB.
