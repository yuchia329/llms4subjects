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

**Status:** ready-for-agent

- [ ] The previous classifier trains on the clean tib-core training split and scores on dev and test
- [ ] Results come through the shared evaluator with frozen bands, making them directly comparable to pipeline results
- [ ] Zero-shot band recall is confirmed to be exactly zero, and the reason is stated
- [ ] Training configuration and wall-clock cost are recorded
- [ ] The graph component is explicitly excluded, with the reason recorded
- [ ] The results row is added to the results document as the reference point
- [ ] Training runs on `nlp2` and its wall-clock cost on that hardware is recorded
- [ ] Predictions are pulled back and scored locally through the shared evaluator, not scored on the host
