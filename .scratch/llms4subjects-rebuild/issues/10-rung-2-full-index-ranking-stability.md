# 10 — Rung 2: full official index, and whether the ranking held

**What to build:**

The same four encoders at the full official training index of 32,043 documents. This rung answers a
question that decides whether the cheap-screening strategy was sound: does the encoder ranking change when
the index grows fourfold?

If the ranking is stable, screening generalises and the fine-tuning choice rests on evidence. If it flips,
that is a finding in its own right and the fine-tuning shortlist has to be reconsidered before any GPU time
is spent.

Only index size moves between rungs one and two. No training enters until rung three, so any change here is
attributable to index size alone.

**Blocked by:** 09

**Status:** ready-for-agent

- [ ] All four encoders run at the full 32,043-document index
- [ ] Encoder ranking at both index sizes is presented side by side
- [ ] Whether the ranking held is stated explicitly, with the rank correlation between rungs
- [ ] The two encoders shortlisted for fine-tuning are named, with the evidence for the choice
- [ ] Band breakdowns are reported at both index sizes so tail behaviour is comparable across scale
- [ ] The rung runs on Apple Silicon without CUDA and without touching `nlp2`
