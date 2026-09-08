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

**Status:** done — **the ranking flipped.** `bge-m3` takes first at 0.5337 dev micro
R@10 against `gte-multilingual-base`'s 0.5298, reversing rung 1's order (Spearman
ρ 0.800, Kendall τ-b 0.667). The shortlist's *membership* survived and its order
did not: the same two encoders lead at both index sizes by 0.0336 over the third,
and swap with each other on 0.0039. Both go to rung 3, which is what carrying two
forward was for. Every difference is the neighbour retriever's — the dense and
lexical columns are identical to four decimals at both index sizes.

- [x] All four encoders run at the full 32,043-document index — `configs/rung2.yaml` plus three siblings, each one flag from its rung-1 counterpart; ~85 minutes of M4 Pro, one matrix computed per encoder since the label tower is encoder-keyed and shared with rung 1
- [x] Encoder ranking at both index sizes is presented side by side — `scripts/compare_rungs.py`, reading the two committed screens in `reference/screens/`; `screen_encoders.py --json` is what writes them
- [x] Whether the ranking held is stated explicitly, with the rank correlation between rungs — stated as a flip, in words, above the tables; ρ 0.800 and τ-b 0.667 implemented in the comparator (tau-b, average ranks, `None` below three encoders where a coefficient carries no information)
- [x] The two encoders shortlisted for fine-tuning are named, with the evidence for the choice — `bge-m3` then `gte-multilingual-base`; `gte` leads at R@50 and R@100, so it is the better input to a reranker that reads 100 candidates while `bge-m3` orders its top ten better, and neither margin is worth defending against the other
- [x] Band breakdowns are reported at both index sizes so tail behaviour is comparable across scale — tail +0.120 to +0.136, head flat, and the zero-shot band *down* 0.005–0.015 while zero-shot recall at 100 holds to four decimals for three of the four encoders (and moves 0.0009 for `multilingual-e5-large`), which makes it displacement by kNN candidates rather than loss
- [x] The rung runs on Apple Silicon without CUDA and without touching `nlp2` — MPS throughout, `requirements/mac.txt`

**Also landed:** the two screens are committed reference artifacts, and
`tests/test_compare_rungs.py` fails if a later config edit makes them
incomparable. A disabled stage is persisted as its off switch alone, so a screen
stays comparable as later tickets add fields to config sections that never ran.

**One loose end:** the fusion weights are still the ones tuned for
`multilingual-e5-base` at rung 1, held fixed across both rungs on purpose, so
0.5337 is a floor. Retuning for the shortlisted encoder is rung 3's first step.
