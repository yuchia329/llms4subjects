# 16 — Dev-only coverage and precision curve

**What to build:**

The submission format is always exactly 50 ranked codes, so a system cannot express abstention and the
official metric cannot measure it. But the workflow this project exists to support is suggest-and-confirm:
a librarian reviewing proposals, where a system that declines to guess on uncertain records is more useful
than one that guesses everywhere.

Produce one figure from scores already computed — no extra inference. As the confidence threshold rises,
what fraction of records still receive an answer, and how does precision on those records improve? Label it
unambiguously as outside the official metric, so it can never be mistaken for a leaderboard-comparable
result.

**Blocked by:** 13

**Status:** ready-for-agent

- [ ] A coverage-versus-precision curve is produced from existing dev scores with no additional inference
- [ ] The curve reports both coverage and precision at several thresholds, including full coverage
- [ ] The figure is labelled as outside the official metric wherever it appears
- [ ] The confidence measure driving the curve is the one produced by the reranking stage
- [ ] The system's output contract is unchanged: 50 codes are still always returned
