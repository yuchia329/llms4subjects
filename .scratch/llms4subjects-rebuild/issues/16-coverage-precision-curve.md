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

**Status:** done — `scripts/coverage_curve.py`, the curve in docs/results.md

- [x] A coverage-versus-precision curve is produced from existing dev scores with no additional inference —
      the harness replays a cached reranking pass and refuses to run one, so the constraint is enforced rather
      than promised
- [x] The curve reports both coverage and precision at several thresholds, including full coverage — ten
      levels from 100% down to 10%, with the gold assignments given up and the achievable precision beside
      each row
- [x] The figure is labelled as outside the official metric wherever it appears — table heading, plot title,
      the harness's closing note, README and docs/results.md, and a test holds the marker in the renderings
- [x] The confidence measure driving the curve is the one produced by the reranking stage —
      `stages.reranker.confidence`, the same signal ticket 13 routes on, and the row names which ranking it
      was read over
- [x] The system's output contract is unchanged: 50 codes are still always returned — the curve is an
      analysis over confidence and never shortens a ranking, which a test holds
