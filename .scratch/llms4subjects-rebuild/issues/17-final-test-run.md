# 17 — Final test run, opened once

**What to build:**

The test set is opened for the first and only time. Every configuration decision has already been made on
dev; this ticket runs the chosen configuration and records what it scores.

Report against the published leaderboard at the same aggregation the organizers used, and alongside it the
record-level micro figures, since the two differ materially — eleven scoring cells holding 0.4% of test
records carry 55% of the official aggregate, and the test set includes French, Spanish, Czech, Turkish,
Dutch and Japanese records that each form their own cell.

Two caveats are reported rather than buried. Scores are given with and without the 163 test records whose
title and abstract exactly duplicate a training record under a different identifier, because a
neighbour-based retriever is unusually strong on precisely those. And the current-model adjudication run
appears as a separate labelled row, never merged into the headline.

Success is a narrow, honest gap to the top four, with the band breakdown explaining where the score comes
from. It is not a leaderboard win.

**Blocked by:** 13, 15, 16

**Status:** ready-for-agent

- [ ] The chosen configuration is fixed before the test set is read, and that configuration is recorded
- [ ] Test scores are produced by the official scorer on submission-format output, not only by the local evaluator
- [ ] Results are reported at official-macro and record-micro aggregation, side by side
- [ ] Scores are reported with and without the 163 duplicate test records
- [ ] The full band breakdown on test is reported, including zero-shot
- [ ] The published leaderboard rows appear in the same table for direct comparison
- [ ] The current-model adjudication result is a separate labelled row
- [ ] The test set is read exactly once; any later change requires a recorded justification
