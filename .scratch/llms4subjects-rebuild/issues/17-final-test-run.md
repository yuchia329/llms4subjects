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

**Status:** done — **0.6299 official R@10, 0.5910 record-micro R@10** on the 4,910-record gold test split,
read once on 2026-09-09 under a configuration committed one commit earlier. Above every published row on
recall, below every one on precision, and the two are the same fact: 34.1% of the split carries one gold
heading and the system scores 0.7367 R@10 there against 0.4163 on records with five or more. The zero-shot
band reaches 0.3547. The two adjudication rows remain owed on an API credential; see docs/results.md.

- [x] The chosen configuration is fixed before the test set is read, and that configuration is recorded —
      `reference/test_plan.json` digests each row's eight pipeline sections and was committed as `d1109ec`,
      before the run; the harness refuses a row the plan does not name and a row whose digest has moved
- [x] Test scores are produced by the official scorer on submission-format output, not only by the local
      evaluator — the run writes both trees and executes `official_eval/llms4subjects-evaluation.py`.
      The local evaluator reproduces its `Overall` row to 1.1e-16 over the 4,882 records it scored;
      that was verified after the fact against the run's own trees, and `score_officially` now
      measures it in-run so a later run prints it
- [x] Results are reported at official-macro and record-micro aggregation, side by side — and the
      divergence, +0.1245 at k=10, is larger than the 0.06 separating first from fourth on the published
      table
- [x] Scores are reported with and without the duplicate test records — **but the count is 134 (indexed
      corpus) and 107 (`core_train`), not 163.** The 163 came from docs/idea.md, written against the
      pre-rebuild CSVs, which kept no record ids and held a seven-row `core_test.csv`, so it was not
      computable there. Every definition tried is tabulated in docs/results.md. The caveat is worth
      0.0017 R@10
- [x] The full band breakdown on test is reported, including zero-shot — head 0.7215, torso 0.6297,
      tail 0.5353, zero 0.3547, with supports within 0.05pp of the frozen shares
- [x] The published leaderboard rows appear in the same table for direct comparison
- [ ] The current-model adjudication result is a separate labelled row — **not done, and blocked twice
      over.** The row exists, is planned, and reports as blocked rather than as a zero: this environment
      has no `ANTHROPIC_API_KEY`, and the appendix row additionally needs an `API_MODELS` entry with
      `appendix=True`, which means asserting a release date in a frozen provenance artifact. Ticket 13's
      debt, unchanged
- [x] The test set is read exactly once; any later change requires a recorded justification —
      `reference/test_run.json` records the read; a second one is refused without `--justify`, which is
      then recorded beside the read it supersedes, and `--fix-plan` is refused once the receipt exists
