# 18 — Writeup and published Artifact page

**What to build:**

Assemble the results into the two deliverables that outlive the code: a results document carrying every
rung's tables, and a single published page suitable for linking from a resume or embedding in a personal
site.

Most of this should already exist, because every earlier ticket appends its numbers as it completes. What
remains is the narrative: the reference row from the previous classifier, the encoder screening and whether
its ranking held, the per-retriever ablation, the fine-tuning delta, the leaderboard comparison, the band
breakdown, and the coverage curve.

Lead with the finding, not the architecture. The pipeline reimplements a mechanism the winning system
already used; what is new is knowing where the score comes from, and that 7.2% of this benchmark's gold
assignments are unreachable by document-similarity methods at any corpus size — including by the system
that won.

**Blocked by:** 14, 17

**Status:** done — the results document now leads with the finding: **812 gold test headings appear on no
document in any available corpus, carrying 7.2% of the split's gold assignments**, derived by
`scripts/zero_shot_bound.py` from the frozen bands, the 70,579-document corpus the run indexed and the
split's own gold. This system reaches 0.323 R@10 on those 851 assignments; a neighbour harvest scores
0.000 on them by construction, and so does every leaderboard system built on document similarity.

- [x] The results document contains every rung's tables, each with band breakdowns — fourteen sections,
      one per experiment, each with the command that produced it, and every scored row broken down over
      the frozen bands; appended as each ticket landed rather than written here. The one section with no
      band table is the coverage curve, which sweeps abstention over a ranking already scored above
- [x] The reference row from the previous classifier appears alongside the pipeline results — in the
      dev table (0.0667 micro R@10 against the pipeline's 0.6044) and in "The finding", where the number
      that matters is its **0.0000** on the zero-shot band at every k
- [x] The leaderboard comparison is presented at the organizers' own aggregation — the four published
      rows and this run's, from the organizers' unmodified script over a submission tree, at their
      macro-over-cells aggregation; the record-micro figure travels beside it and the +0.1245 divergence
      is reported as a result
- [x] The model cutoff discipline and the single-use test discipline are both stated — "The two
      disciplines every number here rests on", in the results document and on the Artifact page
- [x] The zero-shot finding is presented as the headline, with the 812-label figure and its derivation —
      "The finding" is the first section, and the 812 is now **measured** rather than estimated: 992
      zero-shot test headings, 180 reached by the all-subjects corpus, **812 unreachable carrying 851
      assignments (7.2%)**, against dev's 880 of 1,053 at 7.1%
- [x] The README carries a short version with the headline number — "The result, in short", before the
      repository layout
- [x] An Artifact page is published and its link recorded —
      <https://claude.ai/code/artifact/d65b0f93-c161-4c34-8008-1129a4c10199>, source committed at
      `docs/812-headings.html` and linked from both the README and docs/results.md
- [x] Every figure in the writeup is reproducible from a committed configuration and a recorded command —
      the bound was the one figure with no command, and now has one:
      `python scripts/zero_shot_bound.py configs/test.yaml --split core_test --predictions
      artifacts/test/headline/submission --json reference/zero_shot_bound.json`, which reads the frozen
      bands and refuses to run until `reference/test_run.json` says the split was opened

**Outcome:** the writeup leads with the bound rather than with the architecture. The one number that had
been carried as an estimate since ticket 15 — 812 labels, 7.2% — is now derived by a committed script
from committed inputs, and its other half is measured beside it: the pipeline reaches 0.323 R@10 on the
851 assignments the whole leaderboard scores 0.000 on.

**Not done here, and why:** the two adjudication rows are still owed on an API credential (ticket 13's
debt, reaching ticket 17 and now 18 unchanged). They are reported as blocked in docs/results.md rather
than as a zero, and nothing in the writeup depends on them.
