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

**Status:** ready-for-agent

- [ ] The results document contains every rung's tables, each with band breakdowns
- [ ] The reference row from the previous classifier appears alongside the pipeline results
- [ ] The leaderboard comparison is presented at the organizers' own aggregation
- [ ] The model cutoff discipline and the single-use test discipline are both stated
- [ ] The zero-shot finding is presented as the headline, with the 812-label figure and its derivation
- [ ] The README carries a short version with the headline number
- [ ] An Artifact page is published and its link recorded
- [ ] Every figure in the writeup is reproducible from a committed configuration and a recorded command
