# 11 — Subject-area group prior over the 66 classification groups

**What to build:**

The vocabulary has no hierarchy — the shared-task release contains no broader or narrower relations at
all — so the plan's original idea of propagating predictions to ancestor terms describes a structure that
does not exist. What does exist is a flat grouping of every subject into one of 66 classification groups,
and it carries real signal: 76.7% of multi-label records confine all their gold labels to at most two
groups, and 30.5% to a single group.

This is also the one place in the project where a dense classification head is the right tool, since there
are 66 classes with roughly 485 training documents each — the opposite of the situation that made the
original 14,607-way head fail.

Predict a distribution over groups and use it to boost candidates in likely groups rather than to filter
out unlikely ones, so the 23.3% of records whose labels span three or more groups are not permanently lost.

**Blocked by:** 01, 07

**Status:** ready-for-agent

- [ ] A classifier predicts a distribution over the 66 classification groups for any document
- [ ] Group predictions boost candidate scores and never remove candidates from the list
- [ ] Group classifier accuracy is reported, including how often the true groups fall within the top two predicted
- [ ] The prior is an ablation flag, and its contribution to dev Recall@10 is reported with and without
- [ ] The band breakdown is reported with the prior enabled, to confirm it does not help the head at the tail's expense
