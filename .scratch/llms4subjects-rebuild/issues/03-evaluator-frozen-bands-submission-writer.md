# 03 — Evaluator, frozen frequency bands, and submission writer

**What to build:**

This is the measurement seam, and it comes before any modelling because every later ticket reports
through it. It is also the only test that protects the headline number.

Deliver a pure function from gold and predicted label lists to metrics, at both aggregations the project
needs: record-level micro averages for model selection, and the official macro-over-cells aggregation for
comparability with the published leaderboard. Slice every metric by frequency band, document language and
record type.

Freeze the frequency bands now, as a committed artifact derived from tib-core train label counts, so that
growing the index later cannot silently reclassify which labels count as tail:

    head    more than 100 occurrences      65 labels     16.0% of test assignments
    torso   10 to 100 occurrences       1,384 labels     44.5%
    tail    1 to 9 occurrences          2,748 labels     30.7%
    zero    never seen                    992 labels      8.9%

Also deliver the submission writer that emits the organizers' directory and file layout with exactly 50
ranked codes per record, so the local evaluator can be validated end to end against their script.

Correctness here is established by equivalence, not by hand-picked expectations: for a fixture of
predictions, the local evaluator and the official scorer must agree at every k the official script sweeps.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] Local evaluator agrees with the official scorer to floating-point tolerance at every k from 5 to 50 on a committed fixture
- [ ] Metrics are reported at both micro and official-macro aggregation, side by side
- [ ] Every metric is sliceable by frequency band, document language and record type
- [ ] Band assignments are read from a committed frozen artifact, never recomputed at evaluation time
- [ ] A test asserts that changing the index does not change any label's band
- [ ] Hand-computed edge cases pass: gold set larger than k, zero hits, perfect prediction, single-record cell
- [ ] Submission writer output is re-readable by the official scorer's own reader and round-trips the input lists
- [ ] The divergence between micro and official-macro aggregation is reported, including which cells dominate the official figure
