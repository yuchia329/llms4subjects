# 13 — LLM adjudication for low-confidence records

**What to build:**

The final ranking stage, applied only where it can pay: roughly the least-confident fifth of records, as
identified by the reranker's confidence measure. The model sees the document and the top 30 candidates and
selects among them.

The output constraint is the point of this ticket, not a detail. A generative model asked for GND codes
freely will produce identifiers that look entirely plausible and do not exist. The model chooses from a
list; it never authors an identifier. Any response containing a code outside the offered set is rejected
and logged, so constraint violations are detected rather than silently scored.

The headline configuration uses a model released before 2025-01-31, keeping the leaderboard comparison
honest. A second, clearly-labelled run with a current model produces the appendix row quantifying what
eighteen months of model progress adds to an otherwise identical pipeline.

**Blocked by:** 12

**Status:** built — the two scored rows await an API key; see docs/results.md

- [x] Only records below the confidence threshold are routed, and the routed fraction is configurable and recorded
- [x] The model selects from the offered candidates and cannot introduce a new identifier
- [x] Responses containing out-of-set identifiers are rejected and logged, with a test covering that path
- [x] Responses are cached by record and prompt revision, so re-scoring incurs no further cost
- [ ] Dev metrics are reported for the routed subset alone and for the full split — the *before* halves are
      measured and in docs/results.md, along with the ceiling a perfect adjudicator could reach (+0.046 micro
      R@10 split-wide); the *after* halves need an API credential this environment does not have
- [ ] The headline run uses a pre-2025-01-31 model; a current-model run is recorded separately as an appendix row
      — the mechanism is built and enforced (`claude-3-5-sonnet-20241022` is registered and pinned;
      `adjudication.appendix` is required for a post-cutoff model and refused on a pre-cutoff one), but neither
      run has been made
- [x] The pipeline produces valid output with adjudication disabled
