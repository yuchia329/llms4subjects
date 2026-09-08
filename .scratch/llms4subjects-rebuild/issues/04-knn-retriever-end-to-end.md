# 04 — Nearest-neighbour retriever, end to end, one encoder

**What to build:**

The first tracer bullet: a complete path from a record to 50 ranked GND codes to a dev score. It also
establishes the pipeline seam that every later stage hides behind.

Retrieve the nearest training records for a document and harvest their gold subjects as candidates. This
is the mechanism the winning leaderboard system used, so the number this ticket produces is a meaningful
reference point rather than a warm-up.

The seam is a single call from records plus configuration to ranked candidate lists. Indexing, retrievers,
fusion, reranking and adjudication all live behind it, reachable only through configuration — so tests
assert contract invariants that survive every later model swap.

One invariant deserves special attention. An earlier draft in this repository built its candidate label
universe from the evaluation split's own gold subjects, ranking only among known-correct answers. Any
number from that construction is meaningless. A test must make that class of mistake impossible.

**Blocked by:** 02, 03

**Status:** done — 343a58b

- [x] A single call takes records plus configuration and returns ranked candidate lists
- [x] Exactly 50 codes are returned per record, in non-increasing score order, with no duplicates
- [x] Every returned code exists in the tib-core vocabulary regardless of index composition
- [x] Perturbing a record's own gold labels leaves its candidate set unchanged
- [x] Document embeddings are cached keyed by encoder and input revision, so a re-run does not recompute them
- [x] Index size is configurable independently of any training-set size
- [x] A dev micro Recall@10 number is recorded, with the band breakdown, in the results document
- [x] Disabling all retrievers yields empty results rather than an error
