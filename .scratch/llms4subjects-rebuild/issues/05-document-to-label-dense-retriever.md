# 05 — Document-to-label dense retriever

**What to build:**

The component the entire design exists to provide. Nearest-neighbour retrieval can only ever return
labels some training record already carries; this retriever scores all 79,427 vocabulary entries directly,
so a label with no training examples is reachable.

Build the label-text renderer — subject area, preferred name, synonyms, field-marked so the encoder sees
structure rather than a concatenated word salad — and index the full vocabulary. German only at this stage;
translation is a later ticket.

Two fields are deliberately excluded, both measured rather than assumed. The definition field holds
cataloguing instructions to librarians, not descriptions of meaning, and covers only 18.1% of the
vocabulary; it stays available as an ablation flag. Related terms are excluded from any graph-expansion
role, because only 7.3% of gold labels have a related-neighbour that is also gold on the same record.

Success for this ticket is not the aggregate score. It is the first non-zero recall in the zero-shot band.

**Blocked by:** 01, 04

**Status:** done — 1f2167f

- [x] All 79,427 vocabulary entries are indexed and scoreable for any document
- [x] Label text is field-marked, with subject area, preferred name and synonyms distinguishable
- [x] The definition field is excluded by default and available as an ablation flag
- [x] Qualifier rendering follows the flag established in ticket 01
- [x] Label embeddings are cached keyed by encoder and label-text revision
- [x] Recall in the zero-shot band is non-zero and recorded, alongside the other three bands
- [x] The retriever's standalone recall is recorded separately from any fused result
