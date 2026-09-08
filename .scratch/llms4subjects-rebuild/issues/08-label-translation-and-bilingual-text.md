# 08 — Label name translation and bilingual label text

**What to build:**

The largest measured asymmetry in the dataset: all label text is German, while 58.7% of gold label
assignments belong to English documents. Document-to-label matching is therefore a cross-lingual task
between a long English abstract and a short German phrase — the hardest configuration for a bi-encoder,
and the most likely reason neighbour retrieval outperforms label matching.

Translate all 79,427 preferred names to English once and cache the result as a committed artifact, so the
pipeline reproduces without re-running translation and does not depend on the translating model. Render
labels bilingually and measure the effect on the English half.

The rank-two team on this benchmark used translation for the same reason, so this is a known-productive
intervention rather than a speculative one.

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] Every vocabulary preferred name has a cached English translation
- [ ] The translation cache is a committed artifact, independent of the model that produced it, and regenerating it is idempotent
- [ ] Label text renders bilingually, and German-only rendering remains available as an ablation
- [ ] The bilingual versus German-only comparison is reported separately for German and English documents
- [ ] Translation adds no per-run cost to indexing or evaluation
