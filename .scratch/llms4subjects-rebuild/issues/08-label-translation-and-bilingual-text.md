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

**Status:** done

- [x] Every vocabulary preferred name has a cached English translation
- [x] The translation cache is a committed artifact, independent of the model that produced it, and regenerating it is idempotent
- [x] Label text renders bilingually, and German-only rendering remains available as an ablation
- [x] The bilingual versus German-only comparison is reported separately for German and English documents
- [x] Translation adds no per-run cost to indexing or evaluation

**Result:** split, and the split is the finding. Translation lifts the English
slice of both retrievers that read label text — lexical +0.0377 micro R@10 and
+0.1274 at k=50, dense +0.0080 — but the dense tower's German slice falls
0.0304, because its English goes *inside* the one vector per label while
lexical's goes beside it as one more surface string. The larger move is the
gain, so `bilingual` stays `true` in the committed rungs; what the pair is worth
fused is ticket 07's row, since recall does not add across retrievers. The
language gap closes 73% on dense and 30% on lexical. Full numbers in
docs/results.md, "rung1 — translated label names".

Follow-up the numbers argue for, not taken here: make `bilingual` per-retriever
so the tower can read German while the lexical index reads both. It needs its
own row against rung 2 rather than a fourth variant of rung 1.
