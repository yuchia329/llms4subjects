# 06 — Lexical label retriever

**What to build:**

German subject headings are compounds that frequently appear verbatim in German titles, and dense
embeddings blur exactly those matches. Measured on training data, a label's own name or one of its
synonyms appears verbatim in the record text for 53.7% of German gold assignments but only 23.0% of
English ones — so this retriever is expected to be strongly language-asymmetric, and reporting that
asymmetry is part of the deliverable.

Where the chosen encoder emits sparse term weights natively, use them, since that yields two retrievers
from one forward pass. Otherwise fall back to a classical lexical index over label strings. Either way
the interface is the one the other retrievers implement.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] Lexical matching over label strings returns scored candidates through the shared retriever interface
- [ ] Where the encoder provides sparse term weights natively, they are used rather than a separate index
- [ ] Standalone recall is recorded, split by document language
- [ ] The German and English recall gap is reported explicitly in the results document
- [ ] The retriever is toggleable by configuration without touching other stages
