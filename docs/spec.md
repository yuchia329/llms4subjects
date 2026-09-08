# Spec: retrieval-based GND subject tagging for TIBKAT records

Status: ready to build. Supersedes the approach in [idea.md](idea.md), which remains as
the reasoning record. Derived from a design interview on 2026-09-07; every figure quoted
here was measured against the clean splits in `TIBKAT_dataset/`.

## Problem Statement

A librarian receiving a new technical record has to assign GND subject headings from a
controlled vocabulary of 79,427 terms. The distribution of those terms is extremely
long-tailed: in 32,043 training records, 45.1% of the observed labels occur exactly once,
90.2% occur ten times or fewer, and the training data touches only 18.4% of the vocabulary
at all. Manual assignment does not scale, and the obvious automation — a text classifier
with one output per label — cannot work here:

- A label with one training example is one positive against ~32,000 negatives. The model
  learns to predict zero.
- 19.1% of the labels in the official gold test set never appear in training. A
  closed-vocabulary classifier scores zero on them by construction.
- The vocabulary grows. A new GND authority record is unreachable until the model is
  retrained.

An earlier attempt in this repository fine-tuned `bert-base-multilingual-cased` with a
dense output layer over the training label set. It improved over its own baseline, so the
approach appeared sound, but it was evaluated on a contaminated split (see
[../legacy/README.md](../legacy/README.md)) and its reported numbers are unusable. The
project therefore has no trustworthy measurement of either the old approach or a
replacement.

Separately, the practitioner needs to know whether this class of approach is *competitive*.
Without a comparison to the SemEval-2025 Task 5 leaderboard, an improvement over one's own
baseline says nothing about whether the method is right.

## Solution

Treat subject assignment as **retrieval over a label vocabulary**, not classification into
a label set. Every GND code carries text — a preferred name, synonyms, and a subject-area
label — so a model that reads label text can rank a label it has never seen in training.

The system produces, for each record, a ranked list of exactly 50 GND codes, assembled in
three stages:

1. **Candidate generation** — three cheap retrievers fused: document-to-document nearest
   neighbours (harvesting the neighbours' gold subjects), document-to-label dense matching,
   and lexical matching over label strings. Optimised for recall at 100.
2. **Reranking** — a cross-encoder scores the document against each candidate's label text,
   reordering the top 100 down to the final 50.
3. **Adjudication** — for the least-confident records only, an LLM selects from the top 30
   candidates, with output constrained to the candidate identifiers so it cannot invent a
   GND code.

Alongside the score, the system reports **where the score comes from**: every metric broken
down by how often each label appeared in training (head / torso / tail / zero-shot). No
published system on this benchmark reports that breakdown, and it is the deliverable that
distinguishes this project from a reimplementation.

The end state is a number comparable to the published leaderboard, an analysis no
leaderboard entry contains, and a writeup that can be linked from a résumé.

## User Stories

### Comparability and honesty of the result

1. As a practitioner, I want the system evaluated with the organizers' own scorer, so that my numbers are directly comparable to the published leaderboard rather than to a metric I invented.
2. As a practitioner, I want every component restricted to models released on or before 2025-01-31, so that my comparison against teams who competed in January 2025 is fair rather than flattered by eighteen months of model progress.
3. As a practitioner, I want the official gold test set opened exactly once at the end of the project, so that my headline number is not the product of repeated peeking.
4. As a practitioner, I want all iteration decisions made on the official dev split, so that model selection cannot contaminate the final measurement.
5. As a practitioner, I want the label-frequency bands fixed against one reference corpus, so that adding training data later cannot silently reclassify which labels count as "tail".
6. As a practitioner, I want my earlier classifier re-run on the clean splits, so that "the classifier approach fails on the tail" is a measurement in my results table rather than an assertion in my prose.
7. As a practitioner, I want the 163 test records that duplicate a training record's title and abstract identified, and scores reported both with and without them, so that a nearest-neighbour retriever's advantage on them is visible rather than hidden.
8. As a practitioner, I want to know when a candidate set was built using any information derived from gold labels, so that I never repeat the oracle-candidate mistake in the earlier `.train_knn_e5.py` draft.
9. As a reader of the writeup, I want to see which retriever contributed each part of the score, so that I can judge whether the pipeline's complexity is justified.
10. As a reader of the writeup, I want to know how much of the final score came from fine-tuning versus from off-the-shelf models, so that I can tell how much of the result I could reproduce without a GPU.

### Candidate generation

11. As a practitioner, I want to embed documents and labels with any of four interchangeable encoders, so that I can compare encoders without rewriting the pipeline.
12. As a practitioner, I want document embeddings and label embeddings cached to disk keyed by encoder and input revision, so that re-running an experiment does not recompute them.
13. As a practitioner, I want a document-to-document retriever that returns the gold subjects of the nearest training records, so that I reproduce the mechanism the winning leaderboard system used.
14. As a practitioner, I want a document-to-label retriever that scores every one of the 79,427 vocabulary entries, so that labels absent from training are reachable at all.
15. As a practitioner, I want lexical matching over label strings, so that German compound headings appearing verbatim in German titles are captured — measured at a 53.7% verbatim hit rate on German records.
16. As a practitioner, I want the three retrievers fused by reciprocal rank fusion with weights tuned on dev, so that no retriever's score scale dominates by accident.
17. As a practitioner, I want each retriever's individual recall reported before fusion, so that I can tell whether fusion helped and which component to improve.
18. As a practitioner, I want candidate generation to emit the top 100 candidates with scores and per-retriever provenance, so that downstream stages and analyses can attribute a final prediction to its source.
19. As a practitioner, I want predictions restricted to the tib-core vocabulary, so that a label harvested from an all-subjects neighbour cannot be emitted as an out-of-vocabulary prediction.
20. As a practitioner, I want to vary the size of the retrieval index independently of the size of any training set, so that "more neighbours" and "more training signal" remain separately attributable.

### Label representation

21. As a practitioner, I want each label rendered as field-marked text — subject area, preferred name, synonyms — so that the encoder sees structure rather than a concatenated word salad.
22. As a practitioner, I want the GND `Definition` field excluded by default, because on inspection it contains cataloguing instructions to librarians rather than descriptions of meaning, and is present on only 18.1% of labels.
23. As a practitioner, I want `Definition` inclusion available as an ablation flag, so that excluding it is a measured decision rather than an assumption.
24. As a practitioner, I want an English translation of every German label name generated once and cached, so that the 58.7% of gold label assignments belonging to English documents are matched against text in their own language.
25. As a practitioner, I want the bilingual label text to be an ablation flag against German-only text, so that the value of translation is a number in the results.
26. As a practitioner, I want the translation cache to be a committed artifact independent of the translating model, so that the pipeline is reproducible without re-running translation.
27. As a practitioner, I want `skos:related` expansion left out of the build, having measured that only 7.3% of gold labels have a related-neighbour that is also gold on the same record.

### Subject-area prior

28. As a practitioner, I want a classifier over the 66 GND classification groups, because this is the one place a dense output layer is appropriate — 66 classes with roughly 485 training documents each.
29. As a practitioner, I want the predicted group distribution used to boost candidates in likely groups rather than to filter out unlikely ones, so that the 23.3% of records whose labels span three or more groups are not permanently lost.
30. As a practitioner, I want the group prior to be an ablation flag, so that its contribution is separable from the retrievers'.

### Reranking and adjudication

31. As a practitioner, I want an off-the-shelf multilingual cross-encoder reranking the top 100 before I consider training one, so that I do not spend GPU budget on a component that may already work untrained.
32. As a practitioner, I want to fine-tune the reranker only if the off-the-shelf version demonstrably helps, so that GPU budget follows evidence.
33. As a practitioner, I want a confidence measure per record after reranking, so that I can route only uncertain records onward.
34. As a practitioner, I want the LLM adjudication stage applied to roughly the least-confident 20% of records, so that API cost stays proportional to benefit.
35. As a practitioner, I want the LLM's output constrained to the identifiers of the candidates it was shown, so that it cannot emit a plausible-looking GND code that does not exist.
36. As a practitioner, I want any LLM response containing an identifier outside the candidate set rejected and logged, so that constraint violations are detected rather than silently scored.
37. As a practitioner, I want LLM responses cached by record and prompt revision, so that re-scoring does not re-bill.
38. As a practitioner, I want the adjudication stage runnable with a pre-deadline model for the headline result and with a current model for an appendix row, so that "what model progress adds to the same pipeline" is one measured line.

### Training

39. As a practitioner, I want to screen all four encoders untrained at two index sizes before training anything, so that I choose what to fine-tune on evidence and at zero GPU cost.
40. As a practitioner, I want to know whether the encoder ranking changes between the small and full index, so that I learn whether cheap screening generalises.
41. As a practitioner, I want to fine-tune the top two encoders rather than only the best, so that a ranking flip caused by fine-tuning is detectable rather than assumed away.
42. As a practitioner, I want contrastive fine-tuning with in-batch negatives plus hard negatives mined from the untrained retriever, so that the model learns from its own confusions.
43. As a practitioner, I want fine-tuning to run as a parameter-efficient adaptation, so that a run fits comfortably in a few hours on the `nlp2` GPU host.
44. As a practitioner, I want the fine-tuned adapter saved as a versioned artifact, so that indexing and evaluation can be re-run without retraining.
45. As a practitioner, I want indexing, retrieval and evaluation to run on Apple Silicon, so that only the two or three training runs need the `nlp2` GPU host.
46. As a practitioner, I want the dataset rebuildable on the GPU host by the same command as locally, so that gitignored data does not have to be copied by hand before a training run.
47. As a practitioner, I want training runs to write their adapters and logs to artifacts I retrieve from the GPU host, so that indexing and evaluation continue locally without re-running training.

### Evaluation and analysis

48. As a practitioner, I want a scoring function that reproduces the official scorer's numbers exactly, so that I can iterate locally without shuffling files into the organizers' directory layout on every run.
49. As a practitioner, I want record-level micro-averaged metrics as the model-selection signal, so that a single-record scoring cell cannot flip which encoder appears to win.
50. As a practitioner, I want the official macro-over-cells `Overall` figure reported next to the micro figure, so that my headline stays comparable to the published table.
51. As a practitioner, I want the divergence between those two aggregations reported as a finding, having measured that eleven scoring cells holding 0.4% of test records carry 55% of the official `Overall` metric.
52. As a practitioner, I want every metric broken down by frozen frequency band, so that a strong aggregate score cannot conceal zero performance on 8.9% of gold assignments.
53. As a practitioner, I want the count of test labels reachable only through label text reported explicitly — measured at 812 labels and 7.2% of gold assignments even when every available document is indexed — so that the limit of document-similarity methods is quantified.
54. As a practitioner, I want band migration reported when the index grows, so that "adding 38,545 documents converts 180 of 992 zero-shot labels into seen ones" is a stated number rather than an invisible effect.
55. As a practitioner, I want metrics also split by document language and record type, so that the cross-lingual gap is tracked rather than averaged away.
56. As a practitioner, I want a dev-only coverage-versus-precision curve, clearly marked as outside the official metric, so that the suggest-and-confirm workflow has one honest result attached to it.
57. As a practitioner, I want every rung's results appended to a results document as it completes, so that the writeup is a byproduct of the work rather than a reconstruction from memory.
58. As a practitioner, I want a submission writer that emits the organizers' exact directory and file layout, so that I can validate against the official scorer end to end.

### Reproducibility and operation

59. As a practitioner, I want each pipeline stage to write a cached artifact keyed by its configuration, so that changing the reranker does not invalidate the index.
60. As a practitioner, I want experiment configuration expressed as data rather than edited into scripts, so that a rung is described by a config file I can commit.
61. As a practitioner, I want large artifacts kept out of version control, so that the repository stays clonable.
62. As a practitioner, I want the dataset rebuildable from the official repository by one command, so that nobody has to trust the CSVs in my repo.
63. As a newcomer to the repository, I want the rejected baseline separated from the current system, so that I can tell which code represents the project's conclusion.
64. As a hiring manager, I want a single page presenting the leaderboard comparison, the band breakdown and the ablations, so that I can assess the work without reading the code.

## Implementation Decisions

### Scope of the benchmark

- Primary track is **tib-core-subjects**. Evaluation is on the official gold test set (4,910 records); all iteration on the official dev split (5,354 records).
- The **all-subjects** train split is usable as additional index and training data. The two tracks were verified split-aligned: all 32,043 tib-core train ids appear in all-subjects train, all 5,354 dev ids in all-subjects dev, and **zero** tib-core test ids appear in all-subjects train or dev. 38,545 all-subjects train records lie outside tib-core train.
- Predictions are always restricted to the 79,427-code tib-core vocabulary, regardless of index composition.
- Model cutoff is **2025-01-31**, the close of the SemEval evaluation window. Applies to encoders, rerankers and the headline LLM. One clearly-labelled appendix row may use a current model in the adjudication stage only.

### Module boundaries

The pipeline is built as an importable package with one module per stage, each consuming
and producing named artifacts:

- **label text builder** — vocabulary entry to field-marked string, with flags for `Definition` inclusion and bilingual rendering. Owns the translation cache.
- **encoder adapter** — uniform interface over the four candidate encoders, hiding their differing pooling, prefix and instruction conventions. Emits dense vectors and, where the model supports it, sparse term weights.
- **index builder** — builds and persists document and label indexes for a given encoder and corpus selection.
- **retrievers** — three implementations behind one interface returning scored candidate lists: document-to-document neighbour label harvesting, document-to-label dense scoring, and lexical label matching.
- **fusion** — reciprocal rank fusion with per-retriever weights, retaining per-retriever provenance for each surviving candidate.
- **group prior** — 66-way classifier over classification groups, exposed as a candidate score adjustment.
- **reranker** — cross-encoder scoring of document against candidate label text.
- **adjudicator** — LLM selection over a candidate list, with hard validation that returned identifiers are a subset of those offered.
- **evaluator** — metrics from gold and predicted label lists, at micro and official-macro aggregations, sliced by band, language and record type.
- **submission writer** — emits the organizers' `<Type>/<lang>/<id>.json` layout with exactly 50 ranked codes.

The rejected classifier lives in a separate `baseline/` area, changed only to read the
clean CSVs. It is a results row, not a dependency.

### Pipeline contract

- Candidate generation emits the **top 100** candidates per record with scores and provenance.
- Reranking reduces to the final **top 50**, matching the submission format.
- Adjudication sees the **top 30** for the routed subset only, and may reorder but never introduce identifiers.
- The final output for every record is always exactly 50 ranked codes. The submission format has no representation for abstention, so abstention is an analysis over confidence scores, never a change to the output contract.

### Label representation

Default rendering is field-marked and bilingual, of the shape:

```
Fachgebiet: Organische Chemie / Organic chemistry
Schlagwort: Polymerisation / Polymerization
Synonyme: Polyreaktion; Kettenpolymerisation
```

- `Definition` excluded by default. Inspection showed it holds cataloguing instructions — for example `gnd:4043744-9` "Ordnung" carries *"Allgemeinbegriff, verknüpfe mit Anwendungsgebiet"* — and it is present on only 18.1% of the vocabulary.
- `Related Subjects` are human-readable German names, not codes, and are excluded from the graph-expansion role; their names may still appear in synonym-adjacent text only if an ablation shows benefit.
- Translation of all 79,427 label names runs once and is cached as a committed artifact, decoupling the pipeline from the translating model.

### The experiment ladder

Three rungs, varying **index size only** until the final rung introduces training:

| rung | index | models | hardware |
|---|---|---|---|
| 1 | 8,000 documents, stratified by record type and language | 4 encoders, off-the-shelf | Apple Silicon |
| 2 | 32,043 documents (official tib-core train) | same 4, off-the-shelf | Apple Silicon |
| 3 | 70,588 documents (all-subjects train) | fine-tune top 2 from rung 2 | `nlp2` GPU host |

Rung 2 answers whether the encoder ranking is stable across index scale. Rung 3 reports
whether the untrained ranking survived fine-tuning, and how many points training added.
Total GPU runs: two fine-tunes plus one baseline re-run.

### Hardware

Two environments, and the split between them is a hard constraint rather than a preference.

- **Local — Apple Silicon (M4 Pro, 48 GB, MPS, no CUDA).** All label-text building, indexing, retrieval, fusion, reranking, evaluation and analysis. Rungs 1 and 2 in full. This is where the loop that runs hundreds of times lives, so it must never require the GPU host.
- **GPU — `nlp2`, reached with `ssh nlp2` (`nlp-gpu-02.be.ucsc.edu`).** Only the three training runs: the two rung-3 contrastive fine-tunes, and the baseline classifier re-run. A reranker fine-tune is a fourth run only if ticket 12 shows it is warranted.

Because the dataset is not tracked in version control, the GPU host rebuilds it with the same
command used locally rather than receiving a copy by hand. Training runs write adapters and logs
as artifacts that are pulled back for local indexing and evaluation, so no evaluation ever depends
on the GPU host being reachable.

### Metrics

- **Model selection** uses record-level micro-averaged Recall@10 on dev.
- **Headline reporting** uses the official aggregation, macro-averaged over the 20 record-type × language cells at k = 5,10,…,50, for comparability with the published table.
- The two are reported side by side, and their divergence is itself a result: eleven cells hold ≤9 records each but carry 55% of the official `Overall`, and the test set includes French, Spanish, Czech, Turkish, Dutch and Japanese records that each form their own cell.
- **Bands are frozen** against tib-core train label counts and never recomputed: head >100 occurrences (65 labels, 16.0% of test assignments), torso 10–100 (1,384 labels, 44.5%), tail 1–9 (2,748 labels, 30.7%), zero-shot 0 (992 labels, 8.9%). Band migration under a larger index is reported separately.
- Test scores are reported with and without the 163 records whose title and abstract exactly duplicate a training record under a different id.

### Reference points

The published tib-core leaderboard, which the results table must sit beside:

| team | P@5 | R@5 | P@10 | R@10 | Avg R@k |
|---|---|---|---|---|---|
| RUC Team | 0.25 | 0.48 | 0.16 | 0.57 | 0.66 |
| Annif | 0.23 | 0.48 | 0.14 | 0.54 | 0.59 |
| LA2I2F | 0.20 | 0.41 | 0.13 | 0.49 | 0.58 |
| DUTIR831 | 0.23 | 0.49 | 0.13 | 0.54 | 0.56 |

Note when reading precision: at 2.40 gold labels per test record, a perfect system scores
P@5 = 0.48 and P@10 = 0.24. RUC's P@5 of 0.25 is roughly 52% of achievable, not 25%.

### Anti-requirements

Encoded from mistakes already made in this repository:

- Candidate sets must never be derived from gold labels. The `.train_knn_e5.py` draft built its label universe from the dev split's own gold subjects, ranking only among known-correct answers; any number from that construction is meaningless.
- Split membership must be carried by record id, not inferred from text. The previous CSVs dropped ids, which is why 142 training records were found inside the gold test set only after the fact.
- Dataset builders must be idempotent. The previous builder appended to its output on every run.
- Structured fields must not be serialised through `str()`. 30% of previous abstracts were stored as stringified Python lists.

## Testing Decisions

There are no tests in the repository today, so this establishes the prior art rather than
following it. A good test here asserts **external behaviour and invariants**, not internal
mechanics: it must not assert that a particular encoder returns a particular vector, nor
that fusion assigns a particular float, because those change with every model swap and
weight retune.

### Seams

Two seams, at the highest points that admit meaningful assertions:

**Seam 1 — `predict(records, config) -> ranked candidate lists`.** The whole pipeline behind
one call. Everything below it — indexing, the three retrievers, fusion, the group prior,
reranking, adjudication — is an implementation detail reachable only through configuration.
Tests run it over a small committed fixture corpus and assert contract invariants:

- the configured candidate count returned per record — 100, of which the submission writer takes the top 50 (ticket 07 moved this line: `predict` returns the whole candidate set with its provenance, because the reranker and the recall ceiling both read past 50) — in non-increasing score order
- no duplicate codes within a record
- every returned code exists in the tib-core vocabulary
- disabling a retriever changes the output; disabling all of them yields empty results rather than a crash
- a label absent from the index but present in the vocabulary can still be returned, which is the property the entire design exists to provide
- no gold label of any input record influences its own candidate set — verified by running with gold labels perturbed and asserting the candidate set is unchanged

**Seam 2 — `evaluate(gold, predictions) -> metrics`.** A pure function over label lists.
Tested by equivalence against the organizers' script: for a fixture of predictions, the
official scorer and this function must agree to floating-point tolerance at every k. This
is the only test that protects the headline number, and it is cheap. Hand-computed cases
cover the edges: a record whose gold set is larger than k, a prediction list with no hits,
a perfect prediction, and a single-record scoring cell.

Band assignment is tested as part of this seam: bands must be read from the frozen
reference artifact, and a test asserts that changing the index does not change any label's
band.

The submission writer is tested only for round-trip: written files re-read by the official
scorer's own reader must reproduce the input lists. Preferring these two seams over
per-module tests is deliberate — the modules are model wrappers whose behaviour is defined
by external weights, so unit-testing them tests the vendors, not this project.

### Fixtures

A committed fixture of ~50 records drawn from dev, spanning both languages and at least
three record types, plus a fixture vocabulary slice containing a label that appears in no
fixture record. Fixtures are small enough to run on CPU in seconds, so the suite stays
usable as a pre-commit check. LLM adjudication is tested against a recorded response,
including one recorded response containing an out-of-set identifier, which must be
rejected.

## Out of Scope

- The **all-subjects track** as an evaluation target. Its train split is used as data; its test set is not scored, and its larger label vocabulary is not modelled.
- **Serving, latency and throughput.** No API, no incremental index updates, no interactive demo. Batch offline scoring only.
- **Abstention as a system behaviour.** The output contract is always 50 codes. Abstention appears once, as a dev-only coverage curve outside the official metric.
- **Hierarchical evaluation and partial credit.** The vocabulary has no hierarchy — zero `skos:broader` triples exist — so broader-term propagation and multi-granularity scoring are dropped rather than deferred.
- **External GND data.** No fetching of broader/narrower relations from the national library or elsewhere; the shared-task vocabulary is the boundary.
- **Qualitative expert evaluation.** The organizers ran subject-specialist assessment; reproducing it requires librarians and is not attempted.
- **Beating the leaderboard.** The target is a narrow, honest gap to the top four, with the analysis as the contribution. Optimisation that would compromise the model cutoff or the single-use test discipline is out of scope even if it would raise the score.
- **Reviving the GCN label-graph component** from the previous design. It modelled a hierarchy that does not exist.

## Further Notes

- The winning system on this track used document-to-document neighbour retrieval, so stage 1 is a reimplementation of a known-good mechanism rather than a speculative bet. The novel content is the attribution analysis, not the architecture.
- The strongest single measurement supporting the design: 812 test labels, carrying 7.2% of gold assignments, remain zero-shot even when all 70,588 available documents are indexed. Document-similarity methods cannot reach them at any scale; only the label-text tower can.
- The cross-lingual asymmetry is where the score lives. English documents carry 58.7% of gold assignments, but a label name appears verbatim in the text for only 23.0% of English assignments against 53.7% of German ones. Translation of label names is the cheapest available intervention.
- Dev is a faithful proxy for test on the dimension that matters most: band profiles agree within 1.2 percentage points on every band.
- The old `requirements.txt` pins CUDA wheels, consistent with the previous work having run on `nlp2`. A second, CUDA-free environment definition is needed for Apple Silicon. Rungs 1–2 and all evaluation must run without CUDA, on the laptop.
- `official_eval/llms4subjects-evaluation.py` sweeps k from 5 to 50 in steps of 5 and writes an Excel workbook with three granularity sheets. It reads gold and predictions from parallel directory trees, which is why the local evaluator exists alongside it.
