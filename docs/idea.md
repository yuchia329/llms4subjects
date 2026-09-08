# Rebuilding the project: retrieve-then-rank instead of a classifier head

Written 2026-09-07, after the dataset was rebuilt from the official
SemEval-2025 Task 5 release. All figures below come from the clean
`tib-core-subjects` splits in `TIBKAT_dataset/`; the numbers quoted in earlier
notes came from the contaminated CSVs recorded in [`../legacy/README.md`](../legacy/README.md).

## Where the first attempt went wrong

The original design fine-tuned `bert-base-multilingual-cased` / `xlm-roberta-base`
with a dense output layer over the training label set. It improved over its own
baseline, so the approach looked sound. The label statistics say otherwise.

| measurement | value |
|---|---|
| labels in `core_train.csv` | 14,607 |
| label assignments | 78,037 |
| labels occurring exactly once | 6,584 (45.1%) |
| labels occurring 10 times or fewer | 13,178 (90.2%) |
| coverage of the 100 most frequent labels | 19.7% of assignments |
| coverage of the 1,000 most frequent labels | 52.9% of assignments |
| GND `tib-core` vocabulary | 79,427 subjects |
| share of that vocabulary seen in training | 18.4% |

Two consequences follow, and neither is fixable by training longer.

**The head cannot learn the tail.** 45% of labels carry a single example. A softmax
or per-label BCE output for such a label sees one positive against ~32k negatives.
It learns to predict zero. Most of the output layer's parameters are dead weight.

**The head cannot represent labels it never saw.** Training covers 18.4% of the
GND `tib-core` vocabulary. On the gold test set:

- 992 of 5,189 test labels (19.1%) never appear in training
- 3,095 of 5,189 test labels (59.6%) have five or fewer training examples
- 783 of 4,910 test documents (16.0%) carry at least one unseen label

A closed-vocabulary classifier scores exactly zero on those, by construction, no
matter how good the encoder is. It also cannot handle a GND authority record added
next month, which is the actual operating condition in a library.

## What replaces it

Subjects are not opaque class indices. Every GND code carries text: `Name`,
`Alternate Name`, `Related Subjects`, `Classification Name`, `Definition`. A model
that reads label text can rank a label it has never been trained on, because the
label's name is its own supervision. That turns extreme multi-label classification
into retrieval.

### Stage 1 — candidate generation (optimise recall)

Three cheap retrievers, fused. Target R@100 above 0.85 before any reranking.

1. **Document-to-document kNN.** Embed `title + abstract`, retrieve the 50 nearest
   training records, take the union of their subject sets as candidates. Librarians
   already tag similar books similarly; this exploits that directly and needs no
   training at all.
2. **Document-to-label dense matching.** Embed each GND subject's concatenated
   metadata, score against the document embedding. This is the only component that
   can reach the 992 unseen test labels.
3. **BM25 over label strings.** German subject headings are compounds that often
   appear verbatim in German titles. Lexical match catches what embeddings blur.

Fuse with reciprocal rank fusion. Weight per language if DE and EN behave
differently — the `lang` column in the CSVs makes that measurable.

### Stage 2 — reranking

Cross-encoder over the top ~100 candidates only. It sees the document and one label
description jointly, so it can resolve the near-duplicates that stage 1 confuses.
Training cost is bounded because it only ever scores retrieved candidates, and the
hard negatives come free from stage 1's mistakes.

### Stage 3 — LLM adjudication, optional

For documents where the reranker is unconfident, give an LLM the abstract and the
top 30 candidates and ask it to select. **Constrain the output to the candidate
IDs.** A generative model asked for GND codes free-form will invent codes that look
plausible and do not exist. The LLM picks from a list; it never authors an
identifier.

## Where the training budget goes

One contrastive fine-tune of the bi-encoder from stage 1, with in-batch negatives
plus hard negatives mined from the untrained retriever. LoRA on a multilingual base
(`bge-m3`, `multilingual-e5-large`, or `gte-multilingual-base`). Hours on one GPU,
not days, and it improves both the document and label towers at once. This single
run should beat everything the classifier head produced.

The cross-encoder is the second priority. Skip it entirely if stages 1 and 3 already
clear the target.

Note what is *not* in the budget: no 14,607-way output layer, so no parameters spent
on labels with one example, and no retraining when the GND vocabulary grows.

## Exploit the hierarchy

GND records carry `Related Subjects` and classification groupings. Propagate each
prediction to its broader terms and evaluate at several granularities. A wrong leaf
under the right parent still narrows the librarian's search, which is worth partial
credit even though the official scorer gives none.

## Measure the right thing

The workflow is suggest-and-confirm, not autonomous tagging. Exact-set micro-F1 is
the wrong target.

- Use Recall@10 / Recall@20 and Precision@5 — the official scorer
  (`official_eval/llms4subjects-evaluation.py`) already reports Precision@k,
  Recall@k and F1@k split by record type and language.
- Break every number down by label frequency band: head (>100 training examples),
  torso (10–100), tail (<10), unseen (0). A single average hides that the tail is
  90% of the vocabulary. This is the breakdown that would have exposed the first
  design immediately.
- Report an abstention rate. A system that says "I am unsure, route to a human" on
  10% of records and is accurate on the rest is more useful than one that guesses
  everywhere.

## Order of work

The sequencing matters more than the model choice.

1. Stage 1 with off-the-shelf embeddings, no training. One day. This measures the
   recall ceiling and tells you whether reranking is even the bottleneck.
2. Add BM25 and RRF fusion. Re-measure.
3. Contrastive fine-tune of the bi-encoder. Largest gain per unit of compute.
4. Cross-encoder reranker over the top 100.
5. LLM adjudication for low-confidence documents only.
6. Calibrate the per-label decision threshold and the abstention rule.

Do not skip step 1. It is nearly free and it sets the ceiling every later stage is
measured against.

## Caveat on the test set

163 test records share an exact title and abstract with a training record under a
different TIBKAT id — parallel German and English editions of the same work. This is
present in the official data, not an artefact of the rebuild. Worth reporting scores
both with and without them, since a kNN retriever will look unusually strong on
exactly those records.
