# Results

One row per experiment, appended as it completes, so the writeup is a byproduct
of the work rather than a reconstruction from memory (docs/spec.md, story 57).

Every number here comes from `scripts/run_experiment.py` — or, for the rejected
baseline, from `python -m baseline.score` — both of which score through
`llms4subjects.stages.evaluator` with the bands frozen in
`reference/frequency_bands.json`; the command that produced a row is quoted with
it. **Model selection is micro Recall@10 on dev**, and the official
macro-over-cells figure is reported beside it because that is what the
leaderboard is quoted on. Every number below is **dev** except the last
section, which is the single test run: the gold split was opened once, on
2026-09-09, under a configuration digested and committed before it was read
(`reference/test_plan.json`, and the receipt in `reference/test_run.json`).

## Dev, by experiment

| experiment | index | retrievers | micro R@10 | official R@10 | micro R@50 |
|---|---|---|---:|---:|---:|
| `rung1-knn` | 8,000 stratified | kNN only | **0.3023** | 0.3948 | 0.4495 |
| `rung1-dense` | none read | dense label tower only | **0.1224** | 0.1260 | 0.2110 |
| `rung1-lexical` | none read | lexical label matching only | **0.1558** | 0.1416 | 0.2532 |
| `rung1` (fused) | 8,000 stratified | all three, RRF | **0.3964** | 0.5340 | 0.5568 |
| `rung1-prior` † | 8,000 stratified | all three + the 66-group prior | **0.4202** | 0.5464 | 0.5840 |
| `rung1-gte-base` ‡ | 8,000 stratified | all three, RRF | **0.4724** | 0.6107 | 0.6481 |
| `rung1-bge-m3` ‡ | 8,000 stratified | all three, RRF | **0.4702** | 0.5981 | 0.6378 |
| `rung1-e5-large` ‡ | 8,000 stratified | all three, RRF | **0.4063** | 0.5500 | 0.5748 |
| `rung2-bge-m3` § | 32,043 — the whole split | all three, RRF | **0.5337** | 0.6367 | 0.7352 |
| `rung2-gte-base` § | 32,043 — the whole split | all three, RRF | **0.5298** | 0.6291 | 0.7395 |
| `rung2-e5-base` § | 32,043 — the whole split | all three, RRF | **0.4961** | 0.6295 | 0.6839 |
| `rung2-e5-large` § | 32,043 — the whole split | all three, RRF | **0.4926** | 0.6251 | 0.6896 |
| `baseline` (rejected) | — | none: a 14,607-way dense classifier | **0.0667** | 0.1623 | 0.1518 |

† `rung1-prior` was measured against the *bilingual* label text, which moved
every rung-1 figure that reads label text; its own unboosted baseline, measured
in the same pass, is 0.4149 micro R@10 rather than the 0.3964 above, so the
prior is worth +0.0053 and not the +0.0238 this column would suggest. See its
section.

The mover is `label_text.bilingual`, not the encoder revision pinning that
landed alongside it (ticket 09). For all four screened encoders the pinned
revision's `model.safetensors`, `config.json`, `tokenizer.json` and pooling
config are byte-identical to today's head — the post-cutoff commits added
model-card evaluation results and ONNX/OpenVINO exports, and nothing a run
loads — so pinning changed no vector and no figure in this document. What it
changes is that the cutoff claim is now checkable; see "rung 1 — four encoders
screened" below.

‡ The three screened encoders (ticket 09), each one flag from `configs/rung1.yaml`
and all bilingual, so the row they are read against is `rung1`'s bilingual 0.4149
rather than the German-only 0.3964 above. `gte-multilingual-base` leads at this
index by 0.0022 over `bge-m3`; rung 2 re-measures all four at the full index and
the top two swap.

§ The same four encoders at the full 32,043-document index (ticket 10), each file
one flag from its rung-1 counterpart, so the pair of rows for one encoder differs
in index size and nothing else. `bge-m3` and `gte-multilingual-base` are the two
carried to rung 3, in that order, on margins of 0.0039 over each other and 0.0336
over the third.

The three rung-1 rows are not competing. They are three mechanisms reaching
different parts of the vocabulary — kNN takes the head at 0.69 and the zero-shot
band at 0.0000, the label tower takes the zero-shot band at 0.26 and the head at
0.05, and lexical matching is nearly flat across all four bands at 0.14–0.19 —
which is what fusion has to work with (ticket 07).

The published leaderboard, for orientation, is test-set and official-aggregation
only: RUC 0.57, Annif 0.54, DUTIR831 0.54, LA2I2F 0.49 at R@10. A dev number and
a test number are not comparable, and neither is a micro figure to a macro one;
the columns are kept apart for that reason.

## rung1-knn — the neighbour retriever alone

    python scripts/run_experiment.py configs/rung1-knn.yaml

`intfloat/multilingual-e5-base`, 8,000 tib-core train documents stratified by
record type and language, subjects harvested from the 20 nearest documents,
nothing else on. 5,354 dev records, 8.8s of retrieval against a warm embedding
cache on the M4 Pro (106s cold, which is the encoder rather than the retrieval).
This is the mechanism the winning system used, so it is a reference point rather
than a warm-up.

|  | R@5 | R@10 | R@25 | R@50 |
|---|---:|---:|---:|---:|
| micro | 0.2319 | 0.3023 | 0.3793 | 0.4495 |
| official macro-over-cells | 0.3446 | 0.3948 | 0.4570 | 0.5322 |

Precision, at 2.44 gold labels per dev record: micro P@5 0.1134, P@10 0.0739.

### By frequency band (micro recall, gold assignments in brackets)

| band | R@5 | R@10 | R@25 | R@50 |
|---|---:|---:|---:|---:|
| head (2,025) | 0.6123 | 0.6923 | 0.7802 | 0.8765 |
| torso (5,766) | 0.2756 | 0.3639 | 0.4651 | 0.5636 |
| tail (4,177) | 0.0493 | 0.1089 | 0.1678 | 0.2052 |
| zero (1,117) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

The zero row is the design argument as a measurement rather than an assertion.
A label that appears in no indexed document cannot be harvested from a
neighbour at any k, at any index size, so 8.5% of dev gold assignments are
unreachable by this mechanism however good the encoder is. Reaching them is
what the document-to-label tower exists for (ticket 05), and it is the reason
the project is built as retrieval over a label vocabulary.

The head-to-tail spread is the second half of the same point: 0.69 against 0.11
at k=10. A single averaged number hides a factor of six.

### Where the official figure comes from

Six of 22 dev cells carry 51.2% of the official recall figure, and they hold
0.99% of the records between them:

| cell | records | share of records | share of the figure |
|---|---:|---:|---:|
| Book / ca | 1 | 0.02% | 10.11% |
| Book / ja | 1 | 0.02% | 10.11% |
| Article / en | 41 | 0.77% | 9.06% |
| Book / it | 3 | 0.06% | 8.42% |
| Conference / fr | 3 | 0.06% | 7.14% |
| Book / nl | 4 | 0.07% | 6.41% |

Which is why the official R@10 (0.3948) sits 0.09 above the micro one (0.3023),
and why model selection uses the micro figure. Two cells of one record each are
worth a fifth of the headline.

### By record type and language (official aggregation, as the organizers average)

| record type | R@10 | | language | R@10 |
|---|---:|---|---|---:|
| Article (54) | 0.8780 | | en (7,736) | 0.3357 |
| Book (8,467) | 0.3891 | | de (5,263) | 0.3816 |
| Conference (1,216) | 0.3435 | | | |
| Report (502) | 0.3365 | | | |
| Thesis (2,846) | 0.2181 | | | |

Theses are the weakest type by a wide margin and the second largest, so they are
where a later rung has the most to gain. German beats English by 4.6 points at
k=10, which is the cross-lingual asymmetry docs/spec.md predicted from label
names appearing verbatim in 53.7% of German assignments against 23.0% of
English ones — the case for translating label names (ticket 08).

The other ten language slices — one of them the four dev records whose language
field is empty — hold 86 gold assignments between them and are left to the run's
own output. The official scorer cannot read any of them: its reader seeds `de`
and `en` and then indexes by directory name.

## rung1-dense — the label tower alone

    python scripts/run_experiment.py configs/rung1-dense-german.yaml

`intfloat/multilingual-e5-base`, every one of the 79,427 tib-core vocabulary
entries rendered field-marked and German-only, embedded, and scored against the
document.

The command above names the *German-only* config because ticket 08 made
`label_text.bilingual` load-bearing and `configs/rung1-dense.yaml` renders both
languages now. Every figure in this section is unchanged by that: the ablation
run reproduces them to four decimals, which is how the flag was shown to be the
only thing the translation changed. The bilingual numbers are in "rung1 —
translated label names" below. No index at all: the document index is what a neighbour harvest reads,
and this mechanism does not read one. 5,354 dev records in 136.5s on the M4 Pro
cold and 6.9s warm: the difference is encoding the 79,427 label texts once. The
tower is cached by encoder and label-text revision, so it is spent again only
when the rendering changes.

|  | R@5 | R@10 | R@25 | R@50 |
|---|---:|---:|---:|---:|
| micro | 0.0931 | 0.1224 | 0.1682 | 0.2110 |
| official macro-over-cells | 0.0942 | 0.1260 | 0.1719 | 0.2279 |

Precision, micro: P@5 0.0455, P@10 0.0299. Standalone, and reported standalone
on purpose: a fused figure would not say which retriever reached which label.

### By frequency band (micro recall, gold assignments in brackets)

| band | R@5 | R@10 | R@25 | R@50 | `rung1-knn` R@10 |
|---|---:|---:|---:|---:|---:|
| head (2,025) | 0.0360 | 0.0538 | 0.0795 | 0.1072 | 0.6923 |
| torso (5,766) | 0.0656 | 0.0900 | 0.1301 | 0.1745 | 0.3639 |
| tail (4,177) | 0.1283 | 0.1628 | 0.2195 | 0.2667 | 0.1089 |
| zero (1,117) | 0.2068 | 0.2623 | 0.3339 | 0.3796 | **0.0000** |

**The zero row is what ticket 05 was for.** 0.2623 at k=10 is 293 of the 1,117
dev gold assignments that no indexed document carries, and 424 at k=50 — labels
kNN cannot return at any k or any index size, because a subject no record has
cannot be harvested from a neighbour. The band was 0.0000 in every column of the
previous row and is the reason this project is built as retrieval over a label
vocabulary rather than classification into a label set.

The band ordering is **inverted** against kNN, monotonically, and that is the
result rather than the aggregate figure. The tower is best exactly where the
neighbour harvest has nothing (zero, 0.26 against 0.00) and worst exactly where
the harvest is strongest (head, 0.05 against 0.69). Head labels are broad
headings — the 65 labels over 100 occurrences each — whose names are generic
enough that no document's text resembles them in particular; a zero-shot label
is typically a specific compound or named entity whose name appears in the
title. The two mechanisms fail in opposite directions, which is the case for
fusing them and the reason ticket 07 measures the pair rather than either alone.

Its own aggregate is well below kNN's (micro R@10 0.1224 against 0.3023), which
is expected and not a defect: 59.5% of dev gold assignments are head or torso,
where a document-similarity signal is far stronger than a name-matching one.

### Language and type

| language (official) | R@10 | | record type (official) | R@10 |
|---|---:|---|---|---:|
| de (5,263) | 0.1701 | | Book (8,467) | 0.1432 |
| en (7,736) | 0.1118 | | Thesis (2,846) | 0.1330 |
| | | | Report (502) | 0.1287 |
| | | | Conference (1,216) | 0.1215 |
| | | | Article (54) | 0.0081 |

German beats English by 5.8 points where kNN's gap was 4.6, and by half again as
much in relative terms — German is 52% ahead here against 14% there. Label text
is German-only at this rung, so an English document is matched against German
label names, and this is the measurement ticket 08's translation is aimed at.
It hits: translation lifts the English slice to 0.1090 and the German one falls
further, for a net loss on this retriever — see "rung1 — translated label names"
below, which is where that asymmetry is resolved rather than predicted.

Articles collapse to 0.0081 from kNN's 0.8780. There are 54 Article assignments
in dev and their gold subjects are head-heavy, so this is the head weakness
concentrated in one small type rather than a separate finding.

### Where the official figure comes from

Five of 22 dev cells carry 51.2% of the official figure between 0.25% of the
records — and they are a different five than kNN's, because a cell of three
records moves 15 points on whether one name happens to match:

| cell | records | share of records | share of the figure |
|---|---:|---:|---:|
| Book / pl | 3 | 0.06% | 15.84% |
| Book / zh | 3 | 0.06% | 15.13% |
| Book / it | 3 | 0.06% | 7.56% |
| Book / (empty) | 4 | 0.07% | 6.55% |
| Book / de | 1,522 | 28.43% | 6.07% |

Which is why the two aggregations sit 0.004 apart here (0.1224 micro against
0.1260 official) and 0.09 apart on `rung1-knn`: the divergence is a property of
the cell weighting meeting a particular model's mistakes, not a constant offset,
and model selection stays on the micro figure for that reason.

## rung1-lexical — verbatim label matching alone

    python scripts/run_experiment.py configs/rung1-lexical-german.yaml

Named as the German-only config for the same reason as the section above:
`configs/rung1-lexical.yaml` matches against English label names too as of
ticket 08. This section's figures are the German-only ones and are unchanged.

BM25 over the vocabulary's own surface strings: each label's preferred name and
each of its synonyms is one indexed string, 150,984 strings for 79,427 labels,
and a label's score is its best-matching string's. No document index and no
embeddings — `multilingual-e5-base` emits no sparse term weights, so it is
loaded only to be asked, and the run reads no vectors at all. 7.3s end to end
for 5,354 dev records, 1.4ms per record, of which most is loading the encoder
that is then never used: the index builds in 0.4s and answers every record in
1.9s, measured on its own.

Where a model does emit sparse term weights, those are used instead of BM25 and
they score the field-marked rendering the dense tower reads, because that is the
forward pass they come free from. What "lexical" matches against therefore
depends on the encoder, and no encoder in the ladder emits them yet.

|  | R@5 | R@10 | R@25 | R@50 |
|---|---:|---:|---:|---:|
| micro | 0.1131 | 0.1558 | 0.2171 | 0.2532 |
| official macro-over-cells | 0.1049 | 0.1416 | 0.1761 | 0.2008 |

Precision, micro: P@5 0.0553, P@10 0.0381. Above the label tower on every
*micro* recall column (R@10 0.1558 against 0.1224) at a small fraction of its
cost — the tower's run takes 136.5s, most of it embedding the vocabulary once,
against 2.3s of index building and matching here — and well below kNN's 0.3023.
In the official aggregation the two swap by k=50 (0.2008 against the tower's
0.2279), which is the cell weighting rather than a different ranking.

### The German-to-English gap, which is the deliverable

docs/spec.md predicted this retriever would be strongly language-asymmetric: a
label's own name or one of its synonyms appears verbatim in the record text for
53.7% of German gold assignments against 23.0% of English ones. Measured, from
the run's `language [micro]` block — every gold assignment of that language
weighted once, which is the aggregation model selection uses:

| language | assignments | R@5 | R@10 | R@25 | R@50 |
|---|---:|---:|---:|---:|---:|
| de | 5,263 | 0.1699 | 0.2322 | 0.3226 | 0.3667 |
| en | 7,736 | 0.0748 | 0.1041 | 0.1461 | 0.1768 |
| gap | | +0.0951 | +0.1281 | +0.1765 | +0.1899 |

German is **2.23× English at k=10** and 2.07× at k=50 — 12.8 points of recall
that this mechanism finds in one language and not the other. The ratio tracks
the predicted verbatim rates (53.7/23.0 = 2.33) closely enough that the
mechanism is doing what it was expected to do and nothing else.

For scale, the same gap on the other two retrievers, in the official
aggregation those rows were written with: kNN 0.3816 de against 0.3357 en (+4.6
points), the label tower 0.1701 against 0.1118 (+5.8), lexical 0.2719 against
0.1132 (+15.9). The asymmetry is roughly three times either of theirs, and it
points the same way in all three, because the label text is German-only at every
rung until ticket 08 translates it. English documents carry 58.7% of gold
assignments, so this is the largest single mismatch in the pipeline and the
strongest available case for translating label names — measured below.

The two aggregations disagree on the *size* of this gap and not its direction —
+12.8 points micro against +15.9 official — which is why the micro block is the
one quoted above: two of the 22 cells hold eight records between them and carry
31% of the official figure, so the official language rows move on whether one
Chinese-language book happens to match.

Translating the label names closes most of this gap and costs the German slice
nothing; that measurement is the next section.

### By frequency band (micro recall, gold assignments in brackets)

| band | R@5 | R@10 | R@25 | R@50 | `rung1-knn` R@10 | `rung1-dense` R@10 |
|---|---:|---:|---:|---:|---:|---:|
| head (2,025) | 0.1422 | 0.1886 | 0.2351 | 0.2691 | 0.6923 | 0.0538 |
| torso (5,766) | 0.1022 | 0.1445 | 0.2109 | 0.2492 | 0.3639 | 0.0900 |
| tail (4,177) | 0.1053 | 0.1456 | 0.2088 | 0.2425 | 0.1089 | 0.1628 |
| zero (1,117) | 0.1459 | 0.1925 | 0.2480 | 0.2847 | **0.0000** | 0.2623 |

**The band profile is flat**, spanning 0.14 to 0.19 across a range where kNN
runs 0.69 to 0.00 and the label tower runs monotonically the other way. That is
the expected shape rather than a surprise: whether a heading's name appears in a
title has nothing to do with how many training records carry it, so frequency is
not a variable this mechanism responds to. It makes lexical matching the third
distinct failure mode of the three — 3.5× the tower on the head, where the tower
is weakest, and 0.1925 on zero-shot labels, where kNN is structurally zero — and
therefore worth its slot in the fusion at almost no cost. Its own two extremes
are the head and the zero-shot band, 0.1886 and 0.1925, which are the bands the
other two retrievers are respectively best and worst at.

### Where the official figure comes from

The two aggregations sit 0.0141 apart at k=10 and 0.0524 apart at k=50, the
widest gap of the three rung-1 rows, and it runs the other way than on both of
them: here the official figure is the *lower* one, because what this retriever
is good at is large German cells rather than the tiny cells that carry the vote.

| cell | records | share of records | share of the figure |
|---|---:|---:|---:|
| Book / zh | 3 | 0.06% | 19.62% |
| Book / es | 5 | 0.09% | 11.02% |
| Conference / de | 106 | 1.98% | 9.84% |
| Book / de | 1,522 | 28.43% | 9.53% |

Four cells carry 50.0% of the figure. Two of them are eight records between
them, and one Chinese-language book matching one heading is worth more of the
official number than the 1,522-record German book cell. This is the third
distinct set of dominant cells in three rows — kNN's were Book/ca and Book/ja,
the tower's Book/pl and Book/zh — which is the point docs/spec.md makes about
the aggregation rather than about any of the three retrievers.

### Types, and what the retriever cannot do

| record type (official) | R@10 |
|---|---:|
| Thesis (2,846) | 0.2194 |
| Book (8,467) | 0.1807 |
| Conference (1,216) | 0.1647 |
| Report (502) | 0.1536 |
| Article (54) | 0.0163 |

Theses are the best type here and the worst under kNN (0.2194 against 0.2181 —
the two are level, where kNN is 1.9× ahead on the micro figure and 2.8× ahead
on the official one), which is the German share of theses showing through.
Articles collapse to 0.0163 for the same reason they collapse for the label
tower: 54 assignments, English, head-heavy.

Two limits worth recording. Every dev record gets candidates — 5,353 of 5,354
fill all 50 slots, because BM25 scores any shared term and an abstract shares
ordinary words with some heading's name — so the retriever is not sparse in
output the way "verbatim matching" suggests; what the score ranks is how much of
a heading is present, and the recall figures above are what that is worth. And
no stemming or decompounding is applied, so `Polymerisationsgrad` in a title
does not reach the label `Polymerisation`. German compounds are exactly where
that loss lands, which makes the measured gap a floor rather than a ceiling.

## rung1 — the three retrievers fused, and the ablation table

    python scripts/run_experiment.py configs/rung1.yaml
    python scripts/ablate_retrievers.py configs/rung1.yaml --tune-weights

Reciprocal rank fusion over all three retrievers, weights `knn` 1.5, `dense` 1.0,
`lexical` 1.0 and `rrf_k` 60, tuned on dev and committed to
[configs/rung1.yaml](../configs/rung1.yaml). Same encoder, same 8,000-document
index and same German-only label text as the three rows above, so the only new
thing here is the combining. 5,354 dev records in 15.0s against a warm embedding
cache: retrieval is 13.7s of that, and fusing 5,354 records × three ranked lists
is the rest.

|  | R@5 | R@10 | R@25 | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|
| micro | 0.3052 | **0.3964** | 0.4712 | 0.5568 | 0.6242 |
| official macro-over-cells | 0.4521 | 0.5340 | 0.6011 | 0.6530 | 0.7043 |

Precision, micro: P@5 0.1492, P@10 0.0969 — above kNN's 0.1134 and 0.0739, so
fusion is not buying recall by spending precision.

A fused candidate carries the fused score and the rank each retriever gave it,
not the retrievers' own scores: those are the three incomparable units fusion
exists to avoid combining. A stage that needs one reads that retriever's own
output through `pipeline.retrieve`, which is what the ablation harness does.

**R@100 is the ceiling.** Candidate generation emits 100 codes per record and
every later stage — reranking, the group prior, adjudication — can only reorder
within them. 0.6242 micro is what those stages have to work with, and the
submission's own 50 already reach 0.5568 of it.

### The ablation table

Every retriever alone, every pair, and all three, one row each, at the tuned
weights. All seven rows come from one retrieval pass:
`scripts/ablate_retrievers.py` runs the retrievers once and fuses each subset of
that pass through the same `pipeline.combine` the pipeline itself calls.

| retrievers | R@5 | R@10 | R@25 | R@50 | R@100 | official R@10 |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.0931 | 0.1224 | 0.1682 | 0.2110 | 0.2564 | 0.1260 |
| knn | 0.2319 | 0.3023 | 0.3793 | 0.4495 | 0.5091 | 0.3948 |
| lexical | 0.1131 | 0.1558 | 0.2171 | 0.2532 | 0.2864 | 0.1416 |
| dense + knn | 0.2555 | 0.3298 | 0.4011 | 0.5073 | 0.5794 | 0.4534 |
| dense + lexical | 0.1639 | 0.2099 | 0.2740 | 0.3256 | 0.3763 | 0.1736 |
| knn + lexical | 0.2731 | 0.3434 | 0.4109 | 0.5084 | 0.5836 | 0.4239 |
| **dense + knn + lexical** | **0.3052** | **0.3964** | **0.4712** | **0.5568** | **0.6242** | **0.5340** |

The single-retriever rows are the three sections above, reproduced to four
decimals by this second code path, which is the check that the ablation harness
and `run_experiment.py` are measuring the same thing.

**Fusion's contribution is a number: +0.0942 micro R@10 over the best single
retriever** (0.3964 against kNN's 0.3023, a 31% relative gain) and +0.1152 at the
ceiling (0.6242 against 0.5091). Every pair beats both of its members, and all
three beat every pair, so no retriever is dead weight — the third one is worth
+0.0530 on top of `knn + lexical`, +0.0666 on top of `dense + knn`, and +0.1865
on top of `dense + lexical`.

The two pairs containing kNN are close to each other (0.3434 and 0.3298) and both
far above the pair without it, which is the expected shape: 59.5% of dev gold
assignments are head or torso, where document similarity is the strongest
available signal. What is less expected is that `knn + lexical` edges out
`dense + knn` at every k including the ceiling — BM25 over label names contributes
more to the neighbour harvest than the label tower does, at a fraction of the
cost — and that
the order reverses in the official aggregation (0.4239 against 0.4534), because
the tower's gains land in the tiny cells that carry the official vote.

### The ablation by frequency band (micro R@10)

| retrievers | head | torso | tail | zero |
|---|---:|---:|---:|---:|
| dense | 0.0538 | 0.0900 | 0.1628 | 0.2623 |
| knn | 0.6923 | 0.3639 | 0.1089 | 0.0000 |
| lexical | 0.1886 | 0.1445 | 0.1456 | 0.1925 |
| dense + knn | 0.6993 | 0.3968 | 0.1463 | 0.0000 |
| dense + lexical | 0.1812 | 0.1793 | 0.2361 | 0.3214 |
| knn + lexical | 0.7274 | 0.4251 | 0.1365 | 0.0000 |
| dense + knn + lexical | 0.7072 | 0.4511 | 0.2262 | 0.1871 |

The fused row is above every single retriever in three of the four bands, and
the margins say what fusion is doing: head 0.7072 against kNN's 0.6923, torso
0.4511 against 0.3639, tail 0.2262 against the tower's 0.1628. The three
mechanisms fail in different places — kNN monotonically down the bands, the
tower monotonically up, lexical flat — and combining them recovers most of each.

**The zero-shot band is the exception, and it is a cost rather than a rounding
error.** The tower alone reaches 0.2623 there; fused, the band drops to 0.1871.
Two rows explain it. `knn + lexical` scores 0.0000 on a band where `lexical`
alone scores 0.1925: kNN is structurally empty in that band, and giving it
weight 1.5 pushes every zero-shot candidate the other retrievers found out of the
top ten. Weights tuned on an aggregate that is 59.5% head-and-torso will do that,
and the tuning had no reason not to: the sweep selects on micro R@10 and is not
shown the bands, so a weighting that protects the zero band was never something
it was asked for. The band is not lost, only demoted: at the ceiling it is 0.4288, against the tower's own 0.4333 (below).
Ranking those candidates back up is what the reranker (ticket 12) and the group
prior (ticket 11) are for, and this row is the number they have to beat.

### The candidate ceiling by band (micro R@100)

| retrievers | head | torso | tail | zero |
|---|---:|---:|---:|---:|
| dense | 0.1407 | 0.2171 | 0.3194 | 0.4333 |
| knn | 0.9215 | 0.6542 | 0.2449 | 0.0000 |
| lexical | 0.3106 | 0.2848 | 0.2686 | 0.3178 |
| dense + knn | 0.9052 | 0.6516 | 0.3864 | 0.3375 |
| dense + lexical | 0.3136 | 0.3519 | 0.4082 | 0.4969 |
| knn + lexical | 0.9111 | 0.6774 | 0.3840 | 0.2525 |
| dense + knn + lexical | 0.9027 | 0.6793 | 0.4654 | 0.4288 |

This is the table the later stages are budgeted against. The fused candidate set
holds the right label for 90.3% of head assignments, 67.9% of torso, 46.5% of
tail and 42.9% of zero-shot ones; nothing downstream can exceed those without a
different candidate generator. The head and torso columns are essentially kNN's
(0.9215 and 0.6542) — fusion neither adds nor destroys much where the harvest is
strong — while tail and zero are where the other two retrievers do their work:
0.4654 against kNN's 0.2449, and 0.4288 against its structural 0.0000.

### Tuning the weights

    python scripts/ablate_retrievers.py configs/rung1.yaml --tune-weights

256 combinations — eight weights each for `knn` and `lexical` against `dense`
held at 1.0, times four values of `rrf_k` — fused and scored on the same single
retrieval pass, 290s in total. Reciprocal rank fusion is invariant to a global
scaling of the weights, so holding one at 1.0 costs no coverage.

| dense | knn | lexical | rrf_k | micro R@10 | micro R@100 |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.50 | 1.00 | 60 | **0.3964** | 0.6242 |
| 1.00 | 2.00 | 1.50 | 60 | 0.3951 | 0.6084 |
| 1.00 | 2.00 | 1.50 | 100 | 0.3930 | 0.5937 |
| 1.00 | 1.50 | 1.00 | 30 | 0.3920 | 0.6225 |
| 1.00 | 1.50 | 1.00 | 100 | 0.3904 | 0.6223 |

The weights are selected on the same dev split this section reports, so the
fused row carries a tuning advantage the single-retriever rows do not: they are
untuned by construction, having nothing to fuse. +0.0942 is therefore an upper
estimate of what fusion is worth, and the honest lower bound is the untuned
equal-weight figure, +0.0682. Rung 2 retunes on the same split for the same
reason, and the test split stays closed until ticket 17.

Selection is micro R@10, as everywhere else in this document. The surface is
flat: equal weights at `rrf_k` 60 score 0.3705, so tuning is worth +0.0259, and
the top five combinations sit within 0.006 of each other. `rrf_k` moves the
figure less than the weights do — the four values are within 0.015 at the best
weights — and the winning setting is the conventional 60, which is worth
recording as "the default was not beaten" rather than as a tuned parameter.
The weights are a statement about three retrievers under one encoder, so rung 2
and rung 3 retune rather than inherit these.

### Where the official figure comes from

The divergence is the widest of any row in this document: official R@10 0.5340
against micro 0.3964, +0.1376.

| cell | records | share of records | share of the figure |
|---|---:|---:|---:|
| Book / ca | 1 | 0.02% | 7.65% |
| Book / ja | 1 | 0.02% | 7.65% |
| Book / zh | 3 | 0.06% | 7.50% |
| Article / en | 41 | 0.77% | 6.75% |
| Book / it | 3 | 0.06% | 6.48% |
| Conference / fr | 3 | 0.06% | 6.08% |
| Book / nl | 4 | 0.07% | 5.65% |
| Conference / de | 106 | 1.98% | 5.05% |

Eight of 22 cells carry 52.8% of the official figure between 3.0% of the
records. The eight are the union of the cells that dominated the three single
retrievers rather than a fourth set — which is what a fused system should look
like — and the concentration is why the leaderboard-comparable number is quoted
beside the micro one and never instead of it.

### Language and type

| language (micro) | R@10 | | record type (official) | R@10 |
|---|---:|---|---|---:|
| de (5,263) | 0.4549 | | Article (54) | 0.8659 |
| en (7,736) | 0.3559 | | Book (8,467) | 0.4912 |
| gap | +0.0990 | | Conference (1,216) | 0.4499 |
| | | | Report (502) | 0.4246 |
| | | | Thesis (2,846) | 0.3601 |

The German-to-English gap is +9.9 points micro, against the lexical retriever's
+12.8 micro — the two aggregations disagree on the size of this gap, so the
figures are quoted micro throughout and kNN's, which this document reports only
in the official aggregation, is left out of the comparison rather than mixed
into it. Label text is German-only at this rung, so it is
still the largest single mismatch in the pipeline (ticket 08). Theses remain the
weakest type and the second largest, at 0.3601 against Books' 0.4912.

For orientation only: the published leaderboard is test-set and official
aggregation — RUC 0.57, Annif 0.54, DUTIR831 0.54, LA2I2F 0.49 at R@10. This row
is dev, at rung 1, with off-the-shelf weights and an 8,000-document index, and
its official-aggregation R@10 is 0.5340. Those numbers are not comparable and
the test split stays closed until ticket 17; the resemblance is a reason to keep
going, not a result.

## baseline — the classifier this project rejects

    ssh nlp2 'cd ~/projects/llms4subjects && .venv/bin/python -u -m baseline.train --epochs 15 --device cuda'
    rsync -az nlp2:~/projects/llms4subjects/artifacts/baseline/ artifacts/baseline/
    python -m baseline.score artifacts/baseline/mbert-dense/predictions-core_dev.json --markdown

`bert-base-multilingual-cased`, CLS pooling, one dense layer over the 14,607
labels that occur in `core_train`, `BCEWithLogitsLoss`, AdamW at 2e-5, 15 epochs,
batch 64, 256 tokens. **1.25 h wall clock on one A100 80GB** (`nlp2`), 300 s per
epoch. Trained on `core_train`, validated on `core_dev`, dev loss falling
monotonically to 0.00133; predictions pulled back and scored here, never on the
host. Configuration and per-epoch cost are in `artifacts/baseline/mbert-dense/run.json`.

This is the reference point rather than a rung: it is the approach docs/spec.md
rejects, re-run on the clean splits so the rejection is a measurement. The
project's earlier numbers for it were taken on the contaminated split described
in [legacy/README.md](../legacy/README.md) — 189 dev records shared with train,
142 eventual gold-test records inside train — and this row replaces them.

|  | R@5 | R@10 | R@25 | R@50 |
|---|---:|---:|---:|---:|
| micro | 0.0472 | 0.0667 | 0.1057 | 0.1518 |
| official macro-over-cells | 0.1423 | 0.1623 | 0.1925 | 0.2454 |

Precision, at 2.44 gold labels per dev record: micro P@5 0.0230, P@10 0.0163.

### By frequency band (micro recall, gold assignments in brackets)

| band | R@5 | R@10 | R@25 | R@50 |
|---|---:|---:|---:|---:|
| head (2,025) | 0.3047 | 0.4311 | 0.6726 | 0.9353 |
| torso (5,766) | 0.0000 | 0.0000 | 0.0036 | 0.0160 |
| tail (4,177) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| zero (1,117) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

The zero row is not a weak number, it is a structural one, and it is the whole
argument of the project stated as a measurement. The head has one column per
label seen in `core_train`, so a label that never occurs there cannot be emitted
at any score, at any k, after any amount of training.
`tests/test_baseline_classifier.py` asserts it with *random* scores, because no
training run can change it. On dev that is 1,117 of 13,085 gold assignments
(8.5%) and 1,053 of 5,563 distinct gold labels (18.9%) — matching the test-side
19.1% docs/spec.md quotes.

The two rows above it were not predicted in advance and are the more damaging
finding: **torso and tail recall are zero too**, out to k=50, and they carry 76%
of dev's gold assignments between them. A 14,607-way sigmoid head trained with
BCE on 78,037 assignments sees each torso label in 10–100 documents and each tail
label in 1–9, against a per-column negative rate above 99.98%, and collapses onto
the frequent columns:

| | |
|---|---:|
| distinct codes emitted anywhere in 5,354 x 50 slots | 78 of 14,607 |
| distinct codes emitted in any top-10 | 26 |
| top-50 slots occupied by head-band labels | 97.4% |
| top-50 slots occupied by torso-band labels | 2.6% |

So the classifier does not so much rank the vocabulary as memorise its 26 most
common headings. Its head-band R@50 of 0.9353 is the same fact from the other
side: where it has hundreds of examples per column it is genuinely strong, and
that is 16% of the assignments.

Against the retrieval mechanisms measured above, at micro R@10: kNN alone
reaches 0.3023, the label tower 0.1224, lexical matching 0.1558, the classifier
0.0667 — after 1.25 h of A100 time against their 8.8 s of retrieval. The gap is
widest exactly where the benchmark's mass is, and the zero column is unreachable
for it by construction, which is why the pipeline is retrieval over a label
vocabulary rather than classification into a label set.

### Where the official figure comes from

Three of 22 dev cells carry 55.3% of the official recall figure, and two of them
hold one record each:

| cell | records | share of records | share of the figure |
|---|---:|---:|---:|
| Book / ca | 1 | 0.02% | 23.42% |
| Book / ja | 1 | 0.02% | 23.42% |
| Book / it | 3 | 0.06% | 8.50% |

The official R@10 (0.1623) therefore sits 0.096 above the micro one (0.0667) —
a larger gap than any retrieval row shows, because a model that only ever emits
common headings does well precisely on the tiny cells whose few records carry
common headings. Read on the official aggregation alone, the rejected approach
looks half as bad as it is.

### The graph component, excluded

docs/spec.md excludes reviving the GCN label-graph refiner: it modelled a
hierarchy the vocabulary does not contain, since the release has zero
`skos:broader` triples. Re-reading the original code added two reasons of its
own. Its edges came from a mapping keyed by classification name but tested by
label, so each group was overwritten down to a single label: 65 classification
names produced 65 two-node components covering 130 of the 14,607 labels, and the
other 14,477 had no edge at all. And its output reached the classifier as the mean over all
label embeddings, repeated across the batch — a batch-constant vector that could
only shift the head's bias, whatever the graph had contained. Reviving it would
have meant designing a new component, not re-running an old one.

`baseline/__init__.py` lists every defect found in the original code and what was
done about each. Three mattered for the measurement: the output layer's column
order came from iterating a `set`, the original resplit `core_train` three ways
and scored against a slice of itself, and its evaluation script ranked 768-wide
CLS embeddings as though they were label scores. Numbers from that path would not
have been comparable to any row here.

### Test

Not measured. The gold test set is opened once, in ticket 17, and this row is a
dev row like every other row in this document. The prediction path exists and is
gated: `python -m baseline.train --predict core_test` refuses to run without
`--open-test-set`, so producing the test row later is a flag rather than a
change to the code that produced this one.

## rung1 — translated label names

    python scripts/translate_labels.py                            # once, 122s
    python scripts/run_experiment.py configs/rung1-dense.yaml     # bilingual
    python scripts/run_experiment.py configs/rung1-dense-german.yaml
    python scripts/run_experiment.py configs/rung1-lexical.yaml   # bilingual
    python scripts/run_experiment.py configs/rung1-lexical-german.yaml

All 79,427 vocabulary entries are German and 58.7% of dev gold assignments belong
to English documents, so the three rows above measured a cross-lingual match for
most of the benchmark. `Helsinki-NLP/opus-mt-de-en` translated the 79,224
distinct German strings the renderer can ask about — every preferred name split
into base and qualifier, plus the 66 classification names — in 122 s on the M4
Pro, once, into `reference/label_translations.json`. 23.8% of them come back
identical to their source (proper nouns, formulae, loanwords) and render in one
language rather than two.

The four runs are two flag changes: `label_text.bilingual`, on the two
retrievers that read label text. kNN is not one of them and cannot be — it
harvests the gold subjects of neighbouring documents and never renders a label.

**The German-only runs reproduce the two sections above to four decimals**
(dense micro R@10 0.1224, lexical 0.1558), which is what makes the comparison a
measurement of the label text and nothing else.

### The result, by document language (micro recall, the aggregation selection uses)

The deliverable is per language, because the intervention is aimed at one of
them. Dev holds 7,736 English gold assignments and 5,263 German.

| | | de R@10 | en R@10 | de R@50 | en R@50 | micro R@10 |
|---|---|---:|---:|---:|---:|---:|
| **dense** | German-only | 0.1537 | 0.1010 | 0.2635 | 0.1752 | 0.1224 |
| | bilingual | 0.1233 | 0.1090 | 0.2229 | 0.1935 | 0.1149 |
| | change | **−0.0304** | **+0.0080** | −0.0406 | +0.0183 | −0.0075 |
| **lexical** | German-only | 0.2322 | 0.1041 | 0.3667 | 0.1768 | 0.1558 |
| | bilingual | 0.2314 | 0.1418 | 0.3831 | 0.3042 | 0.1781 |
| | change | **−0.0008** | **+0.0377** | +0.0164 | +0.1274 | +0.0223 |

**Translation helps the English half of both retrievers and the verdict still
splits, because of what the German half does.** On the lexical retriever German
is unmoved (−0.0008 at k=10, *+*0.0164 at k=50) while English gains 36% at k=10
and 72% at k=50. On the label tower English gains 7.9% and German loses 19.8% —
3.8 times as much recall as English gained — and the retriever is net worse.

The mechanism is the difference between the two, not a property of the
translations, and it is visible in how each retriever consumes the English:

- **Lexical adds a string.** Each label's English name is one more surface form
  BM25 can match, beside its German name and its synonyms — 150,984 strings
  become 212,175. A German document still matches the German name exactly as
  it did; nothing it used to match was taken away. Recall can only go up, and it
  does, by 8.3 points of micro R@50.
- **Dense adds words to one vector.** The tower has one embedding per label, and
  `Polymere / Polymers` is a different point in space from `Polymere`. Every
  German name in the vocabulary moved. A bilingual query representation would
  cost nothing, but a bilingual *label* representation is paid for in the
  language that already matched.

So the ticket's premise — that German-only label text is why label matching
trails neighbour retrieval — is half right. It is the reason the *English* half
trails, and translating fixes that on both retrievers. It is not free on a
retriever whose label representation is a single vector.

### The language gap, which is what the ticket set out to close

The de-minus-en gap at k=10, before and after, on all three retrievers:

| retriever | German-only | bilingual | closed by |
|---|---:|---:|---:|
| dense | +0.0527 | +0.0143 | 73% |
| lexical | +0.1281 | +0.0896 | 30% |

kNN has no row: it reads no label text, so its gap — +4.6 points, quoted above
in the official aggregation and not comparable to this micro table — is not a
quantity translation can reach. What it measures is the language of the
*documents* in the index, which is ticket 10's variable rather than this one's.

Both of the two close, and they close in opposite ways. The tower's gap closes by 73% and
almost all of it is German falling. The lexical gap closes by 30% and all of it
is English rising: at k=50 that retriever goes from German being 2.07× English
to 1.26×, which is the largest single language correction in the project so far.

### By frequency band (micro recall, bilingual, gold assignments in brackets)

| band | dense R@10 | Δ | lexical R@10 | Δ |
|---|---:|---:|---:|---:|
| head (2,025) | 0.0375 | −0.0163 | 0.1931 | +0.0045 |
| torso (5,766) | 0.0808 | −0.0092 | 0.1576 | +0.0131 |
| tail (4,177) | 0.1597 | −0.0031 | 0.1858 | +0.0402 |
| zero (1,117) | 0.2632 | +0.0009 | 0.2283 | +0.0358 |

The tower's loss is monotone in frequency and lands almost entirely on the head:
those 65 labels are broad German headings, they are the ones whose vectors moved
most, and they were already this retriever's weakest band. Its zero-shot band —
the band ticket 05 exists for — is untouched at 0.2632, so nothing about the
argument for the dense retriever changes.

The lexical gains run the other way, biggest on tail and zero-shot: a specific
compound or named entity is exactly the kind of heading whose English name
appears verbatim in an English abstract. Its zero-shot band goes 0.1925 to
0.2283 for no model and no training.

### Cost, and what "no per-run cost" means

| | German-only | bilingual |
|---|---:|---:|
| dense, 5,354 dev records | 7.0 s warm | 186.2 s cold, 7 s warm |
| lexical, 5,354 dev records | 7.5 s | 7.0 s |
| translation | — | 122 s, once, committed |

The 186.2 s is re-embedding the 79,427-label tower, which any change to the
label rendering costs once — the cache keys on the texts, so the bilingual tower
and the German-only one are two files and switching between them is free
thereafter. Nothing in either run loads a translation model; the renderer looks
strings up in a dict, and `tests/test_label_translations.py` asserts against the
source that no module on the rendering path can import one.

### What this changes

`label_text.bilingual` stays `true` in the committed rungs. The two retrievers
that read label text are fused, not chosen between, and standalone they move in
opposite directions by unequal amounts: lexical +0.0223 micro R@10 against the
tower's −0.0075. Those two numbers are not addable — recall over a fused list is
not the sum of recall over the parts, and neither run here is fused — so what
the pair is worth under fusion is a row ticket 07 measures, not one this section
can compute. The reason to expect it positive is that the larger move is the
gain, and that translation changes *which* labels each retriever reaches rather
than only how many.

The finding argues for making the flag per-retriever rather than global, so the
tower can read German while the lexical index reads both. That is a config
change with no new mechanism behind it, and it is left as a follow-up rather
than taken here: it needs its own row against `rung2`, where the encoder ranking
is settled, rather than a fourth variant of rung 1.

## rung1 — the subject-area group prior

    python scripts/train_group_prior.py configs/rung1-prior.yaml   # fit the head, 21s
    python scripts/ablate_group_prior.py configs/rung1-prior.yaml  # sweep the weight
    python scripts/run_experiment.py configs/rung1-prior.yaml

The vocabulary has no hierarchy to propagate through — zero `skos:broader`
triples — but every one of the 79,427 subjects carries exactly one of 66
classification groups, and those groups are where a dense output layer finally
fits: 486 tib-core train documents per group, against the one-positive-in-32,000
that made the rejected 14,607-way head predict zero.

The head is one logistic regression per group over the encoder's frozen document
vectors, trained one-against-the-rest because a record's gold labels can name
more than one group. Its output is normalised to a distribution and **boosts**
candidates in likely groups; it never filters unlikely ones, because 23.3% of
multi-label records span three or more groups and a stage that can drop a
candidate is a filter whatever it is called. It is applied after fusion, so it
reorders the 100 candidates and cannot change which 100 they are — the candidate
ceiling in every row below is the unboosted one.

Fitting the 66 regressions costs 6-9 s on the M4 Pro once the document vectors
are cached, and the whole command 21 s including the model load and the accuracy
report below, so this is not a fourth GPU run.

### The classifier, on its own

| documents | scored | true groups per record | any in top 1 | any in top 2 | any in top 3 | all in top 2 | all in top 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| core_train (32,043) | 32,043 | 1.73 | 0.7246 | 0.8669 | 0.9173 | 0.5537 | 0.6560 |
| core_dev (5,354) | 5,354 | 1.74 | 0.7107 | 0.8538 | 0.9070 | 0.5295 | 0.6390 |

Against a 1/66 = 0.0152 chance rate, and with train and dev 1.4 points apart, so
the head is neither guessing nor memorising. The figure the boost's headroom is
bounded by is `all in top 2` — every true group of a record inside the two the
head names — at 0.5295 on dev.

**The head wants the corpus, not the index sample.** Trained on the 8,000
documents `index.size` samples for the retrieval index (121 per group) it scores
dev top-1 0.6137 and all-in-top-2 0.4701; trained on all 32,043 documents of
`index.corpora` (486 per group) it scores 0.7107 and 0.5295 — 9.7 points of
top-1 for no change to what is retrieved. So the head reads the corpora and
ignores `index.size`, which is also what docs/spec.md story 20 asks for: index
size and training-set size stay separately attributable. The one-off cost is
encoding those 32,043 documents, 1,255 s on the M4 Pro, and it is a cache entry
rung 2 reads back rather than recomputes.

### What the boost is worth, swept on dev (micro recall)

| weight | R@5 | R@10 | R@25 | R@50 | official R@10 |
|---|---:|---:|---:|---:|---:|
| 0 — the prior runs, contributes nothing | 0.3205 | 0.4149 | 0.4955 | 0.5686 | 0.5376 |
| 0.05 | 0.3219 | 0.4157 | 0.4963 | 0.5713 | 0.5376 |
| 0.1 | 0.3233 | 0.4182 | 0.4967 | 0.5729 | 0.5414 |
| 0.25 | 0.3255 | 0.4192 | 0.4986 | 0.5793 | **0.5501** |
| **0.5** (committed) | 0.3250 | **0.4202** | 0.5032 | 0.5840 | 0.5464 |
| 1.0 | 0.3250 | 0.4144 | 0.5062 | 0.5859 | 0.5425 |
| 2.0 | 0.3146 | 0.3993 | 0.4986 | 0.5830 | 0.5005 |

The weight is in units of a record's own candidate score range — at 1.0 a group
the head is certain of moves a candidate across the whole list — which is the
only unit that means the same thing for a fused list of reciprocal-rank sums, a
lone cosine similarity and a lone BM25 score. Weight 0 is not the same row as
the prior being off: it runs, costs its fit, and reproduces the unboosted
ranking exactly, which `scripts/ablate_group_prior.py` asserts rather than
assumes.

**+0.0053 micro R@10, +0.0088 official R@10.** The selection metric peaks at
0.5 and the official macro figure one step earlier at 0.25; the two rows are
0.001 apart on micro R@10, so the committed weight follows the project's stated
selection rule rather than the more flattering column. Past 1.0 the boost starts
overriding the retrievers and every column falls.

### By frequency band (micro R@10, at the committed weight)

| band | unboosted | boosted | change |
|---|---:|---:|---:|
| head (2,025) | 0.7022 | 0.7180 | **+0.0158** |
| torso (5,766) | 0.4610 | 0.4651 | +0.0041 |
| tail (4,177) | 0.2612 | 0.2648 | +0.0036 |
| zero (1,117) | 0.2310 | 0.2292 | **−0.0018** |

This is the check the ticket asks for, and it half fails. Three of the four
bands gain, in descending order of label frequency, and the zero-shot band
loses 0.0018 — 2 gold assignments of 1,117. The direction is structural rather
than noise: it appears at every weight above 0.1 and grows with the weight, to
−0.0269 at 2.0. A group prediction is a statement about what a document is
about, so it is loudest for the 65 broad headings that name whole subject areas,
and a zero-shot label reached only through its label text is competing for rank
against exactly those headings.

So the prior is a head-band instrument. It buys 1.6 points of the band that was
already this pipeline's strongest and gives back a fifth of a point of the band
the design exists to reach.

### What this changes

`group_prior.enabled` stays **false** in `configs/rung1.yaml` and the rest of
the ladder; `configs/rung1-prior.yaml` is the committed ablation and this
section is its number. +0.005 micro R@10 for a stage with its own training step,
its own artifact and a measurable cost to the zero-shot band is not a trade this
rung should make silently, and the honest reading of the sweep is that the boost
is worth about a third of what retuning the fusion weights was (+0.026).

Two things would change the verdict, and both belong to later rungs rather than
this one. At rung 2 the index grows to the 32,043 documents the head already
trains on, so the retrievers get stronger while the head does not — the boost
should be re-swept there rather than assumed to hold. And a per-band weight, or
a boost that skips the head band, would keep the gain without the zero-shot
cost; that is a new mechanism, so it needs its own row and is not taken here.

The unboosted column above is re-measured in the same pass rather than taken
from the `rung1` section, because the encoder revision pinning landed in this
checkout in between and moved every rung-1 figure. The pair is internally
consistent: both columns come from one retrieval pass over one index.

## rung1 + an off-the-shelf cross-encoder — screened four ways, kept as a second opinion

    python scripts/rerank_report.py configs/rung1-rerank-base.yaml --sample 300 --query label --mix fuse --sweep-mix
    python scripts/rerank_report.py configs/rung1-rerank.yaml --sample 100 --query label

Ticket 12's question is a budget question: an off-the-shelf multilingual
reranker runs first, and a fine-tune is a fourth run on the `nlp2` GPU host only
if the untrained model demonstrably helps. Two rerankers were screened, both
inside the 2025-01-31 cutoff and both pinned in
[reference/model_releases.json](../reference/model_releases.json):
`BAAI/bge-reranker-base` (XLM-R base, 278M, released 2023-09-11) and
`BAAI/bge-reranker-v2-m3` (XLM-R large, 568M, 2024-03-15).

**Every figure in this section is measured on a 300-record stratified sample of
dev, not on all 5,354**, and is therefore not comparable to the rows in the
summary table above. The reason is the cost of the stage: at `input_k: 100` a dev
pass is 535,400 document-label pairs through a model that reads both sides
jointly, which is 3.9 hours for the smaller reranker on this laptop's MPS
backend and 25 hours for the larger one. The sample is drawn by the same seeded,
type-and-language-stratified selector the index uses, because the dev CSV is
grouped and its first 300 records are 300 English ones. Within the section
everything is comparable: one retrieval pass, one candidate set, one gold set,
and the fused row re-measured in the same pass.

The candidates are the rung-1 fused top 100, whose sample figures are micro P@5
0.1700, P@10 0.1073, R@10 0.4529, R@50 0.5992.

### Precision has a ceiling, and it is not 1.0

At 2.37 gold labels per record on this sample, a perfect system scores **0.4480
at k=5** and **0.2367 at k=10**: a record with two gold labels cannot do better
than 0.4 at k=5. So the fused P@5 of 0.1700 is 37.9% of achievable rather than
17% of anything, and the leaderboard leader's test-set 0.25 at k=5 is roughly
half of what is reachable rather than a quarter. Every precision figure below is
printed against that ceiling by the harness.

### The screen: replacing the fused ranking loses

docs/spec.md's pipeline contract has reranking *reduce* the top 100 to the final
50 — the cross-encoder's order replaces the fused one. Under that contract, four
input shapes were screened, since a cross-encoder is asymmetric and its two
inputs are a query and a passage:

| reranker | query side | label text | P@5 | P@10 | R@10 | R@50 | pairs/s |
|---|---|---|---:|---:|---:|---:|---:|
| — (fused candidates) | — | — | **0.1700** | **0.1073** | **0.4529** | 0.5992 | — |
| `bge-reranker-base` | document | field-marked | 0.0913 | 0.0600 | 0.2532 | 0.5105 | 30 |
| `bge-reranker-base` | document | name only | 0.1280 | 0.0800 | 0.3376 | 0.5809 | 37 |
| `bge-reranker-base` | label | field-marked | 0.1387 | 0.0927 | 0.3910 | **0.6076** | 25 |
| `bge-reranker-base` | label | name only | 0.1480 | 0.0960 | 0.4051 | 0.6132 | 42 |

Every row loses at the k values that matter, and the losses are large — the
worst shape halves P@5. Two things about the shape do matter, and both are now
config flags rather than assumptions. Presenting the **label as the query** is
worth +0.047 P@5 with the field-marked text and +0.020 with the name-only text,
which is the opposite of how the stage was described before it was measured; a
subject heading behaves like a query and a document does not. And the **name-only
bilingual rendering** (`Erdbebensicherheit / Earthquake safety`) beats the
field-marked three-line form the label tower reads, by +0.009 P@5 and at 1.7x
the throughput, because the field-marked form is a cataloguing record and these
models were trained on passages.

The larger reranker is not better, and rules itself out on cost. On a
100-record sample (its own fused reference: P@5 0.1600, R@10 0.4280) it scores
P@5 0.1420 and R@10 0.3520 — the same shape of loss as the smaller model — at
**6 pairs a second against 42**, which is 25 hours for one dev pass. It is
`configs/rung1-rerank.yaml`, screened and not used.

### The band breakdown is why "replace" is the wrong question

The losses are not spread evenly. For the label-as-query row above:

| band | assignments | fused R@10 | replaced R@10 | difference |
|---|---:|---:|---:|---:|
| head | 131 | 0.7023 | 0.4198 | **−0.2825** |
| torso | 321 | 0.4953 | 0.4112 | −0.0841 |
| tail | 200 | 0.2900 | 0.3450 | **+0.0550** |
| zero | 59 | 0.2203 | 0.3729 | **+0.1526** |

The cross-encoder is the worst thing that has happened to the head band in this
project and the best thing that has happened to the zero-shot band. That is not
a contradiction: the head is where 20 neighbouring documents agree and document
similarity is already almost right, and the zero-shot band is reachable only
through label text, which is the one thing a cross-encoder reads carefully.
+0.1526 R@10 on the band that carries 8.9% of gold assignments and defines the
project's thesis is the largest single-band movement any component has produced.

A mechanism that is right about different records than the one before it is a
fusion problem, not a replacement one. So `reranker.mix: fuse` combines the two
orders by reciprocal rank over the same candidate set — the retrieval order at
weight 1.0, the cross-encoder's at `mix_weight`.

### Fusing the two orders wins everywhere

| mix_weight | P@5 | P@10 | R@10 | R@50 |
|---|---:|---:|---:|---:|
| 0.0 — the fused ranking, reproduced through this path | 0.1700 | 0.1073 | 0.4529 | 0.5992 |
| 0.25 | 0.1793 | 0.1110 | 0.4684 | 0.6203 |
| 0.5 | **0.1847** | 0.1123 | 0.4740 | 0.6329 |
| **1.0** | 0.1833 | **0.1153** | **0.4866** | **0.6371** |
| 2.0 | 0.1753 | 0.1097 | 0.4627 | 0.6357 |
| 4.0 | 0.1633 | 0.1047 | 0.4416 | 0.6287 |

Weight 0 reproduces the fused figures to four decimals, which is what makes the
other rows attributable: the pass runs, costs its 13 minutes, and changes
nothing. The curve rises to a maximum and falls again rather than wandering,
which is the shape of a signal rather than of 300 records of noise. Selection is
micro R@10, as everywhere else in this project, so **`mix_weight: 1.0`** is
committed to the config; P@5 peaks at 0.5 and the two are within 0.0014 of each
other at both weights.

The name-only rendering that won the replacement screen does **not** win here.
Swept the same way, it peaks at micro R@10 0.4782 (`mix_weight` 0.25) against
the field-marked form's 0.4866, while reaching a marginally higher P@5 of 0.1847
at weight 1.0 — so the two dimensions interact rather than compose: the
name-only form is the better ranker on its own and the field-marked form is the
more useful second opinion, which is what a fusion should prefer. Selection on
micro R@10 therefore keeps the field-marked text the label tower already
renders, and `label_form` stays at its `rendered` default.

At the tuned weight, against the same candidates:

| | P@5 | P@10 | R@10 | R@50 | official P@5 | official R@10 |
|---|---:|---:|---:|---:|---:|---:|
| fused | 0.1700 | 0.1073 | 0.4529 | 0.5992 | 0.1667 | 0.5576 |
| replaced | 0.1387 | 0.0927 | 0.3910 | 0.6076 | 0.1159 | 0.4125 |
| **fused with the reranker** | **0.1833** | **0.1153** | **0.4866** | **0.6371** | **0.1689** | **0.5844** |

+0.0133 P@5 and +0.0337 R@10 over the candidates it was given, or 37.9% to
40.9% of achievable precision at k=5. The band breakdown keeps most of what
replacement bought and gives back most of what it cost:

| band | assignments | fused R@10 | fused with the reranker | difference |
|---|---:|---:|---:|---:|
| head | 131 | 0.7023 | 0.6565 | −0.0458 |
| torso | 321 | 0.4953 | 0.5296 | +0.0343 |
| tail | 200 | 0.2900 | 0.3700 | +0.0800 |
| zero | 59 | 0.2203 | 0.2712 | +0.0509 |

Both document languages gain — German R@10 0.4712 to 0.5288, English 0.4365 to
0.4518 — so this is not a cross-lingual trade either. The head band still pays
0.046, and that is the honest cost of the stage.

### The confidence measure, and which ranking it should read

Ticket 12 owes the adjudicator (ticket 13) and the coverage curve (ticket 16) a
per-record confidence. `reranker.confidence` is the mean score of a record's top
5, and it is defined over any scored ranking, which turned out to matter: the
three rankings calibrate very differently against measured correctness.

| the confidence is read from | Pearson against per-record P@5 | P@5 of the least-confident 20% | overall P@5 |
|---|---:|---:|---:|
| the fused ranking (mean RRF score) | **+0.4142** | 0.0933 | 0.1700 |
| the fused-with-reranker ranking | +0.3692 | 0.0967 | 0.1833 |
| the reranker's own relevance, replacing | −0.0499 | 0.0933 | 0.0913 |

Read from the fused ranking, the measure works: in confidence deciles, P@5 falls
from 0.3200 to 0.0800 and the share of records with no correct label in their
top 5 rises from 6.7% to 66.7%, monotonically apart from two adjacent buckets.

Read from the cross-encoder's own relevance under replacement, it is worse than
useless — its **most** confident decile has a mean relevance of 0.9966 and 66.7%
of its records have no hit at all. An off-the-shelf reranker on this task is
confidently wrong, which is exactly the failure mode that would poison a router:
the records it is surest about are the ones it has ruined. That is a finding
about the measure as much as about the model, and it is the reason `confidence`
takes a ranking rather than being defined on reranked output alone.

Ticket 13 should route on the confidence of whatever ranking the pipeline
produces — fused, or fused-with-reranker at 0.3692 — and never on a bare
cross-encoder relevance.

### What the reranker actually moved

Under replacement, the mean pre-rerank rank of a code that ended up in the final
top 5 was 41.1 of 100 with the document as query and 32.8 with the label as
query, and 0.86 of the 5 codes fusion ranked first survived in the top 5: the
model was reordering nearly at random with respect to the fused list. Fused at
weight 1.0, that mean rank is 5.6 and 2.97 of fusion's top 5 survive — the
reranker is promoting from the top 20 and moving a handful of candidates, which
is what a +0.013 P@5 should look like.

### Cost, and the recommendation on fine-tuning

The pass runs on Apple Silicon with no CUDA, as the ticket requires: 30,000
pairs in 789 seconds on the M4 Pro at `max_length: 512`, batch 32, MPS. Two
implementation details are load-bearing there. Sorting the pairs by length
before batching is worth 3 to 4 times the throughput (4 to 17 pairs a second for
the larger model, 15 to 48 for the smaller), because a batch is padded to its
longest member and 100 candidates for one record all carry the same document.
And the harness caches the cross-encoder's **scores** rather than the ranking, so
the `mix_weight` sweep above — six rankings through the real stage — costs
seconds rather than six passes.

**Recommendation: do not spend a fourth `nlp2` run on a reranker fine-tune, and
keep the off-the-shelf model in `fuse` mode.** The reasoning, in the order it
should be read:

1. The off-the-shelf model does help, but only as a second opinion: +0.0337
   micro R@10 fused, against −0.0619 replaced. The ticket's condition for asking
   for GPU time — that the untrained version demonstrably helps — is met in the
   narrowest sense and not the sense the fine-tune would build on. A fine-tune
   optimises the model's own ranking, which is the one measurement that came out
   negative.
2. What the fine-tune would buy is bounded by what the component can reach.
   Reranking cannot add a candidate, so the candidate set's own recall is the
   ceiling: on this sample micro R@100 is 0.6610, and a perfect reranker over
   these candidates would score that at every k. The fused ranking starts at
   R@10 0.4529, which leaves 0.2081 for the stage to win, and the untrained
   model in `fuse` mode has taken 0.0337 of it — 16% — with no training at all.
   A fine-tune is bidding for a share of the remaining 0.1744, and only for
   records whose gold labels are already among the 100 candidates.
3. What it would cost is a fourth run on rented hardware plus the pair mining
   that feeds it: contrastive pairs from 32,043 training records against a
   79,427-code vocabulary, with hard negatives drawn from the retrievers, which
   is the same mining the rung-3 encoder fine-tunes need and would have to be
   built twice. Against the three runs already budgeted (two rung-3 encoder
   fine-tunes and the baseline re-run), it is the least attributable of the
   four: an encoder fine-tune moves candidate generation, whose ceiling is the
   thing every later stage is bounded by.
4. The cheaper experiments are not exhausted. `input_k` is untested below 100,
   and the movement figures suggest most of the gain comes from the top 20 —
   a 20-pair pass would be 5x cheaper and might keep most of +0.034. A per-band
   `mix_weight`, or one that skips the head band, would address the 0.046 the
   head still pays. Both are Apple Silicon experiments with no GPU budget
   attached, and both should be run before any fine-tune is considered.

If a fine-tune is nevertheless wanted later, the honest framing for the budget
is: one LoRA fine-tune of `bge-reranker-base` on mined TIBKAT pairs, of the same
order as one rung-3 encoder run, in exchange for a stage whose total remaining
headroom at k=10 is +0.14 and whose untrained version has already taken +0.034
of it.

### What this changes

`reranker.enabled` stays **false** in `configs/rung1.yaml` and the rest of the
ladder: this section is a 300-record sample, and a stage that costs 3.9 hours a
dev pass does not belong in the loop that runs hundreds of times until rung 2
has fixed the candidates it reranks. `configs/rung1-rerank-base.yaml` carries the
tuned setting — label as query, the field-marked label text, `mix: fuse`,
`mix_weight: 1.0` — and is the configuration ticket 17 should consider for the
single test run, where one 4-hour pass buys +0.034 R@10 and is paid once.

## rung 1 — four encoders screened

    python scripts/verify_model_releases.py --force        # once, the registry
    python scripts/screen_encoders.py configs/rung1.yaml \
        configs/rung1-e5-large.yaml configs/rung1-bge-m3.yaml \
        configs/rung1-gte-base.yaml

The cheapest decision in the project: which encoder deserves the GPU budget.
Four off-the-shelf encoders through the whole of stage one, nothing trained, all
of it on the M4 Pro. The screen refuses configs that differ in anything but
their encoder — index, label text, fusion weights, `rrf_k` and input length are
held at the values [configs/rung1.yaml](../configs/rung1.yaml) carries — so what
is ranked is the model and not a retuning that arrived with it. All four rows
are the bilingual label text, so the `rung1` figure they are read against is
0.4149 rather than the 0.3964 in the summary table.

| encoder | dim | R@5 | R@10 | R@50 | R@100 | official R@10 |
|---|---:|---:|---:|---:|---:|---:|
| **`gte-multilingual-base`** | 768 | 0.3679 | **0.4724** | 0.6481 | 0.7136 | 0.6107 |
| `bge-m3` | 1024 | 0.3748 | **0.4702** | 0.6378 | 0.7069 | 0.5981 |
| `multilingual-e5-base` | 768 | 0.3205 | **0.4149** | 0.5686 | 0.6452 | 0.5376 |
| `multilingual-e5-large` | 1024 | 0.3149 | **0.4063** | 0.5748 | 0.6485 | 0.5500 |

**The encoder is worth +0.0575 micro R@10** — 0.4724 against the 0.4149 every
rung-1 row so far was measured at, a 13.9% relative gain for a config change and
no training. It is worth +0.0684 at the candidate ceiling (0.7136 against
0.6452), which is headroom every later stage inherits.

### Size does not carry, and the ranking is not the parameter count

`multilingual-e5-large` is the same family at twice the parameters and it is
**worse than its own base model**: 0.4063 against 0.4149. Nothing about the run
differs but the checkpoint, so this is not a tuning artifact, and the
per-retriever split says where it happens:

| encoder | dense | knn | lexical | fused |
|---|---:|---:|---:|---:|
| `gte-multilingual-base` | 0.1999 | 0.3695 | 0.1781 | 0.4724 |
| `bge-m3` | 0.2375 | 0.3473 | 0.1781 | 0.4702 |
| `multilingual-e5-base` | 0.1149 | 0.3023 | 0.1781 | 0.4149 |
| `multilingual-e5-large` | 0.0852 | 0.3193 | 0.1781 | 0.4063 |

The large model is the **better** document-to-document encoder (kNN 0.3193
against 0.3023) and the **worst** document-to-label one (0.0852 against 0.1149,
a quarter below its base). The two towers are not one capability: scaling within
the E5 family bought neighbour similarity and lost cross-lingual short-text
matching, and because the label tower is what reaches the tail, the fused number
followed the tower rather than the neighbours.

The lexical column is the screen's own control. BM25 over label strings reads no
vectors, so it must be identical for every row, and it is — 0.1781 to four
decimals, four times over. A screen where that column moved would have a
variable nobody declared.

Two more cross-checks land on numbers measured by other code paths: kNN at
0.3023 for `multilingual-e5-base` reproduces `rung1-knn` exactly, and its dense
0.1149 and lexical 0.1781 reproduce the bilingual rows of "rung1 — translated
label names". Three harnesses, the same four decimals.

### By frequency band (micro R@10)

| encoder | head | torso | tail | zero |
|---|---:|---:|---:|---:|
| `gte-multilingual-base` | 0.6914 | **0.5437** | 0.3311 | 0.2363 |
| `bge-m3` | 0.6874 | 0.5338 | **0.3361** | **0.2498** |
| `multilingual-e5-base` | 0.7022 | 0.4610 | 0.2612 | 0.2310 |
| `multilingual-e5-large` | **0.7116** | 0.4646 | 0.2387 | 0.1791 |

**The encoder choice buys torso and tail, and the head is where the losing pair
wins.** `gte` is +0.0827 on torso and +0.0699 on tail against `e5-base`, and
−0.0108 on head. That is the shape the project wants: 65 head labels carry 16.0%
of assignments and are already at 0.69–0.71 whatever the encoder, while the 4,132
torso and tail labels carry 75.2% and are where the four models actually differ.

The zero-shot band ranks differently from the aggregate: `bge-m3` takes it at
0.2498, ahead of `gte`'s 0.2363, and it also has the strongest label tower
(0.2375 against 0.1999). Those two facts are the same fact — the tower is the
only mechanism that reaches a label no indexed document carries — and they are
the reason the ladder fine-tunes the top **two** rather than only the winner
(docs/spec.md, story 41). On the band the design exists for, the runner-up is
ahead.

### Wall clock, and what these numbers are worth

| encoder | parameters | load | stage one | total | cache |
|---|---:|---:|---:|---:|---|
| `gte-multilingual-base` | 305M | 29.8s | 1732.4s | 1762.2s | computed 3 |
| `bge-m3` | 568M | 11.8s | 6029.7s | 6041.5s | computed 3 |
| `multilingual-e5-base` | 278M | 4.2s | 14.9s | 19.1s | warm |
| `multilingual-e5-large` | 560M | 8.2s | 1863.9s | 1872.1s | computed 3 |

A pass encodes three matrices — the 8,000 index documents, all 79,427 label
texts, and the 5,354 dev records — and a warm one reads them back from disk,
which is why `e5-base`'s 19.1s is a measure of I/O and not of the encoder. The
screen counts what each pass actually computed rather than asking whether a
cache directory exists, because a run that found one matrix and computed two
paid nearly a cold run's price; `--cold` forces a throwaway artifact root when
the seconds are the whole point.

Re-run warm, both `gte-multilingual-base` and `multilingual-e5-base` score
identically to four decimals in 14.3s and 15.3s, which is what the loop that
runs hundreds of times actually costs once an encoder's vectors exist.

**Read the three cold rows as approximate.** The laptop was running two
concurrent reranker passes from ticket 12 for part of this screen, and swap
reached 41 GB of 42 GB during `bge-m3`'s tower; its 6,042s is the row that
suffered most, and the honest statement is that `gte` is cheaper than `bge-m3`
by a wide margin rather than by 3.4x. The comparison the *decision* rests on is
unaffected: recall is deterministic given the encoder and the seeded index, and
those figures would be identical on an idle machine.

The cost argument nevertheless points the same way as the score. `gte` is the
smallest of the three cold rows at 305M parameters and emits 768 dimensions
against `bge-m3`'s 1024, so it is cheaper to run, cheaper to index and cheaper
to fine-tune, and it won.

### The model cutoff, made checkable

| encoder | created | pinned revision | dated |
|---|---|---|---|
| `Alibaba-NLP/gte-multilingual-base` | 2024-07-20 | `ca1791e0bcc1` | 2025-01-09 |
| `BAAI/bge-m3` | 2024-01-27 | `5617a9f61b02` | 2024-07-03 |
| `intfloat/multilingual-e5-base` | 2023-05-19 | `d13f1b27baf3` | 2024-02-15 |
| `intfloat/multilingual-e5-large` | 2023-06-30 | `ab10c1a7f42e` | 2024-02-15 |

Every candidate predates 2025-01-31, the close of the shared task's evaluation
window, and the table above is generated from
[reference/model_releases.json](../reference/model_releases.json) rather than
typed — `python scripts/verify_model_releases.py` re-derives every date from the
hub and `--offline` re-checks the committed file, so the claim is a command
rather than a comment.

The reason it is a registry and not a sentence: **both E5 checkpoints had
commits landed on them in April 2026**, so a config naming `intfloat/
multilingual-e5-base` without a revision resolves to a 2026 state of a 2023
model, and every date in the claim stays true while the claim stops being.
`stages.encoders.resolve` now loads the pinned commit, and `models.check_cutoff`
refuses a model the registry does not vouch for before any weights are fetched.
`gte-multilingual-base` also ships its own modelling code, which loading
executes, so the registry pins that repository and commit too
(`Alibaba-NLP/new-impl@40ced75c3017`, 2024-08-11).

For all four encoders the pinned revision's `model.safetensors`, `config.json`,
`tokenizer.json` and pooling config are byte-identical to today's head — the
post-cutoff commits added model-card evaluation results and ONNX/OpenVINO
exports and nothing a run loads. So pinning moved no number in this document.
That is the finding, not a disappointment: the cost of the guarantee is zero
today, and it is the only version of the guarantee that survives a vendor
re-uploading weights under the same name.

### The index, and that it is reproducible

8,000 of the 32,043 tib-core train documents, drawn stratified by record type and
language from `index.seed: 42`. Every one of the 39 corpus cells is represented,
and the nine cells above 200 documents each hold their corpus share to within
0.002:

| record type | language | corpus | corpus share | sampled | sampled share |
|---|---|---:|---:|---:|---:|
| Book | en | 12,772 | 0.3986 | 3,175 | 0.3969 |
| Book | de | 9,135 | 0.2851 | 2,281 | 0.2851 |
| Thesis | de | 3,760 | 0.1173 | 939 | 0.1174 |
| Conference | en | 2,066 | 0.0645 | 516 | 0.0645 |
| Thesis | en | 1,975 | 0.0616 | 493 | 0.0616 |
| Report | de | 801 | 0.0250 | 200 | 0.0250 |
| Conference | de | 643 | 0.0201 | 161 | 0.0201 |
| Report | en | 442 | 0.0138 | 110 | 0.0138 |
| Article | en | 248 | 0.0077 | 62 | 0.0077 |

The other thirty cells hold fewer than 50 documents each — books, theses,
reports and conferences in Arabic, Czech, Danish, Greek, Spanish, Finnish,
French, Hebrew, Italian, Japanese, Dutch, Norwegian, Polish, Romanian, Turkish,
Ukrainian, Chinese and Estonian, plus German articles and the records whose
language field is empty — and each contributes at least one document, because a
cell whose proportional share rounds to nothing is given one anyway. An index
missing a language outright would be a different experiment rather than a
smaller one, and the official aggregation gives every cell one vote regardless
of size.

### What this decides

- **Rung 2 runs `gte-multilingual-base` and `bge-m3`**, and the fusion weights
  are retuned for whichever it settles on: the weights in `configs/rung1.yaml`
  were tuned for `multilingual-e5-base`, whose retriever balance is not these
  models' (its label tower is half as strong). The screen deliberately did not
  retune them, so `gte`'s 0.4724 is a floor rather than its best.
- **Rung 3 fine-tunes those same two**, subject to rung 2 confirming the ranking
  survives the index growing to 32,043 documents. The E5 pair is out, and the
  result that dropped them is worth carrying into the writeup: within one family
  the larger checkpoint was the worse retriever here.
- **The off-the-shelf pipeline is at 0.4724 dev micro R@10 and 0.6107 official
  R@10**, with no GPU spent and no model trained. That is the number
  fine-tuning has to beat, and it is what makes "how much of the score came from
  training" answerable at the end (docs/spec.md, story 10).

## rung 1 + LLM adjudication — the routing, the ceiling, and what is still owed

Ticket 13. The last stage: the least-confident fifth of the split is shown its
top 30 candidates and a language model chooses among them. What follows is the
half of that ticket which does not need an API key — routing, the constraint,
the cost — measured on the whole dev split. The scored rows the ticket also asks
for are named at the end as outstanding, with the command that produces them.

    python scripts/adjudicate_report.py configs/rung1-adjudicate.yaml --dry-run
    python scripts/adjudicate_report.py configs/rung1-adjudicate.yaml

`--dry-run` routes, builds every prompt and calls nothing, so the shape of a run
is checkable before any budget is spent on it.

### The constraint, which is the point of the stage

A generative model asked for GND codes freely produces identifiers that look
entirely plausible and do not exist. So the model never authors one: it is shown
a numbered list of the candidates with their label text and answers with codes
copied from it, and the answer is checked against the set it was offered. A
response naming anything else is rejected **whole** — not trimmed to its valid
part, because a model that invented one identifier is not evidence about the
ones it did not invent — logged to `artifacts/adjudicated/<key>/rejections.jsonl`
with its reason, and the record keeps the ranking it arrived with.

That makes the stage a reordering and nothing else: it cannot add a code, drop
one, or change the length of the list, so the 50-code output contract survives
whatever the model says, including nothing at all. The rejection path is covered
by a test that feeds a recorded response naming `gnd:9999999-9` and asserts the
log, the reason and the unchanged ranking.

### Who gets routed, and how much harder they are

On the 5,354 dev records, with `route_fraction: 0.2` over the fused ranking:

| | records | gold/record | P@5 | R@10 | micro R@100 (the ceiling) |
|---|---:|---:|---:|---:|---:|
| the whole split | 5,354 | 2.44 | 0.1567 | 0.4149 | 0.6452 |
| **the routed subset** | 1,070 | 2.09 | **0.0850** | **0.2604** | **0.5325** |

Routing works, in the only sense that matters here: the fifth it picks scores
0.0850 P@5 against the split's 0.1567 and 0.2604 R@10 against 0.4149. These are
the records the retrievers were least sure of and were in fact worst on, which
is what ticket 12's calibration table predicted — confidence over the *fused*
ranking correlates +0.41 with per-record P@5, and this is that correlation spent
rather than measured.

The confidence band is narrow — 0.0238 minimum, 0.0300 median, 0.0502 maximum,
with the routed set cut at ≤ 0.0265. That is a property of reciprocal rank
fusion rather than of the records: RRF scores are sums of `1/(60 + rank)` terms,
so they compress into a small range and a routing threshold is a percentile, not
a meaningful absolute. Nothing downstream reads the threshold as a number, and
nothing should.

### What the stage can possibly be worth

Reranking cannot add a candidate and neither can this, so the routed subset's
own micro R@100 — **0.5325** — is what a perfect adjudicator would score on it
at every k. Against the 0.2604 it starts from, that is 0.2721 of headroom at
k=10, the largest any late stage in this project has been offered.

But it is headroom on a fifth of the records. The routed subset holds 2,231 of
the split's 13,085 gold assignments, or 17.05%, so the split-wide arithmetic is:

    0.2721 × 0.1705 = +0.046 micro R@10, if the model were perfect

That is the ceiling of the whole stage as configured, and it is worth stating
before any money is spent rather than after: **routing 20% of records caps the
achievable gain at +0.046**, against the +0.034 the off-the-shelf cross-encoder
already takes for free. A larger `route_fraction` raises the cap and the bill in
the same proportion, and the coverage curve (ticket 16) is where that trade
belongs.

### What a pass costs

1,070 prompts, 8,506,615 characters — 7,950 mean, 11,429 longest — so roughly
2.13M input tokens at four characters a token, and at most 548K output tokens at
`max_output_tokens: 512`. At `claude-3-5-sonnet-20241022`'s list rates of $3 and
$15 per million tokens that is about $6.40 in and up to $8.20 out; the rates are
an assumption to re-check, the token counts are measured.

Responses are cached under the `adjudicated` artifact stage and **written
through as each one arrives**, so an interrupted run keeps what it has already
bought and a re-score of a finished run bills nothing. Rate limits and provider
overloads are retried with a doubling backoff; anything else — a bad key, an
unknown model name — stops the pass rather than being written down as a
thousand records the model got wrong. A failed call bought nothing and is not a
constraint violation. The key covers the model's
pinned id, the prompt revision and every knob that changes a prompt, so a
reworded prompt is a new measurement rather than the old answers under a new
name.

### The model, and the cutoff

The headline model is `claude-3-5-sonnet-20241022`, released 2024-10-22, inside
the 2025-01-31 cutoff. Hosted models have no weights to pin, so the registry
gained a second kind of entry for them: `origin: api`, pinned to the dated model
id the request actually names — `claude-3-5-sonnet-20241022` is one set of
weights where `claude-3-5-sonnet` is whichever is current — with the provider's
announcement as the source. That is a *declared* date rather than a fetched one,
and the registry records which kind of evidence each entry has rather than
letting the two look alike.

The appendix row docs/spec.md allows — a current model in this stage alone —
needs `adjudication.appendix: true`, and the flag is refused on a model inside
the cutoff. So the headline row and the model-progress row are told apart by
configuration rather than by prose, and neither can be filed as the other.

### What is still owed

Two of the ticket's acceptance criteria need an API credential, which this
environment does not have:

- dev metrics for the routed subset and the full split **after** adjudication
  (the "before" halves are the table above);
- the appendix row on a current model.

Everything they need is committed and exercised: the harness, the config, the
cache, the constraint, the log, and the registry entry for the headline model.
Each is one command away —

    export ANTHROPIC_API_KEY=...
    python scripts/adjudicate_report.py configs/rung1-adjudicate.yaml

— and the appendix row is that config with `adjudication.appendix: true` and a
current model declared in `scripts/verify_model_releases.py`'s `API_MODELS` with
`appendix=True`, which is the one thing that lets the registry hold a model the
cutoff does not cover. Nothing else changes. Until those run, the honest summary of this stage is the ceiling above:
**at most +0.046 micro R@10 for roughly 2.1M input tokens**, and no measured
figure yet.

## rung 2 — the same four encoders at the full index, and the ranking did not hold

    python scripts/screen_encoders.py configs/rung2.yaml \
        configs/rung2-e5-large.yaml configs/rung2-bge-m3.yaml \
        configs/rung2-gte-base.yaml --json reference/screens/rung2.json
    python scripts/compare_rungs.py \
        reference/screens/rung1.json reference/screens/rung2.json

The question this rung exists to answer is about the *method*, not the models:
rung 1 screened four encoders at 8,000 documents and shortlisted two, and that
shortlist is only worth something if the ranking survives the index it was
measured at. So the index grows fourfold to the whole official tib-core train
split — 32,043 documents — and **nothing else moves**. Each rung-2 config is its
rung-1 counterpart with `index.size: null` — and `index.stratify: false`, which
is inert once there is no sample to draw — and no other edit, fusion weights
included, even though those were tuned for `multilingual-e5-base`; retuning here
would have put a tuning change inside a scaling measurement.
`scripts/compare_rungs.py` refuses a pair of screens that disagree on any
section but `index`, so the rule is enforced rather than remembered.

| encoder | dim | R@5 | R@10 | R@50 | R@100 | official R@10 |
|---|---:|---:|---:|---:|---:|---:|
| **`bge-m3`** | 1024 | 0.4219 | **0.5337** | 0.7352 | 0.7878 | 0.6367 |
| `gte-multilingual-base` | 768 | 0.4076 | **0.5298** | 0.7395 | 0.7939 | 0.6291 |
| `multilingual-e5-base` | 768 | 0.3769 | **0.4961** | 0.6839 | 0.7440 | 0.6295 |
| `multilingual-e5-large` | 1024 | 0.3716 | **0.4926** | 0.6896 | 0.7508 | 0.6251 |

### The ranking flipped, and the honest reading of that

| encoder | 8,000 | 32,043 | rank @ 8,000 | rank @ 32,043 | Δ |
|---|---:|---:|---:|---:|---:|
| `bge-m3` | 0.4702 | 0.5337 | 2 | **1** | +0.0635 |
| `gte-multilingual-base` | 0.4724 | 0.5298 | 1 | **2** | +0.0573 |
| `multilingual-e5-base` | 0.4149 | 0.4961 | 3 | 3 | +0.0812 |
| `multilingual-e5-large` | 0.4063 | 0.4926 | 4 | 4 | +0.0863 |

**The ranking did not hold.** `bge-m3` and `gte-multilingual-base` swap the top
two places, which is one adjacent transposition of four: Spearman ρ 0.800,
Kendall τ-b 0.667. The E5 pair keeps its order, including the result rung 1 was
built to check — the larger checkpoint is still the worse retriever, though the
gap closes from 0.0086 to 0.0035.

The reading that survives scrutiny is narrower than "screening failed". The two
encoders that swapped are **0.0022 apart at rung 1 and 0.0039 apart at rung 2**,
against a table spread of 0.0411, and they swap in opposite directions: the pair
is effectively tied at both index sizes, and the *order within it* is not a
measurement this project can claim. What the cheap screen got right is the
**membership** of the shortlist — the same two encoders lead at both index sizes,
by margins of 0.0336 and larger over the third — and what it got wrong is which
of the two is first. A project that had shortlisted one encoder on the rung-1
screen would have taken the wrong one, by 0.0039. That is the argument for
carrying two forward (docs/spec.md, story 41), now measured rather than assumed.

`gte-multilingual-base` also *leads* at the candidate ceiling — 0.7939 against
0.7878 at R@100, and 0.7395 against 0.7352 at R@50 — while losing at R@10. So
the two are separated less by retrieval quality than by ordering within the top
ten, and since candidate generation feeds a cross-encoder that reads 100
candidates, `gte`'s pool is the better input to the stage that follows. Neither
encoder dominates, and both go to rung 3.

### Only the neighbour retriever moved, which is the control

| encoder | index | dense | knn | lexical | fused |
|---|---:|---:|---:|---:|---:|
| `bge-m3` | 8,000 | 0.2375 | 0.3473 | 0.1781 | 0.4702 |
| `bge-m3` | 32,043 | 0.2375 | 0.4667 | 0.1781 | 0.5337 |
| `gte-multilingual-base` | 8,000 | 0.1999 | 0.3695 | 0.1781 | 0.4724 |
| `gte-multilingual-base` | 32,043 | 0.1999 | 0.4867 | 0.1781 | 0.5298 |
| `multilingual-e5-base` | 8,000 | 0.1149 | 0.3023 | 0.1781 | 0.4149 |
| `multilingual-e5-base` | 32,043 | 0.1149 | 0.4255 | 0.1781 | 0.4961 |
| `multilingual-e5-large` | 8,000 | 0.0852 | 0.3193 | 0.1781 | 0.4063 |
| `multilingual-e5-large` | 32,043 | 0.0852 | 0.4479 | 0.1781 | 0.4926 |

The dense and lexical columns are **identical to four decimals at both index
sizes**, for all four encoders, and they have to be: the dense retriever scores
the 79,427-entry label tower and BM25 reads the labels' own strings, so neither
reads an indexed document. That makes them the control on the whole comparison,
the same role the lexical column played inside the rung-1 screen, and it
localises every difference above to the one retriever that grew — kNN, up
between +0.1172 and +0.1286.

It also says where the flip comes from. `gte` keeps the stronger neighbour
retriever at the full index (0.4867 against 0.4667) and `bge-m3` keeps the
stronger label tower (0.2375 against 0.1999), exactly as at rung 1. What changed
is the value of the tower *given* a strong neighbour list: with four times as
many documents to draw its twenty neighbours from, the labels kNN still misses
are increasingly ones only the tower reaches, so the encoder with the better
tower gains more from fusion than its own kNN column suggests. The flip is a fusion effect, not a reversal in either tower.

### The encoders converge as the index grows

The spread between best and worst falls from 0.0661 to 0.0411 — a third of the
encoder gap closed by index size alone — and the ordering of the *gains* is the
reverse of the ordering of the scores: the two weakest encoders gained most
(+0.0812 and +0.0863) and the two strongest least (+0.0635 and +0.0573).

Index size partially substitutes for encoder quality, which is a caution in two
directions. A screen at a small index **exaggerates** the difference between
encoders, so the 8,000-document figures are the wrong basis for a claim about how
much the encoder is worth; and rung 3 grows the index again, to 70,588
all-subjects documents, so some part of whatever fine-tuning appears to buy will
be the index rather than the training. That is separable only because index size
and training-set size are separate knobs here (docs/spec.md, story 20), and it is
why rung 3 has to report a fine-tuned and an off-the-shelf row at the *same*
index.

### Where the score comes from: the tail, at the zero-shot band's expense

| encoder | index | head | torso | tail | zero |
|---|---:|---:|---:|---:|---:|
| `bge-m3` | 8,000 | 0.6874 | 0.5338 | 0.3361 | 0.2498 |
| `bge-m3` | 32,043 | 0.6958 | 0.5879 | **0.4601** | 0.2355 |
| `gte-multilingual-base` | 8,000 | 0.6914 | 0.5437 | 0.3311 | 0.2363 |
| `gte-multilingual-base` | 32,043 | 0.6795 | 0.5888 | 0.4582 | 0.2211 |
| `multilingual-e5-base` | 8,000 | 0.7022 | 0.4610 | 0.2612 | 0.2310 |
| `multilingual-e5-base` | 32,043 | 0.7338 | 0.5482 | 0.3814 | 0.2256 |
| `multilingual-e5-large` | 8,000 | 0.7116 | 0.4646 | 0.2387 | 0.1791 |
| `multilingual-e5-large` | 32,043 | 0.7358 | 0.5550 | 0.3747 | 0.1710 |

**A fourfold index buys the tail**: +0.1240 for `bge-m3`, +0.1271 for `gte`,
+0.1202 and +0.1360 for the E5 pair, against +0.0084 and −0.0119 on the head for
the top two. That is the
same shape the encoder choice had at rung 1 and for the same reason — the head is
already near its ceiling whatever the pipeline does, and the 4,132 torso and tail
labels carry 75.2% of assignments.

**The zero-shot band goes down**, for every encoder, by 0.0054 to 0.0152. It is
displacement rather than loss, and the candidate ceiling proves it: zero-shot
recall at 100 is *unchanged to four decimals* for three of the four encoders —
0.5184 for `bge-m3`, 0.4942 for `gte`, 0.4584 for `multilingual-e5-base` — and
moves by 0.0009 for `multilingual-e5-large` (0.4288 to 0.4297), the one encoder
whose kNN list is reordered enough to push a stray zero-shot label into the 100.
A zero-shot label
appears on no training record by definition, so it appears on no indexed document
either and kNN can never propose one; growing the index adds nothing to that band
and merely lets kNN's candidates outrank the tower's in the fused top ten. Every
zero-shot label the pipeline could reach at rung 1 is still in the 100 at rung 2,
which is exactly the situation the reranker and the adjudicator exist for — and a
reason to read their zero-shot rows at this index rather than at rung 1's.

### The official aggregation barely separates these four

Micro spreads the four encoders over 0.0411 at the full index; the official
macro-over-cells figure spreads them over 0.0116, and it ranks them differently:
`bge-m3` 0.6367, then **`multilingual-e5-base` 0.6295** and `gte` 0.6291, then
`multilingual-e5-large` 0.6251. Model selection is micro R@10 throughout this
project (docs/spec.md, story 4) and the shortlist is unaffected — the two
encoders it names lead on micro by 0.0336 — but the number the leaderboard is
read on would call this a four-way tie. One vote per `<record type> × language`
cell is what does it: the encoders differ most on the tail, and the cells the
official average weights up are small ones whose labels are not tail-heavy.

### What the rung cost

| encoder | index pass | what it computed |
|---|---:|---|
| `bge-m3` | 2211.6s | the 32,043-document index |
| `multilingual-e5-large` | 1760.0s | the 32,043-document index |
| `gte-multilingual-base` | 1069.1s | the 32,043-document index |
| `multilingual-e5-base` | 28.7s | nothing — warm |

One matrix per encoder, not three: the label tower and the dev split were already
cached from rung 1, and `multilingual-e5-base`'s full-index vectors were on disk
too, from an earlier full-index pass on 2026-09-07, which is why its row is warm
at 28.7s. So the whole rung cost about 85 minutes of M4 Pro, on MPS, with no CUDA
and nothing trained — the ladder's cheap rung is cheap because the expensive half
of every pass is the 79,427-label tower, and that is encoder-keyed and shared.

The committed screens are the warm re-run, so their own cost column reads `warm`
throughout; the figures above are the cold index passes from the first run.

### What this decides

- **Rung 3 fine-tunes `bge-m3` and `gte-multilingual-base`** — the same two
  rung 1 named, now chosen at the index size the decision is made at, and in
  neither case on a margin worth defending against the other.
- **Cheap screening generalises for membership, not for order.** The 8,000-
  document screen picked the right pair and the wrong winner. Reported as a
  finding rather than as a vindication, because the ladder was designed on the
  assumption that the ranking would survive and it did not.
- **The fusion weights are still `multilingual-e5-base`'s**, held fixed across
  both rungs on purpose. Both shortlisted encoders have a label tower roughly
  twice as strong as the encoder those weights were tuned for, so 0.5337 is a
  floor; retuning is rung 3's first step.
- **The off-the-shelf pipeline is at 0.5337 dev micro R@10 and 0.6367 official
  R@10**, no GPU spent and no model trained. That is what fine-tuning has to
  beat, and it is +0.0613 on the rung-1 figure for a change of index size alone.
- Both screens are committed — [reference/screens/rung1.json](../reference/screens/rung1.json)
  and [rung2.json](../reference/screens/rung2.json) — so the comparison is
  re-derivable in milliseconds rather than in four hours, and
  `tests/test_compare_rungs.py` fails if a later config edit makes them
  incomparable.

## rung 1 — coverage against precision, the one figure outside the official metric

Ticket 16. Everything else in this document is a number the leaderboard could
also produce. This section is not, and cannot be: the submission format is
exactly 50 ranked codes with no representation for fewer, so a system on this
benchmark cannot decline a record and the official metric cannot reward
declining one. **Nothing below is comparable to a published figure.**

It is measured anyway, because the workflow the project exists to support is
suggest-and-confirm — a librarian confirming proposals — and there the useful
question is not how a system does on every record but how it does on the records
it is willing to answer, and how many it gives up to get there.

    python scripts/coverage_curve.py configs/rung1.yaml
    python scripts/coverage_curve.py configs/rung1.yaml \
        --figure artifacts/coverage/rung1-fused.png

The pass costs no inference of its own. The confidence measure is
`reranker.confidence` — the mean score of a record's top 5, the same signal
ticket 13 routes on — read over a ranking that has already been computed, and
each row is the retained subset scored through the evaluator every other table
here goes through. A config with reranking on is replayed from its cached
scoring pass or refused; the harness will not spend a cross-encoder pass to draw
a picture.

### The curve, over the fused ranking (5,354 dev records)

| coverage asked | coverage taken | records | confidence >= | P@5 | of achievable | R@10 on answered | gold answered | no hit in 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100% | 100.0% | 5354 | 0.0238 | 0.1567 | 33.9% | 0.4149 | 100.0% | 39.2% |
| 90% | 90.0% | 4819 | 0.0253 | 0.1662 | 35.3% | 0.4334 | 91.6% | 35.9% |
| 80% | 80.0% | 4284 | 0.0265 | 0.1746 | 36.5% | 0.4467 | 82.9% | 33.3% |
| 70% | 70.0% | 3748 | 0.0277 | 0.1831 | 38.0% | 0.4655 | 73.0% | 30.8% |
| 60% | 60.0% | 3213 | 0.0289 | 0.1910 | 39.1% | 0.4796 | 63.2% | 28.7% |
| 50% | 50.0% | 2677 | 0.0300 | 0.1991 | 40.3% | 0.4947 | 53.4% | 26.7% |
| 40% | 40.0% | 2142 | 0.0312 | 0.2105 | 42.1% | 0.5154 | 43.1% | 24.1% |
| 30% | 30.0% | 1607 | 0.0325 | 0.2192 | 42.9% | 0.5257 | 32.8% | 23.0% |
| 20% | 20.0% | 1071 | 0.0343 | 0.2366 | 45.5% | 0.5460 | 22.3% | 19.4% |
| 10% | 10.0% | 536 | 0.0366 | 0.2627 | 49.5% | 0.5833 | 11.5% | 15.3% |

The full-coverage row is the `rung1` fused system exactly as the rest of this
document reports it — 0.1567 P@5, 0.4149 micro R@10 — which is what makes every
row under it attributable to the threshold and to nothing else.

Precision rises monotonically as the threshold does, and it rises smoothly: no
elbow, no coverage level where the curve suddenly pays off. Declining the
least-confident half of dev buys **+0.0424 P@5** (0.1567 to 0.1991) and takes
the share of records with no correct label in their top five from 39.2% to
26.7%. Declining nine records in ten buys 0.2627, which is 49.5% of what those
records' gold set sizes allow — the closest to a ceiling anything in this
project has come, on 536 records.

That smoothness is itself the result. A confidence measure that separated a
clean subset from a hopeless one would show a knee, and this one does not: it
ranks records correctly (ticket 12 measured +0.41 Pearson against per-record
P@5) without finding a natural place to cut. So the operating point is a
policy decision about review capacity, not a discovery, and the honest summary
is the exchange rate rather than a recommended threshold: **+0.0085 P@5 per 10
points of coverage given up over the first half of the sweep, +0.0159 over the
second**. The rate roughly doubles as the curve tightens, which says the measure
is at its most useful at the bottom — it is better at naming the records that
are hopeless than at naming the ones that are easy.

### What it costs, printed beside what it buys

The `gold answered` column is why the curve is a trade rather than an
improvement. At 50% coverage the answered records hold 53.4% of the split's
13,085 gold assignments; the other 46.6% are not wrong, they are unanswered, and
a system that stopped there would leave every one of them to be assigned by
hand. `R@10 on answered` is recall over the retained records alone and must be
read the same way: 0.4947 at half coverage is not an improvement on 0.4149, it
is a different denominator.

Precision is also quoted against its ceiling, as everywhere else here: at 2.44
gold labels per dev record a perfect system scores 0.4622 at k=5, and the
ceiling *moves along the curve* — the confident records carry slightly larger
gold sets, so the achievable figure rises from 0.4622 to 0.5306 as coverage
falls, and the `of achievable` column is the one that compares rows fairly.

### The threshold is a percentile, not a number

The whole sweep spans confidences from 0.0238 to 0.0366. That narrowness is a
property of reciprocal rank fusion rather than of the records — an RRF score is
a sum of `1/(60 + rank)` terms, so the scale is compressed and the absolute
value means nothing outside the run that produced it. The curve therefore
reports coverage as the control and the threshold as an observation, which is
the same conclusion ticket 13 reached for routing.

Records tied at the threshold are answered or declined together, so a requested
coverage and the coverage actually taken can differ; both columns are printed.
On dev at these ten levels they never do, because the fused scores are nearly
all distinct.

### Over the reranked ranking, on the 300-record sample

The confidence measure is defined over any scored ranking, and which ranking it
is read from changes what it knows (ticket 12). The pipeline's reranked ranking
is the one the stage actually emits — the cross-encoder's order fused with the
retrieval order at `mix_weight: 1.0` — and its curve is the one the
suggest-and-confirm workflow would live on if reranking were switched on. It is
measured on the same 300-record stratified sample the reranker was screened at,
because a dev pass through the cross-encoder costs 3.9 hours and the cached
scoring pass from that screen is what this replays.

    python scripts/rerank_report.py configs/rung1-rerank-base.yaml --sample 300
    python scripts/coverage_curve.py configs/rung1-rerank-base.yaml --sample 300

| coverage | records | P@5 fused | P@5 reranked | no hit in 5, fused | no hit in 5, reranked |
|---|---:|---:|---:|---:|---:|
| 100% | 300 | 0.1700 | **0.1833** | 35.0% | 33.3% |
| 90% | 270 | 0.1800 | **0.1948** | 31.5% | 30.4% |
| 80% | 240 | 0.1892 | **0.2050** | 28.3% | 26.7% |
| 70% | 210 | 0.1962 | **0.2105** | 27.6% | 26.2% |
| 60% | 180 | 0.2089 | **0.2244** | 24.4% | 22.8% |
| 50% | 150 | 0.2147 | **0.2387** | 24.0% | 19.3% |
| 40% | 120 | 0.2300 | **0.2517** | 19.2% | 15.8% |
| 30% | 90 | 0.2489 | **0.2600** | 16.7% | 13.3% |
| 20% | 60 | 0.2700 | **0.2900** | 13.3% | 6.7% |
| 10% | 30 | **0.3200** | 0.3000 | 6.7% | 10.0% |

Two readings, and only the first is safe. The reranked curve sits above the
fused one at every coverage level from 100% down to 20%, by 0.011 to 0.024 P@5 —
the reranker's +0.013 at full coverage does not wash out under thresholding, and
by 50% coverage it has roughly doubled. Reranking and abstention are not
competing for the same records.

The second reading is that the two cross at 10% coverage, where the fused curve
scores higher. That row is 30 records and means nothing; it is printed because
suppressing a row that disagrees with the sentence above it would be the wrong
habit. Everything below 20% coverage on this sample is 60 records or fewer, and
the full-split fused curve above is where the shape of the tail should be read.

The reranked ranking's confidence spans 0.0260 to 0.0312 — narrower still than
the fused one's, because `mix: fuse` returns reciprocal-rank scores rather than
the model's own relevance. That is the right choice for routing and for this
curve, and ticket 12 is why: read from the cross-encoder's bare relevance under
`mix: replace`, the same measure correlates −0.05 with per-record precision, and
a curve drawn on it would slope the wrong way. The harness names which ranking
each figure was read from for exactly this reason.

### What this changes

Nothing in the pipeline, which is the point. No stage learned to abstain, no
config gained a threshold, and the submission writer still emits exactly 50
codes for every record — abstention is a reading of the confidence column, and
`tests/test_coverage_curve.py` holds the curve to being an analysis: it returns
measurements, never a ranking, and the sequence it is handed comes back
untouched.

What the project gains is the one honest number attached to the workflow it
claims to serve, and its price in the same table: **+0.0424 P@5 for half the
records, at the cost of 46.6% of the gold assignments left unanswered**. The
figure is dev-only, the harness refuses `core_test`, and every rendering of it —
table heading, plot title, this section — carries "outside the official metric",
because a coverage-restricted precision is higher than the same system's
official precision by construction and would otherwise read as a result the
leaderboard could compare.

The PNG is written where the `--figure` flag points and is not tracked;
`artifacts/` is a cache, and the command above reproduces it in a minute from
vectors already on disk.


## rung 3 — the all-subjects index, and what the GPU actually bought

Ticket 15. Two things move at this rung: the index grows from the 32,043-document
tib-core training split to the all-subjects one, and the two encoders rung 2
shortlisted are contrastively fine-tuned. They are reported apart, because they
are separable only by measurement: each encoder runs off the shelf *and*
fine-tuned at the same index, so the difference between those two rows is the
training and the difference between the off-the-shelf row and its rung-2 row is
the corpus.

```
python build_tibkat_csv.py --subset all-subjects
python scripts/verify_split_alignment.py
python scripts/mine_hard_negatives.py configs/rung2-bge-m3.yaml
python scripts/mine_hard_negatives.py configs/rung2-gte-base.yaml
# on nlp2
.venv/bin/python -u scripts/train_encoder.py configs/rung3.yaml \
    --negatives-from configs/rung2-bge-m3.yaml --batch-size 8 --negatives 2 \
    --gradient-checkpointing
# back on the Mac
python scripts/rung3_report.py \
    --pair configs/rung3-untrained.yaml configs/rung3.yaml \
    --pair configs/rung3-gte-base-untrained.yaml configs/rung3-gte-base.yaml \
    --against reference/screens/rung2.json --json reference/rung3-report.json
```

### The corpus is not what the ticket assumed, in two ways

The all-subjects training split was budgeted as 70,588 documents that are free
of contamination because the two shared-task tracks are split-aligned. Checking
that before indexing it — `scripts/verify_split_alignment.py`, the one thing in
this project that opens `core_test` before ticket 17, and it reads record ids
and nothing else — found both halves of the assumption need qualifying.

| corpus | rows | documents | duplicate ids | in `core_dev` | in `core_test` |
|---|---:|---:|---:|---:|---:|
| `core_train` | 32,043 | 32,043 | 0 | 0 | 0 |
| `all_train` | 70,633 | 70,588 | 45 | **9** | **0** |

**The test claim holds exactly.** None of the 4,910 gold test records is
anywhere in `all_train`. That is the claim rung 3 rests on, and it is now
checked rather than cited.

**The dev claim does not.** Nine `core_dev` records are in the all-subjects
training split. They are named by id in
[reference/split_alignment.json](../reference/split_alignment.json) and dropped
from every index, so the number of indexed documents is **70,579**, not 70,588.
Nine records out of 70,588 would have moved no metric detectably; that is not
the reason to drop them. The reason is that the split alignment is the whole
argument for using this corpus, and an argument with nine known exceptions in it
is not one that can be checked later.

**And the corpus double-counts 45 documents.** `all_train` is 70,633 rows under
70,588 ids: 45 documents the release files twice, agreeing on text and metadata,
and disagreeing on gold for 20 of them. They are merged into one document
carrying the union of both rows' assignments — 25 assignments that keeping
either row alone would have dropped. Indexed as they came, those 45 would vote
twice in every neighbour harvest, and every table saying "70,588 documents"
would have been describing 70,633.

Both facts are properties of the data rather than of a run, so both are recorded
in a committed attestation that every run reads before it indexes anything. A
rebuild of the dataset invalidates it by revision, and the next run then refuses
to index an unchecked corpus rather than assuming the last check still applies.

### The index grew by 2.2× and bought almost nothing

| encoder | rung 2 (32,043) | rung 3 (70,579) | index adds |
|---|---:|---:|---:|
| `BAAI/bge-m3` | 0.5337 | **0.5411** | +0.0073 |
| `Alibaba-NLP/gte-multilingual-base` | 0.5298 | **0.5376** | +0.0079 |

Dev micro R@10, off the shelf at both index sizes, nothing else moved.

Read against rung 2 this is the rung's first finding, and it is a negative one.
Going from 8,000 to 32,043 documents — 4× — bought +0.0635 and +0.0573 for these
two encoders. Going from 32,043 to 70,579 — 2.2× — buys **+0.0073 and +0.0079**,
an eighth as much for half the growth again. Per doubling of the index, rung 2
was worth about +0.03 and rung 3 is worth about +0.006.

The control holds exactly, which is what makes the comparison a measurement: the
dense and lexical columns are identical to four decimals at both index sizes,
because neither reads an indexed document.

| encoder | index | dense | knn | lexical | fused |
|---|---:|---:|---:|---:|---:|
| `bge-m3` | 32,043 | 0.2375 | 0.4667 | 0.1781 | 0.5337 |
| `bge-m3` | 70,579 | 0.2375 | **0.4901** | 0.1781 | 0.5411 |
| `gte-multilingual-base` | 32,043 | 0.1999 | 0.4867 | 0.1781 | 0.5298 |
| `gte-multilingual-base` | 70,579 | 0.1999 | **0.5061** | 0.1781 | 0.5376 |

So the whole effect is the neighbour retriever's, again — +0.0234 and +0.0194 on
kNN alone — and most of it is lost in fusion, where the tower's candidates are
competing for the same top ten. The reading is that the neighbour retriever is
approaching what this benchmark's document similarity can give: at 70,579
documents a dev record's twentieth-nearest neighbour is already close enough
that adding more neighbours changes which near-duplicate is cited rather than
which subject is proposed.

### Where the extra documents do land: the tail, and the zero band

| encoder | index | head | torso | tail | zero |
|---|---:|---:|---:|---:|---:|
| `bge-m3` | 32,043 | 0.6958 | 0.5879 | 0.4601 | 0.2355 |
| `bge-m3` | 70,579 | 0.6894 | 0.5940 | 0.4719 | **0.2578** |
| `gte-multilingual-base` | 32,043 | 0.6795 | 0.5888 | 0.4582 | 0.2211 |
| `gte-multilingual-base` | 70,579 | 0.6805 | 0.5917 | 0.4733 | **0.2399** |

The zero-shot band goes **up** — +0.0223 and +0.0188 — and at rung 2 it went
down, for every encoder. That reversal is the clearest thing in the table and it
is the band-migration effect arriving as a score: at rung 2 the extra documents
were more tib-core training records, which carry no zero-shot label by
definition, so growing the index could only crowd the tower's zero-shot
candidates out of the top ten. The all-subjects records are from a different
track, and some of them carry labels tib-core train never used. For the first
time in this project the neighbour retriever can propose a zero-shot label.

The head band is flat to slightly negative, as it has been at every rung.

### Band migration: 173 of 1,053, and the 880 that no corpus reaches

Bands stay frozen against tib-core train counts throughout, so none of this
changes a band assignment anywhere; migration is reported as its own quantity.

| frozen band | labels in dev gold | reached by the larger corpus | still unreachable |
|---|---:|---:|---:|
| head | 65 | 65 | 0 |
| torso | 1,404 | 1,404 | 0 |
| tail | 3,041 | 3,041 | 0 |
| zero | 1,053 | **173** | **880** |

| weighting | zero-shot before | reached | still zero |
|---|---:|---:|---:|
| one vote per label | 1,053 | 173 | 880 |
| one vote per gold assignment | 1,117 | 192 | 925 |

Of the 1,053 labels in the dev gold that no tib-core training record carries,
38,545 extra documents reach **173** and leave **880** — carrying **7.1% of all
gold assignments** — reachable only through label text, at any corpus size. That
is the project's sharpest number, and it is a bound rather than a result: a
label that appears on no document cannot be proposed by document similarity, so
7.1% of this benchmark is out of reach for the entire family of methods the
leaderboard is made of, including the winning one. Only the label tower, the
cross-encoder and the adjudicator can score there.

**All 173 land in the tail band**, not one in torso or head. They are labels
that appear in one to nine of the 70,579 documents, so kNN can propose them in
principle and will rarely rank them: reaching a label is not retrieving it, and
the +0.02 the zero band gained is what 173 barely-present labels are worth.

The dev figure is also a check on the ticket's own estimate, which was made
against the test split: 180 of 992 test labels converting, leaving 812 carrying
7.2% of assignments. Dev says 173 of 1,053, leaving 880 carrying 7.1%. Two
different splits agreeing to a tenth of a point on the fraction is the strongest
evidence available before ticket 17 that the bound is a property of the
benchmark rather than of a split.

### Fine-tuning is worth seven times what the corpus is

| encoder | rung 2 | rung 3, off the shelf | rung 3, fine-tuned | index adds | training adds |
|---|---:|---:|---:|---:|---:|
| `BAAI/bge-m3` | 0.5337 | 0.5411 | 0.5909 | +0.0073 | **+0.0498** |
| `Alibaba-NLP/gte-multilingual-base` | 0.5298 | 0.5376 | **0.6044** | +0.0079 | **+0.0668** |

This is the rung's headline and the answer to how much of the final score needs
a GPU at all. Of the +0.0571 and +0.0747 that rung 3 adds over rung 2, the
corpus contributes an eighth and the fine-tune the rest. Two LoRA runs of 197
and 87 minutes on one shared A100 are worth roughly seven times what 38,545
extra documents are.

| encoder | weights | R@5 | R@10 | R@50 | R@100 | official R@10 |
|---|---|---:|---:|---:|---:|---:|
| `bge-m3` | off the shelf | 0.4312 | 0.5411 | 0.7485 | 0.7994 | 0.6379 |
| `bge-m3` | fine-tuned | 0.4813 | 0.5909 | 0.7733 | 0.8235 | 0.7089 |
| `bge-m3` | **training adds** | +0.0501 | +0.0498 | +0.0248 | +0.0241 | +0.0710 |
| `gte-multilingual-base` | off the shelf | 0.4121 | 0.5376 | 0.7558 | 0.8069 | 0.6506 |
| `gte-multilingual-base` | fine-tuned | 0.4895 | 0.6044 | 0.7886 | 0.8378 | 0.6927 |
| `gte-multilingual-base` | **training adds** | +0.0774 | +0.0668 | +0.0328 | +0.0310 | +0.0421 |

The gain is concentrated at the top of the ranking: +0.05 and +0.08 at R@5
against +0.024 and +0.031 at the candidate ceiling. Fine-tuning is mostly
*ordering* the candidates the untrained retriever already found, which is what a
contrastive objective over in-batch negatives should do, and it means the stage
after this one — the cross-encoder, which reranks the top 100 — has less left to
recover than rung 1's reranker screen assumed.

### The untrained ranking did not survive training

Rung 2 shortlisted two encoders 0.0039 apart and said the order within the pair
was not a claim this project could make. It was right to.

| encoder | untrained | fine-tuned | trainable | GPU minutes |
|---|---:|---:|---:|---:|
| `bge-m3` | **0.5411** (1st) | 0.5909 (2nd) | 7.1M | 196.7 |
| `gte-multilingual-base` | 0.5376 (2nd) | **0.6044** (1st) | 2.9M | 86.6 |

`gte-multilingual-base` starts 0.0035 behind and finishes 0.0135 ahead, because
it gains half again as much from training (+0.0668 against +0.0498) — on 2.9M
trainable parameters against 7.1M, in 87 GPU-minutes against 197. A project that
had shortlisted one encoder on the untrained screen would have taken the wrong
one, for the second rung running, and would have paid 2.3× the GPU time to do
it. Carrying two encoders forward (docs/spec.md, story 41) has now been worth it
twice, and it is the only reason this is a measurement rather than an assumption.

**The official metric disagrees, and is reported rather than reconciled.** On
macro-over-cells R@10 the order is the other way: `bge-m3` 0.7089 against `gte`
0.6927. Model selection in this project is dev micro R@10 throughout
(docs/spec.md, story 4), so `gte-multilingual-base` is the rung-3 encoder, but
the leaderboard number would name the other one. The same disagreement appeared
at rung 2 and for the same reason — one vote per `<record type> × language` cell
weights up small cells whose labels are not tail-heavy — and it is the strongest
argument in this project for reporting both.

### Training moved the label tower, not the neighbours

| encoder | weights | dense | knn | lexical | fused |
|---|---|---:|---:|---:|---:|
| `bge-m3` | off the shelf | 0.2375 | 0.4901 | 0.1781 | 0.5411 |
| `bge-m3` | fine-tuned | **0.3483** | 0.5137 | 0.1781 | 0.5909 |
| `gte-multilingual-base` | off the shelf | 0.1999 | 0.5061 | 0.1781 | 0.5376 |
| `gte-multilingual-base` | fine-tuned | **0.3834** | 0.5213 | 0.1781 | 0.6044 |

The document-to-label retriever gains **+0.1108 and +0.1835** — `gte`'s nearly
doubles — while the neighbour retriever gains +0.0236 and +0.0152. That is the
objective doing exactly what it was pointed at: the loss scores a document
against label *text*, so it trains the tower directly and reaches document-to-
document similarity only as a side effect.

It also explains the encoder flip. `gte` had the weaker tower untrained (0.1999
against 0.2375) and has the stronger one trained (0.3834 against 0.3483), so
what rung 1 and rung 2 measured as a property of the model was in part a
property of *its off-the-shelf training data*, and one epoch of in-domain pairs
reverses it.

The lexical column is identical to four decimals in all four rows, as it must
be: BM25 over the labels' own strings reads no vector, so no fine-tune can move
it. It is the control that says these four rows differ only where they should.

### Training buys the head and sells the zero-shot band

| encoder | weights | head | torso | tail | zero |
|---|---|---:|---:|---:|---:|
| `bge-m3` | off the shelf | 0.6894 | 0.5940 | 0.4719 | 0.2578 |
| `bge-m3` | fine-tuned | 0.7862 | 0.6724 | 0.4879 | 0.2014 |
| `bge-m3` | **delta** | +0.0968 | +0.0784 | +0.0160 | **−0.0564** |
| `gte-multilingual-base` | off the shelf | 0.6805 | 0.5917 | 0.4733 | 0.2399 |
| `gte-multilingual-base` | fine-tuned | 0.8084 | 0.6793 | 0.5059 | 0.2167 |
| `gte-multilingual-base` | **delta** | +0.1279 | +0.0876 | +0.0326 | **−0.0233** |

This is the table the whole band breakdown exists for, and it is the least
comfortable result in the project.

Fine-tuning is **monotonic in label frequency**: the head gains most (+0.10,
+0.13), the torso next (+0.08, +0.09), the tail barely moves (+0.02, +0.03), and
**the zero-shot band goes backwards** (−0.056, −0.023). The head was already
near its ceiling before training and gained most anyway; the 992-odd labels that
carry 8.9% of the benchmark and that this whole retrieval design was chosen to
reach are the ones training makes *worse*.

The mechanism is visible in the objective. Every training pair is a document and
one of its gold headings, so every gradient step is evidence about a label some
training record carries. A label no record carries appears in the loss only as a
negative, if at all — it is in the vocabulary the tower encodes but never in a
positive — so the tower learns a geometry fitted to the 14,607 seen labels, and
the 64,820 unseen ones are pulled around by it without ever being pulled
towards anything. The head-heavy gain and the zero-shot loss are the same fact.

Two things follow, and both are for later tickets rather than claims here.
First, the zero-shot regression is smaller for the encoder that gained more
overall, so it is not a fixed cost of training. Second, this is precisely the
band the cross-encoder was measured to be good at — ticket 12 found it gains
0.15 R@10 on the zero-shot band while losing 0.28 on the head — so the two
stages fail in opposite directions, and rung 3's ranking is the input the
reranker should be screened against rather than rung 1's.

### What the rung cost

| what | where | cost |
|---|---|---:|
| mining hard negatives, both encoders | Mac, warm rung-2 vectors | 7.5 min |
| `bge-m3` fine-tune, 2 epochs, 19,521 steps | `nlp2`, shared A100 | 196.7 min |
| `gte` fine-tune, 2 epochs, 19,521 steps | `nlp2`, shared A100 | 86.6 min |
| both off-the-shelf rows at 70,579 documents | Mac, MPS | 88.5 min |
| both fine-tuned rows at 70,579 documents | Mac, MPS | 116.6 min |

About 5 hours of Apple Silicon and 4.7 of shared GPU. The two off-the-shelf rows
re-ran in 72 and 78 seconds inside the report because their vectors were already
on disk; the fine-tuned rows cost 78 and 39 minutes each, because an adapter is
part of the `encoder` section and therefore a different cache key — a trained
encoder shares no vector with its untrained self, by design.

**The GPU was shared throughout**, with a vLLM process holding 74.7 GB of the
A100's 80. That is why the batch is 8 pairs with 2 mined negatives rather than
the 32 and 4 the settings default to: three OOM crashes at larger batches cost
about an hour before the run that finished. In-batch negatives are most of the
contrastive signal, so a batch of 8 is the one place this rung is knowingly
under-powered, and the +0.05/+0.07 it bought is a floor rather than the method's
ceiling. Both runs report **0 batches skipped**, so nothing was silently dropped
once they started.

### What this decides

- **Rung 3's encoder is `gte-multilingual-base`, fine-tuned, at 0.6044 dev micro
  R@10** — +0.0707 over rung 2's best and +0.1320 over rung 1's, with no stage
  after candidate generation turned on yet.
- **The GPU is where the points are.** Index size is spent: 2.2× the corpus for
  +0.008. Any further budget belongs in training or in the stages after
  retrieval, not in more documents.
- **Fine-tuning trades the zero-shot band for the head**, which is the opposite
  of this project's stated reason for choosing retrieval over classification.
  The trade is currently worth it on aggregate — +0.067 fused against −0.023 on
  a band carrying 8.9% — but it means the retrieval design's own selling point
  now rests on the reranker and the adjudicator rather than on the tower.
- **880 dev labels, 7.1% of gold assignments, are unreachable by any document
  corpus**, and no amount of GPU changes that: it is a property of the
  benchmark, and it bounds every leaderboard system that ranks by document
  similarity.
- **The fusion weights are still `multilingual-e5-base`'s**, held fixed since
  rung 1 so that this rung measures the fine-tune rather than a retune. With the
  tower now nearly twice as strong as the weights assume, 0.6044 is a floor, and
  retuning them is the cheapest experiment left in the project.


## The test set, opened once

Ticket 17. Every configuration decision was already made on dev, so this section
adds no decisions — it runs the chosen one and records what it scores.

    python scripts/final_test.py --fix-plan   # committed as d1109ec
    python scripts/final_test.py --json artifacts/test/run.json

The configuration is `configs/test.yaml`: rung 3's fine-tuned
`gte-multilingual-base` over the 70,579-document all-subjects index, its 100
fused candidates reordered by `bge-reranker-base` in `fuse` mode and cut to the
submission's 50. The group prior is off and adjudication is a row of its own.
Nothing in that sentence was chosen after the split was read, and the evidence
is a digest committed one commit earlier: `reference/test_plan.json` fixes each
row at `f235176597e2` and `093578bccaeb`, and the run refuses a row whose
configuration has moved since.

4,910 records, 11,798 gold assignments, 2h35m on the M4 Pro.

### The headline, beside the published table

| system | P@5 | R@5 | P@10 | R@10 | Avg R@k |
|---|---:|---:|---:|---:|---:|
| RUC Team | 0.25 | 0.48 | 0.16 | 0.57 | 0.66 |
| Annif | 0.23 | 0.48 | 0.14 | 0.54 | 0.59 |
| LA2I2F | 0.20 | 0.41 | 0.13 | 0.49 | 0.58 |
| DUTIR831 | 0.23 | 0.49 | 0.13 | 0.54 | 0.56 |
| **this run** | **0.2068** | **0.5056** | **0.1346** | **0.6299** | **0.7550** |

This run's row is the organizers' own script's `Overall`, computed inside the
run over a submission tree it wrote, not by the local evaluator. The two agree:
over the 4,882 records their script scored, this project's evaluator reproduces
its `Overall` row to **1.1e-16** on every metric at every k — verified against
the trees under `artifacts/test/headline/official/` that the run itself left
behind, and measured in-run by `score_officially` from this commit onward, so a
future run prints the figure rather than having it checked afterwards. The four
published rows are quoted from docs/spec.md at the two decimals they were
published at.

**The result is above the top published row on recall and below every one of
them on precision, and those two facts are the same fact.** Read the ratio: at
2.40 gold labels per record, a system whose hits were spread evenly across
records would score P@5/R@5 ≈ 0.478. RUC's ratio is 0.52 and this run's is
0.409, which says this run's hits are concentrated on records with *few* gold
labels — and a record with one gold label is one where a single hit is full
recall.

| gold labels | records | assignments | micro R@10 |
|---:|---:|---:|---:|
| 1 | 1,671 | 1,671 | 0.7367 |
| 2 | 1,544 | 3,088 | 0.6992 |
| 3 | 830 | 2,490 | 0.6048 |
| 4 | 403 | 1,612 | 0.5273 |
| 5 or more | 456 | 2,926 | 0.4163 |
| **total** | **4,904** | **11,787** | |

Computed from the submission tree, so it covers the 4,904 records that reached
it rather than all 4,910: the six with a blank cell half have no file there.
34.1% of them carry exactly one gold heading, and that third is where the
recall figure lives. The honest summary of the leaderboard row is
therefore not "this beats RUC": it is that this system finds *a* correct heading
more often than the published systems and finds *all* of a record's headings
less often, and the aggregate the leaderboard is read on rewards the first.

### The two aggregations, which diverge by more than the gap to the leaderboard

| aggregation | R@5 | R@10 | R@25 | R@50 |
|---|---:|---:|---:|---:|
| record-micro, every record | 0.4631 | 0.5910 | 0.7391 | 0.8082 |
| official-macro, all 20 cells | 0.5516 | 0.7156 | 0.8248 | 0.8558 |
| official scorer, the 9 cells it can read | 0.5056 | 0.6299 | 0.7889 | 0.8467 |

The divergence at k = 10 is **+0.1245**, which is larger than the 0.08 that
separates the best published R@10 from the worst of the four. docs/spec.md
predicted this from the cell sizes and it is confirmed: 9 of the 20 cells carry
52.5% of the **local 20-cell macro** figure, and seven of those nine hold nine
records or fewer.

| cell | records | share of records | share of the 20-cell macro |
|---|---:|---:|---:|
| Conference / es | 2 | 0.04% | 6.45% |
| Thesis / (blank) | 1 | 0.02% | 6.45% |
| Book / cs | 1 | 0.02% | 6.14% |
| Book / ja | 1 | 0.02% | 6.14% |
| Book / fr | 9 | 0.18% | 5.81% |
| Report / fr | 1 | 0.02% | 5.54% |
| Book / (blank) | 5 | 0.10% | 5.49% |
| Book / de | 1,465 | 29.84% | 5.26% |
| Conference / de | 98 | 2.00% | 5.17% |

**Those shares are of the local 20-cell macro, and seven of the nine cells above
contribute nothing to the 0.6299 headline at all** — because eleven of the
twenty cells are ones the organizers' script cannot read. Its reader seeds its
result with `de` and `en` and five record types and then subscripts that
dictionary by directory name, so the 28 records in French, Spanish, Czech,
Turkish, Dutch, Japanese and blank-language cells raise a `KeyError` inside
their code. Six of those 28 are the blank-cell-half records, which the
submission layout also has no directory for. 4,882 of 4,910 records reach the
official trees; all 4,910 reach the local evaluator.

| cell | records | in the local 20-cell macro | in the official figure |
|---|---:|---|---|
| Book / de | 1,465 | 5.26% | scored |
| Conference / de | 98 | 5.17% | scored |
| Book / fr | 9 | 5.81% | **unreadable** |
| Book / (blank) | 5 | 5.49% | **unreadable** |
| Conference / es | 2 | 6.45% | **unreadable** |
| Book / cs, Book / ja, Report / fr, Thesis / (blank) | 1 each | 5.5–6.5% each | **unreadable** |

So the divergence has two separable causes and the table above separates them.
One is the aggregation itself: a cell holding one German conference volume gets
the same vote as one holding 1,465 German books, and that is the metric working
as specified. The other is that a third of the cells this project can score are
cells the benchmark's own scorer drops — which means the published leaderboard's
`Overall` never contained them either, and the like-for-like comparison is the
9-cell row rather than the 20-cell one. It is the 9-cell row that is quoted
against the leaderboard above.

### By frozen frequency band, which is what this project is for

| band | gold assignments | test R@10 | dev R@10, rung 3 | change |
|---|---:|---:|---:|---:|
| head | 1,885 (16.0%) | 0.7215 | 0.8084 | −0.0869 |
| torso | 5,245 (44.5%) | 0.6297 | 0.6793 | −0.0496 |
| tail | 3,622 (30.7%) | 0.5353 | 0.5059 | **+0.0294** |
| zero | 1,046 (8.9%) | 0.3547 | 0.2167 | **+0.1380** |

The band supports land within 0.1pp of the shares docs/spec.md froze the bands
at, which is the strongest single check that the test split behaves as dev
predicted.

The right-hand columns are the same encoder's dev row from rung 3 **without the
cross-encoder**, so the comparison is confounded twice — a different split and a
different pipeline — and is offered as a reading rather than a measurement. But
the shape is exactly the one ticket 12 measured on a 300-record dev sample: the
cross-encoder gains on the zero-shot band and loses on the head. Rung 3 closed
by warning that fine-tuning "buys the head and sells the zero-shot band", and
that the retrieval design's selling point now rested on the reranker. On test,
the zero-shot band scores **0.3547**, against 0.2167 for the tower alone on dev
— the stage bought back more than the fine-tune sold.

That band is 992-odd labels carrying 8.9% of the benchmark that no closed
classifier can reach at all, and it is the one number in this table that the
whole architecture was chosen to produce.

### The duplicate caveat, which turns out to be small

| row | micro R@10 | without | change | official-macro R@10 | without | change |
|---|---:|---:|---:|---:|---:|---:|
| headline | 0.5910 | 0.5893 | −0.0017 | 0.7156 | 0.7138 | −0.0018 |

**The 163 does not reproduce.** docs/spec.md and docs/idea.md commit to
reporting the score with and without "the 163 test records whose title and
abstract exactly duplicate a training record under a different identifier".
Against the clean splits that set is **134** records (2.7%) for the corpus this
run indexed, and **107** against `core_train` alone. No definition tried reaches
163:

| count | definition |
|---:|---|
| 107 | exact (title, abstract) against `core_train` |
| 125 | exact against `core_train` + `core_dev` |
| 134 | exact against `all_train` — the corpus this run indexed |
| 142 | whitespace- and case-normalised against `all_train` + `all_dev` |
| 313 | title alone against `core_train` |

The run's own receipt carries the first and third of those, which are the two
the report reads; the rest were computed against the committed splits during
the same opening, and `llms4subjects.testset.duplicated_from` reproduces any of
them from a corpus and a key function.

The figure came from docs/idea.md, written against the pre-rebuild CSVs, and
legacy/README.md records why it cannot be recovered: those files kept **no
record ids**, so "under a different TIBKAT id" was not computable on them, and
their `core_test.csv` held seven rows. The caveat was real and the count was not
measurable when it was written.

What it is worth is now measured, and it is **0.002 of R@10**. A neighbour
retriever does have the answer handed to it on those 134 records, but they are
2.7% of the split and the system is not much better on them than on the rest.
The caveat is reported because the spec committed to reporting it, and the
finding is that it does not move the headline.

### What did not run

| row | config | why |
|---|---|---|
| adjudication, headline model | `configs/test-adjudicate.yaml` | `ANTHROPIC_API_KEY` is not set |
| adjudication, current model (appendix) | none | needs the credential *and* an `API_MODELS` entry with `appendix=True` |

Both are ticket 13's outstanding debt reaching ticket 17 unchanged, and neither
is merged into anything above. The stage, its cache, its constraint, its
rejection log and its config are committed and exercised; what is missing is a
credential, and for the appendix row a registry entry, which is the only thing
that lets the model registry hold a model the 2025-01-31 cutoff does not cover.
Registering one means asserting a release date in a frozen provenance artifact,
so it is left to whoever has the model rather than guessed here.

The stage's ceiling remains the one dev measured: routing the least-confident
fifth caps the achievable gain at **+0.046 micro R@10**, because the routed
subset holds 17.05% of gold assignments and its own R@100 is 0.5325.

### What this decides

- **The test figure is 0.6299 official R@10 and 0.5910 record-micro R@10**, from
  the organizers' script on a submission tree, under a configuration committed
  before the split was opened.
- **The gap to the top four is not a gap in the direction the ticket expected.**
  Recall is above every published row and precision is below every published
  row. Both follow from where the hits fall: 34.1% of the split has one gold
  heading and this system reaches 0.7367 R@10 there against 0.4163 on records
  with five or more.
- **The aggregation is worth more than the method.** +0.1245 separates this
  system's two aggregations at k = 10; 0.08 separates the best published R@10
  from the worst of the four. Any comparison on this benchmark that quotes one
  number is quoting the cell sizes as much as the system — and a third of the
  cells this project can score are cells the benchmark's own scorer drops.
- **The zero-shot band is the result.** 0.3547 R@10 on 8.9% of gold assignments
  that a closed-vocabulary classifier scores zero on by construction, and the
  band the fine-tune had been degrading. Retrieval over label text, plus a
  cross-encoder that reads it, is what reaches them.
- **The duplicate caveat costs 0.002**, and the 163 that motivated it was a
  pre-rebuild figure that does not reproduce at any definition.
- **The split is now read.** `reference/test_run.json` records the date, the
  plan it ran under and every figure; a second read is refused unless it carries
  a justification, which is recorded beside the read it supersedes.
