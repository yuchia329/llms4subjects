# Results

One row per experiment, appended as it completes, so the writeup is a byproduct
of the work rather than a reconstruction from memory (docs/spec.md, story 57).

Every number here comes from `scripts/run_experiment.py` — or, for the rejected
baseline, from `python -m baseline.score` — both of which score through
`llms4subjects.stages.evaluator` with the bands frozen in
`reference/frequency_bands.json`; the command that produced a row is quoted with
it. **Model selection is micro Recall@10 on dev**, and the official
macro-over-cells figure is reported beside it because that is what the
leaderboard is quoted on. All numbers below are **dev**: the gold test split is
opened once, at the end of the project (ticket 17).

## Dev, by experiment

| experiment | index | retrievers | micro R@10 | official R@10 | micro R@50 |
|---|---|---|---:|---:|---:|
| `rung1-knn` | 8,000 stratified | kNN only | **0.3023** | 0.3948 | 0.4495 |
| `rung1-dense` | none read | dense label tower only | **0.1224** | 0.1260 | 0.2110 |
| `rung1-lexical` | none read | lexical label matching only | **0.1558** | 0.1416 | 0.2532 |
| `rung1` (fused) | 8,000 stratified | all three, RRF | **0.3964** | 0.5340 | 0.5568 |
| `rung1-prior` † | 8,000 stratified | all three + the 66-group prior | **0.4202** | 0.5464 | 0.5840 |
| `baseline` (rejected) | — | none: a 14,607-way dense classifier | **0.0667** | 0.1623 | 0.1518 |

† `rung1-prior` was measured after the encoder revision pinning landed, which
moved every rung-1 figure; its own unboosted baseline, measured in the same
pass, is 0.4149 micro R@10 rather than the 0.3964 above, so the prior is worth
+0.0053 and not the +0.0238 this column would suggest. See its section.

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
