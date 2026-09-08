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
| `baseline` (rejected) | — | none: a 14,607-way dense classifier | **0.0667** | 0.1623 | 0.1518 |

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

    python scripts/run_experiment.py configs/rung1-dense.yaml

`intfloat/multilingual-e5-base`, every one of the 79,427 tib-core vocabulary
entries rendered field-marked and German-only, embedded, and scored against the
document. No index at all: the document index is what a neighbour harvest reads,
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
(`label_text.bilingual` reads `true` in every committed config and nothing reads
it yet — the flag becomes load-bearing in ticket 08, and until then the rendering
is German whatever the config says. The loader refuses `false` for exactly that
reason: it would otherwise score identically to the default and be reported as
"translation does not help".)

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

    python scripts/run_experiment.py configs/rung1-lexical.yaml

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
strongest available case for translating label names.

The two aggregations disagree on the *size* of this gap and not its direction —
+12.8 points micro against +15.9 official — which is why the micro block is the
one quoted above: two of the 22 cells hold eight records between them and carry
31% of the official figure, so the official language rows move on whether one
Chinese-language book happens to match.

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
