# Results

One row per experiment, appended as it completes, so the writeup is a byproduct
of the work rather than a reconstruction from memory (docs/spec.md, story 57).

Every number here comes from `scripts/run_experiment.py`, which scores through
`llms4subjects.stages.evaluator`; the command that produced a row is quoted with
it. **Model selection is micro Recall@10 on dev**, and the official
macro-over-cells figure is reported beside it because that is what the
leaderboard is quoted on. All numbers below are **dev**: the gold test split is
opened once, at the end of the project (ticket 17).

## Dev, by experiment

| experiment | index | retrievers | micro R@10 | official R@10 | micro R@50 |
|---|---|---|---:|---:|---:|
| `rung1-knn` | 8,000 stratified | kNN only | **0.3023** | 0.3948 | 0.4495 |

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
