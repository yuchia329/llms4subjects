# LLMs4Subjects

Retrieval-based GND subject tagging for TIBKAT records — SemEval-2025 Task 5,
`tib-core-subjects` track.

Subject assignment is treated as **retrieval over a label vocabulary**, not
classification into a label set, because 19.1% of the labels in the gold test
set never appear in training and a closed-vocabulary classifier scores zero on
them by construction. Candidates come from three fused retrievers, a
cross-encoder reranks them, and an LLM adjudicates the least-confident records.
Every metric is reported broken down by how often each label appeared in
training, which no published system on this benchmark does.

- [docs/spec.md](docs/spec.md) — what is being built and why, with the measured
  figures behind each decision
- [docs/idea.md](docs/idea.md) — the earlier design, kept as the reasoning record
- [docs/results.md](docs/results.md) — every experiment's dev numbers, appended as it lands
- [docs/artifacts.md](docs/artifacts.md) — artifact cache, configs, and the two hosts
- [legacy/README.md](legacy/README.md) — the contaminated dataset the earlier
  numbers were measured on

## Repository layout

```
llms4subjects/          the pipeline, one module per stage under stages/
  config.py             experiment configuration, loaded from configs/*.yaml
  contracts.py          the artifacts stages pass between each other
  corpus.py             the only module that reads the dataset
  artifacts.py          the cached-artifact convention
  hardware.py           device selection, and what to say when CUDA is absent
  pipeline.py           predict(...) — the seam tests assert against
  stages/               label_text, encoders, indexes, retrievers, fusion,
                        group_prior, reranker, adjudicator, evaluator, submission
baseline/               the rejected classifier, kept runnable as a results row
configs/                one committed YAML per rung of the experiment ladder
reference/              frozen reference artifacts, small and tracked on purpose
official_eval/          the organizers' scorer, unmodified
scripts/                run_experiment.py, ablate_retrievers.py,
                        screen_encoders.py, verify_model_releases.py, the
                        fixture and band freeze commands
tests/                  contract and invariant tests
```

Stages take their inputs as arguments and return named artifacts; a stage never
reads the dataset or another stage's files, so the pipeline is importable rather
than a set of scripts that re-read CSVs. `baseline/` depends on
`llms4subjects/`, never the reverse.

## Environments

The split is a hard constraint: everything except training runs on Apple
Silicon without CUDA.

```
uv venv && uv pip install -r requirements/mac.txt -r requirements/dev.txt
```

The three training runs — two rung-3 fine-tunes and the baseline re-run — go to
the GPU host with `requirements/gpu.txt`. See
[docs/artifacts.md](docs/artifacts.md) for the host workflow, including how the
dataset is rebuilt there rather than copied, and how adapters come back.

## Running

```
python -m llms4subjects configs/rung2.yaml            # resolve a config: keys, device, cache hits
python scripts/run_experiment.py configs/rung1-knn.yaml   # predict dev and score it
python scripts/rerank_report.py configs/rung1-rerank.yaml --sample 300   # what reranking changes
python scripts/translate_labels.py --report           # translation cache coverage
python scripts/verify_model_releases.py --offline     # models against the 2025-01-31 cutoff
pytest                                                # contract and invariant tests
```

`run_experiment.py` is the harness every rung is measured through: it reads the
dataset, calls `predict`, and scores the result with the shared evaluator, so a
number in [docs/results.md](docs/results.md) comes from one code path however
the model that produced it was built. `--limit N` shortens a run, `--submission
DIR` also writes the organizers' tree, and `--split core_test` is refused —
the gold test split is opened once, at the end of the project.

## Dataset

The dataset is not tracked in git. One command rebuilds all of it — records and
vocabulary — from the official release, sparse-cloning
<https://github.com/jd-coderepos/llms4subjects> into `.cache/` on first run:

```
python build_tibkat_csv.py
```

`TIBKAT_dataset/` holds the official `tib-core-subjects` splits:

| file | records | unique labels | assignments |
|---|---:|---:|---:|
| `core_train.csv` | 32,043 | 14,607 | 78,037 |
| `core_dev.csv` | 5,354 | 5,563 | 13,085 |
| `core_test.csv` | 4,910 | 5,189 | 11,798 |

`core_test.csv` is the organizers' gold-standard test set, released after the
competition. Columns are `id, type, lang, title, abstract, subjects`; `id` is the
TIBKAT record id, so splits are traceable and provably disjoint. 19.1% of test labels
never occur in train, which any label-set-closed classifier cannot recover.

`official_eval/llms4subjects-evaluation.py` is the organizers' scorer
(Precision@k / Recall@k / F1@k, broken down by record type and language).

The earlier CSVs were contaminated: 142 training records also appeared in the gold
test set, and 8,966 titles came from `all-subjects` rather than `tib-core-subjects`.
They were removed on 2026-09-07; `legacy/` records their statistics, and the files
themselves remain in git history on branch `initial_submission`. Metrics measured on
that data are not comparable to metrics measured now.

## Vocabulary and qualifier rendering

`GND_dataset/` holds the two vocabulary files, copied verbatim from the release and
re-keyed by GND code, plus `GND-Name-Qualifiers-*.json` and `qualifier_stats.json`
written by the same rebuild.

GND disambiguates homographs with a qualifier: `Interaktion,Naturwissenschaft` is a
different sense from `Interaktion,Soziologie`. The vocabulary files this project used
until 2026-09-07 had those qualifiers stripped, altering 18,043 of the 79,427 tib-core
entries and merging distinct senses into one string. The rebuild restores them.

`llms4subjects.stages.label_text` renders an entry as field-marked text, with the
qualifier rendering as a flag (`LabelTextConfig.qualifiers`), three modes:

| mode | rendering | notes |
|---|---|---|
| `parenthetical` | `Verlegung (Ortswechsel)` | default; reads as natural language to an encoder |
| `raw` | `Verlegung Ortswechsel` | the release string, untouched |
| `stripped` | `Verlegung` | the pre-rebuild form, kept as an ablation |

Preferred names are the one case the release itself flattens — the JSON holds
`Verlegung Ortswechsel` where the accompanying `*_dnb-skos.ttl` holds
`Verlegung (Ortswechsel)` — so the rebuild recovers 4,741 tib-core (24,765 all-subjects)
name boundaries into the sidecar map that the renderer reads.

How much of the vocabulary each mode moves, from `qualifier_stats.json`:

| vocabulary | entries | entries with a qualifier | qualified terms | name | synonym | related |
|---|---:|---:|---:|---:|---:|---:|
| tib-core | 79,427 | 17,959 (22.6%) | 22,068 | 4,746 | 12,463 | 4,859 |
| all | 204,739 | 61,557 (30.1%) | 77,583 | 24,772 | 40,600 | 12,211 |

## Retrieval

`llms4subjects.pipeline.predict` is the seam: records in, 50 ranked GND codes
per record out, everything else reachable only through configuration.

```python
candidates = predict(records, config, vocabulary, index_records, store)
```

`records` and `index_records` are separate arguments, and that separation is the
project's central anti-requirement made structural: nothing being predicted can
contribute its own gold subjects to its own candidate set. A record that appears
in the index is dropped from its own neighbourhood by id, and
`tests/test_pipeline.py` perturbs every input record's gold labels and asserts
the candidate sets do not move. The earlier `.train_knn_e5.py` draft built its
label universe out of the evaluation split's own gold subjects; no number from
that construction means anything.

The retrievers are three, behind one interface:

| retriever | mechanism | reaches |
|---|---|---|
| `knn` | subjects of the nearest indexed documents | labels some indexed record carries |
| `dense` | the document scored against all 79,427 label vectors | any label, seen or not |
| `lexical` | label strings matched against the document text | verbatim headings |

A code's kNN score is the summed similarity of the neighbours carrying it, so
two close documents agreeing on a subject outrank one document mentioning it.
`neighbours` documents rarely carry 50 distinct subjects — 20 neighbours at 2.4
subjects each is roughly 40 codes — so the scan continues past `neighbours` to
fill the 50-slot contract, and everything it finds there is ranked below
everything harvested. Otherwise a code seen twice at neighbours 30 and 40 could
outrank one seen at neighbour 2, and `neighbours` would be a suggestion rather
than a parameter.

The `dense` retriever is the one the design exists for. It renders every
vocabulary entry as field-marked text, embeds all 79,427 of them, and scores the
document against the whole tower, so a label is reachable by having a name
rather than by having a training example. That is not a marginal gain: 8.5% of
dev gold assignments are carried by no training record at all, and `rung1-knn`
scores 0.0000 on them at every k by construction. Label text is bilingual by
default (see below), `Definition` is off by default and available as an ablation
(`label_text.include_definition`), and `Related Subjects` are not part of the
text at all — see "Vocabulary and qualifier rendering" above and
[docs/results.md](docs/results.md) for what each choice measures.

## Bilingual label text

Every one of the 79,427 vocabulary entries is German, and 58.7% of gold label
assignments belong to English documents, so matching a document against a label
name is a cross-lingual task for most of the benchmark. The two retrievers that
read label text render it in both languages:

```
Fachgebiet: Theoretische und Physikalische Chemie / Theoretical and Physical Chemistry
Schlagwort: Polymere / Polymers
Synonyme: Makropolymere; Hochpolymere; Polymer
```

The English is looked up, never produced: `reference/label_translations.json`
holds all 79,224 distinct German strings the renderer can ask about, translated
once by `python scripts/translate_labels.py` and committed. A run reads a dict,
and no module on the rendering path can import a translation model, so the
second language costs indexing and evaluation nothing. Synonyms stay German —
they are alternate surface forms of a name whose English is on the line above.

`label_text.bilingual: false` renders German-only and reads no cache at all,
which is the ablation the results document reports separately for German and
English documents. For the lexical retriever the same flag adds each label's
English name as one more surface string to match rather than joining it to the
German one, because a lexical match is against a whole string.

Two senses of one word share one entry: the cache is keyed by German string
rather than by GND code, so `Interaktion,Naturwissenschaft` and
`Interaktion,Soziologie` share the translation of `Interaktion` and differ in
the translated qualifier, which is what keeps the homograph distinction the
vocabulary rebuild recovered. See [docs/artifacts.md](docs/artifacts.md).

The three are combined by **reciprocal rank fusion**: a code's fused score is
the sum over the retrievers that found it of `weight / (rrf_k + rank)`. Rank and
not score, because the three score in incomparable units — a summed cosine
similarity over neighbouring documents, a single cosine similarity against a
label vector, a BM25 score — and fusing the numbers themselves would hand the
ranking to whichever unit happens to be largest. `rrf_k` is what a rank is worth
against agreement between retrievers, and the weights are tuned on dev and
committed to the config so that no retriever's influence is an accident.

Every surviving candidate keeps the rank each retriever gave it, so a prediction
can be attributed to the component that found it. Candidate generation emits the
top `fusion.candidates` — 100 — of which the submission takes the top 50; the
rest are what the reranker reads and what the recall ceiling is measured at.

`scripts/ablate_retrievers.py` prints the per-retriever ablation table: every
retriever alone, every pair and all three, with band-level recall for each. The
retrievers run once and each row fuses a subset of that one pass, so seven rows
cost one pass over the index. `--tune-weights` sweeps the weights and `rrf_k` on
dev and prints the block to paste into a config. See
[docs/results.md](docs/results.md) for what the table says.

### Reranking

`reranker.enabled` puts a cross-encoder between the candidates and the
submission: it scores each record's text against each candidate's label text
jointly, and returns the top `output_k` — 50, the submission length — with the
retrievers' provenance intact. Off by default, and off means the fused ranking
is returned untouched, so every earlier row in
[docs/results.md](docs/results.md) still comes from the same call.

`reranker.mix` decides what the model's opinion does to the ranking it was
given. `replace` is the original contract — the cross-encoder's order wins — and
it loses 0.06 micro R@10 on the 300-record dev sample it was screened over,
because it wrecks the head band (−0.28) while gaining the zero-shot band
(+0.15). `fuse` combines the two orders by reciprocal rank instead, and wins
0.034 R@10 and 0.013 P@5 over the candidates it was handed. A mechanism that is
right about different records than the one before it is a fusion problem rather
than a replacement one, and this is the measurement that says so.

Two off-the-shelf multilingual rerankers are screened before any GPU time is
asked for, both pinned pre-cutoff in `reference/model_releases.json`, and which
side of the pair is the query is a flag rather than a guess because these models
are asymmetric:

```
python scripts/rerank_report.py configs/rung1-rerank.yaml --sample 300
python scripts/rerank_report.py configs/rung1-rerank-base.yaml --sample 300 --query label
```

The harness prints the fused and reranked metrics side by side, each precision
figure next to the maximum achievable at that k, the band and language
breakdowns, and the calibration of the per-record confidence measure that
`reranker.confidence` emits for the adjudicator to route on. It caches the
model's scores rather than the ranking it produced, so `--sweep-mix` tunes
`mix_weight` over six rankings through the real stage for the price of none.

`confidence` reads any scored ranking, and which one it reads matters more than
the reranker does: over the fused ranking it correlates +0.41 with per-record
P@5, over the reranker's own relevance −0.05, whose most confident decile has
66.7% of its records with no correct label at all. An off-the-shelf reranker on
this task is confidently wrong. The whole screen — two models, both pairings,
both label renderings, replacement against fusion, and the recommendation
against spending a fourth GPU run on a fine-tune — is in
[docs/results.md](docs/results.md).

Predictions are restricted to the tib-core vocabulary whatever the index holds,
which is what keeps a rung-3 all-subjects index from widening the label
universe. Enabling a stage that is not built yet raises rather than being
ignored, because a silently skipped reranker would be reported as an ablation
that never ran.

Encoding is the expensive part and it is cached by encoder plus a digest of the
input texts, so the same dev split costs 106s once and 9s thereafter. See
[docs/artifacts.md](docs/artifacts.md).

## The model cutoff, and which encoder won

Every component is restricted to models released on or before **2025-01-31**,
the close of the SemEval-2025 Task 5 evaluation window, so the comparison
against teams who competed in January 2025 is fair rather than flattered by
later model progress. A model name does not carry that claim — both E5
checkpoints this project started from had commits landed on them in April 2026 —
so it is carried by [reference/model_releases.json](reference/model_releases.json),
which records each model's creation date and pins it to the newest commit inside
the cutoff:

```
python scripts/verify_model_releases.py --offline   # re-check the committed file
python scripts/verify_model_releases.py --force     # re-derive it from the hub
```

`llms4subjects.models.check_cutoff` refuses a model the registry does not vouch
for before any weights are fetched, and `stages.encoders.resolve` is what turns
a config into the pinned revision actually loaded. A model that ships its own
modelling code has that repository and commit pinned too, since loading it runs
it.

Four encoders were screened off the shelf at the 8,000-document rung-1 index,
one config each, everything but the encoder held fixed:

| encoder | dev micro R@10 | zero-shot band |
|---|---:|---:|
| `Alibaba-NLP/gte-multilingual-base` | **0.4724** | 0.2363 |
| `BAAI/bge-m3` | 0.4702 | **0.2498** |
| `intfloat/multilingual-e5-base` | 0.4149 | 0.2310 |
| `intfloat/multilingual-e5-large` | 0.4063 | 0.1791 |

```
python scripts/screen_encoders.py configs/rung1.yaml configs/rung1-e5-large.yaml \
    configs/rung1-bge-m3.yaml configs/rung1-gte-base.yaml
```

The screen refuses configs that differ in anything but their encoder, and refuses
two configs naming the same one, because otherwise it ranks a retuning rather
than a model. `multilingual-e5-large` scoring below its own base model is the
result worth knowing: within one family the larger checkpoint was the better
document-to-document encoder and the worse document-to-label one, and the label
tower is what reaches the tail. Rungs 2 and 3 carry the top two forward; see
[docs/results.md](docs/results.md).

## Evaluation

Every result the project reports goes through `llms4subjects.stages.evaluator`,
which scores a mapping of gold label lists against a mapping of ranked
predictions and returns both aggregations side by side:

| aggregation | weighting | used for |
|---|---|---|
| micro | every gold assignment once | model selection |
| official macro-over-cells | one vote per `<record type> × language` cell | the headline, comparable to the leaderboard |

The organizers' arithmetic is reproduced exactly, rounding included, and
`tests/test_evaluator.py` holds it there: for a committed fixture it runs their
script end to end and asserts agreement at every k from 5 to 50, on the
`Overall` row, on every cell and on both of their other two sheets.

The two aggregations diverge, and the divergence is a result rather than a
footnote. On the fixture, a cell holding one record — 2% of the records — carries
45% of the official recall figure; `EvaluationReport.divergence` ranks the cells
by how much of it they account for, and `render(report)` returns the whole
report — both aggregations at all three metrics, then the slices — as text to
append to the results document.

Metrics slice by document language, record type, and label frequency band:

| band | tib-core train occurrences | labels in the band | of those, in test | share of test assignments |
|---|---|---:|---:|---:|
| head | more than 100 | 65 | 65 | 16.0% |
| torso | 10 to 100 | 1,543 | 1,384 | 44.5% |
| tail | 1 to 9 | 12,999 | 2,748 | 30.7% |
| zero | never seen | rest of the vocabulary | 992 | 8.9% |

The first two columns are the frozen artifact; the last two are the test-side
figures from [docs/spec.md](docs/spec.md), which is where they stay until ticket
17 opens the test split.

The band assignment is **frozen** in `reference/frequency_bands.json` and read,
never derived, so adding documents to the index cannot reclassify which labels
count as tail and quietly make one rung's tail number incomparable to the last.
Growing the index genuinely would move labels — `test_frequency_bands.py` proves
that on this data — which is why the file exists. Rewriting it is a deliberate
act:

```
python scripts/freeze_bands.py            # report the bands, refuse to overwrite
python scripts/freeze_bands.py --force    # rewrite the reference
```

`llms4subjects.stages.submission` writes the organizers' `<Type>/<lang>/<id>.json`
tree with exactly 50 ranked codes per record, so a run can be validated against
their script end to end. One limitation is theirs, not ours: their reader seeds
its results with `de` and `en` and then indexes by directory name, so it raises
`KeyError` on the French, Spanish, Czech, Turkish, Dutch and Japanese records the
splits contain. The writer emits the true cell and those records are scored
locally.

## The rejected baseline

`baseline/` holds the earlier approach: `bert-base-multilingual-cased` with a
dense output layer over the 14,607 training labels. It is archived, not
developed — it stays runnable only so that "a dense output layer cannot serve
this problem" is a measured row in the results table rather than an assertion.
That row is in [docs/results.md](docs/results.md), and it is the reference point
every pipeline row is read against.

```
python -m baseline.train --smoke      # first training step on a laptop (MPS)
python -m baseline.train --epochs 15  # the real run, on the GPU host
python -m baseline.score artifacts/baseline/mbert-dense/predictions-core_dev.json
```

Training and scoring are deliberately separate commands on separate machines.
The full label matrix is 32,043 records by 14,607 labels, which is a GPU-host
workload, so the run happens on `nlp2` and writes a checkpoint, a `run.json`
recording its configuration and wall clock, and ranked predictions; those come
back and are scored here, through the same evaluator and the same frozen bands
as every pipeline result. `--smoke` takes a prefix of the split so the training
loop can be exercised locally first.

The zero-shot band recall is exactly zero, and structurally so: the head has one
column per label seen in `core_train`, so a label that never occurs there cannot
be emitted at any score. `tests/test_baseline_classifier.py` asserts that with
random scores, which is the point — no training run can change it.

The graph component of the original is not revived: it modelled a hierarchy the
vocabulary does not contain (zero `skos:broader` triples exist), its edges came
from a mapping that put every label in a group of one, and its output was a
batch-constant vector that could only shift the head's bias.
`baseline/__init__.py` lists every defect found in the original code and what
was done about each, since a measurement taken through a broken metric would
not have been a measurement of the approach.
