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
scripts/                one-off analysis scripts, and the two freeze commands
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
python -m llms4subjects configs/rung2.yaml   # resolve a config: keys, device, cache hits
pytest                                       # contract and invariant tests
```

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

```
python -m baseline.train --smoke      # first training step on a laptop (MPS)
python -m baseline.train --epochs 15  # the real run, on the GPU host
```

The full label matrix is 32,043 records by 14,607 labels, which is a GPU-host
workload; `--smoke` takes a prefix of the split so the training loop can be
exercised locally. The graph component in `baseline/train.py` is not revived by
the current design: it modelled a hierarchy the vocabulary does not contain
(zero `skos:broader` triples exist).
