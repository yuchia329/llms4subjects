# LLMs4Subjects

Retrieval-based GND subject tagging for TIBKAT records — SemEval-2025 Task 5,
`tib-core-subjects` track.

Subject assignment is **retrieval over a 79,427-entry label vocabulary**, not
classification into a label set: 19.1% of the labels in the gold test split
never appear in training, and a closed-vocabulary classifier scores zero on them
by construction. Three fused retrievers propose candidates, a cross-encoder
reranks, an LLM adjudicates the least-confident fifth, and every metric is
broken down by how often the label appeared in training.

## Result

Gold test split, 4,910 records, opened once on 2026-09-09 under a configuration
committed one commit earlier. Organizers' own scorer:

| system            | P@5    | R@5        | P@10   | R@10       | Avg R@k    |
| ----------------- | ------ | ---------- | ------ | ---------- | ---------- |
| RUC Team (winner) | 0.25   | 0.48       | 0.16   | 0.57       | 0.66       |
| Annif             | 0.23   | 0.48       | 0.14   | 0.54       | 0.59       |
| DUTIR831          | 0.23   | 0.49       | 0.13   | 0.54       | 0.56       |
| LA2I2F            | 0.20   | 0.41       | 0.13   | 0.49       | 0.58       |
| **this run**      | 0.2068 | **0.5056** | 0.1346 | **0.6299** | **0.7550** |

- **812 test headings appear on no document in any released corpus**, carrying
  851 assignments — 7.2% of the split. Every published system here ranks by
  document similarity, so all of them score 0.000 there. This one scores
  **0.323 R@10**, because it scores documents against label _text_.
- Recall is above every published row; precision is below the top two (P@10
  0.1346 against 0.16 and 0.14) and inside the published rounding of the other
  two. Same fact twice: this system finds _a_ correct heading more often and
  _all_ of a record's headings less often.
- Aggregation moves the headline further than method does. One set of
  predictions scores 0.5910 record-micro and 0.7156 one-vote-per-cell, 0.1245
  apart at k=10, where 0.08 separates first from fourth above.
- Every model is pinned to a revision released on or before **2025-01-31**, the
  close of the evaluation window.

| document                                         | what is in it                                                                                                  |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| [docs/results.md](docs/results.md)               | every experiment's numbers, band breakdowns, wall clocks                                                       |
| [docs/spec.md](docs/spec.md)                     | what is built and why, with the figure behind each decision                                                    |
| [docs/artifacts.md](docs/artifacts.md)           | artifact cache, configs, the two hosts                                                                         |
| [docs/idea.md](docs/idea.md)                     | the earlier design, kept as the reasoning record                                                               |
| [legacy/README.md](legacy/README.md)             | the contaminated dataset earlier numbers used                                                                  |
| [docs/812-headings.html](docs/812-headings.html) | the bound as a single page ([published](https://claude.ai/code/artifact/d65b0f93-c161-4c34-8008-1129a4c10199)) |

## How it got here

I built the first version in fall 2024, the first serious training project of
my NLP master's, on a belief I no longer hold: that the strongest pretrained
model with a classification head over the label set answers any labelling task.
It landed nowhere near the leaderboard and cost more compute than the task
deserved — it is still here, runnable, as `baseline/`. Two years later I rebuilt
it as retrieval. Five measurements, in the order they were made.

### 1. What the winning method cannot reach

kNN over document embeddings — harvest the subjects of the nearest training
neighbours — is what won this track, and it is a reasonable first row at 0.3023
dev micro R@10. By label frequency band:

| kNN alone, dev | head   | torso  | tail   | zero-shot  |
| -------------- | ------ | ------ | ------ | ---------- |
| micro R@10     | 0.6923 | 0.3639 | 0.1089 | **0.0000** |

Structural, not unlucky: a label no indexed document carries cannot be harvested
from a neighbour at any k, from any index.

### 2. A quarter of the data, three retrievers, one fusion

Rung 1 indexes 8,000 documents — a quarter of `core_train` — and holds
everything but the variable under test fixed. Two mechanisms that read label
_text_ join kNN: `dense` (the document scored against all 79,427 label vectors)
and `lexical` (BM25 over label strings). Seven rows, one retrieval pass:

| retrievers          | dev micro R@10 | head       | torso      | tail       | zero-shot  |
| ------------------- | -------------- | ---------- | ---------- | ---------- | ---------- |
| `knn`               | 0.3023         | 0.6923     | 0.3639     | 0.1089     | 0.0000     |
| `dense`             | 0.1224         | 0.0538     | 0.0900     | 0.1628     | **0.2623** |
| `lexical`           | 0.1558         | 0.1886     | 0.1445     | 0.1456     | 0.1925     |
| `dense` + `knn`     | 0.3298         | 0.6993     | 0.3968     | 0.1463     | 0.0000     |
| `knn` + `lexical`   | 0.3434         | **0.7274** | 0.4251     | 0.1365     | 0.0000     |
| `dense` + `lexical` | 0.2099         | 0.1812     | 0.1793     | 0.2361     | **0.3214** |
| **all three**       | **0.3964**     | 0.7072     | **0.4511** | **0.2262** | 0.1871     |

kNN falls monotonically down the bands, the label tower rises monotonically,
lexical is flat — so they are fused rather than chosen between:

```
score(code) = Σ over the retrievers that found it of  weight / (rrf_k + rank)
```

Ranks, not scores: a summed cosine over neighbours, a cosine against a label
vector and a BM25 score are incomparable units. `rrf_k` sets what a top rank is
worth against two retrievers agreeing further down — small values let rank 1
dominate, large ones flatten the curve until agreement decides; the conventional
60 was swept against three other values and not beaten.

|                                                                      | dev micro R@10 | against     |
| -------------------------------------------------------------------- | -------------- | ----------- |
| best single retriever (`knn`)                                        | 0.3023         | —           |
| all three, equal weights                                             | 0.3705         | +0.0682     |
| all three, tuned (`knn` 1.5, `dense` 1.0, `lexical` 1.0, `rrf_k` 60) | **0.3964**     | **+0.0942** |

Every pair beats both its members and all three beat every pair. The cost is in
the last column of the first table: fused, the zero-shot band is 0.1871 against
the tower's own 0.2623, because kNN's weight pushes zero-shot candidates out of
the top ten. The band is demoted, not lost — 0.4288 of it survives in the
100-candidate set the reranker reads.

### 3. Which encoder deserved the GPU budget

Four off-the-shelf multilingual encoders, all published before the 2025-01-31
cutoff, one config each, everything but the encoder fixed — first at 8,000
documents, then at the full 32,043:

| encoder                             | @ 8,000    | zero-shot @ 8,000 | @ 32,043   | rank  |
| ----------------------------------- | ---------- | ----------------- | ---------- | ----- |
| `Alibaba-NLP/gte-multilingual-base` | **0.4724** | 0.2363            | 0.5298     | 1 → 2 |
| `BAAI/bge-m3`                       | 0.4702     | **0.2498**        | **0.5337** | 2 → 1 |
| `intfloat/multilingual-e5-base`     | 0.4149     | 0.2310            | 0.4961     | 3 → 3 |
| `intfloat/multilingual-e5-large`    | 0.4063     | 0.1791            | 0.4926     | 4 → 4 |

`multilingual-e5-large` loses to its own base model: the better
document-to-document encoder (kNN 0.3193 against 0.3023) and the worse
document-to-label one (dense 0.0852 against 0.1149), and the label tower is what
reaches the tail. **The order did not survive the index growing** — Spearman ρ
0.800, Kendall τ-b 0.667 — so cheap screening generalises for the shortlist's
membership, not its order: the top two lead the third by 0.0336 and separate
from each other by 0.0039. Both carried forward.

### 4. Full corpus, then the GPU

Both survivors at the 70,579-document all-subjects index, each off the shelf and
contrastively fine-tuned, so corpus and training are separable:

| encoder                             | rung 2 | rung 3, off the shelf | rung 3, fine-tuned | index adds | training adds |
| ----------------------------------- | ------ | --------------------- | ------------------ | ---------- | ------------- |
| `Alibaba-NLP/gte-multilingual-base` | 0.5298 | 0.5376                | **0.6044**         | +0.0079    | **+0.0668**   |
| `BAAI/bge-m3`                       | 0.5337 | 0.5411                | 0.5909             | +0.0073    | +0.0498       |

More documents stopped paying and training started: 2.2× the corpus for +0.008,
against two LoRA runs of 87 and 197 GPU-minutes for +0.07 and +0.05. The
untrained order flipped again — `gte` starts 0.0035 behind and finishes 0.0135
ahead, on 2.9M trainable parameters against 7.1M.

### 5. The test split, opened once

Fine-tuned `gte-multilingual-base`, three fused retrievers, cross-encoder over
the 100 candidates, cut to 50. Scores are in "Result" above; where they come
from is here:

| gold labels on the record | records | share | micro R@10 |
| ------------------------- | ------- | ----- | ---------- |
| 1                         | 1,671   | 34.1% | **0.7367** |
| 2                         | 1,544   | 31.5% | 0.6992     |
| 3                         | 830     | 16.9% | 0.6048     |
| 4                         | 403     | 8.2%  | 0.5273     |
| 5 or more                 | 456     | 9.3%  | 0.4163     |

Counted over the 4,904 records that reached the submission tree. A third of
the split carries one gold heading, and that third is where the
recall figure lives.

## Where the tail goes

Bands are frozen in `reference/frequency_bands.json` before any run reads them,
so growing an index cannot reclassify what counts as tail:

| band      | train occurrences | labels                 | share of test assignments | test R@10  |
| --------- | ----------------- | ---------------------- | ------------------------- | ---------- |
| head      | more than 100     | 65                     | 16.0%                     | 0.7215     |
| torso     | 10 to 100         | 1,543                  | 44.5%                     | 0.6297     |
| tail      | 1 to 9            | 12,999                 | 30.7%                     | 0.5353     |
| zero-shot | never seen        | rest of the vocabulary | 8.9%                      | **0.3547** |

The smallest band, followed through the pipeline, is the shortest description of
what the rebuild did:

| where                            | zero-shot R@10 | why                                                 |
| -------------------------------- | -------------- | --------------------------------------------------- |
| the 2024 classifier              | 0.0000         | structural: no output column exists for the label   |
| kNN alone, dev                   | 0.0000         | structural: no document carries it                  |
| label tower alone, dev           | 0.2623         | reachable by having a name                          |
| three retrievers fused, dev      | 0.1871         | demoted by weights tuned on the aggregate           |
| the 100 fused candidates, dev    | 0.4288         | what the reranker has to work with                  |
| after fine-tuning the tower, dev | 0.2167         | training **costs** this band (−0.023)               |
| the final run, test              | **0.3547**     | the cross-encoder buys back more than training sold |

Dev and test rows differ by split and the last two by a stage, so read the column
as a direction, not a controlled series. Three things it says:

- **Fine-tuning is monotonic in label frequency, and the sign flips at the end.**
  Head +0.10 to +0.13, torso +0.08 to +0.09, tail +0.02 to +0.03, zero-shot
  −0.056 and −0.023. Every training pair is a document and a label some record
  carries, so the tower fits the 14,607 seen labels and drags the other 64,820
  around without ever pulling them anywhere.
- **Mechanisms right about different records get fused, not swapped.** Fusing
  three retrievers beats the best of them by +0.0942; letting the cross-encoder
  _replace_ the fused ranking loses 0.06 (+0.15 zero-shot, −0.28 head), while
  fusing its order with the one it was handed wins +0.034.
- **Some of the tail is out of reach of the whole method family** — the 812
  headings above, at 0.323 R@10 here and 0.000 for anything ranking by document
  similarity.

## The pipeline

`predict(records, config, vocabulary, index_records, store)` is the seam:
records in, 50 ranked GND codes out, everything else reachable only through
configuration. `records` and `index_records` are separate arguments so that
nothing being predicted can contribute its own gold subjects to its own
candidate set.

| stage         | what it does                                          | measured                                         |
| ------------- | ----------------------------------------------------- | ------------------------------------------------ |
| `label_text`  | renders each entry as field-marked bilingual text     | German-only reported as an ablation              |
| `knn`         | summed similarity of the neighbours carrying a code   | 0.3023 dev R@10 alone                            |
| `dense`       | document against all 79,427 label vectors             | the only retriever reaching unseen labels        |
| `lexical`     | BM25 over label strings                               | identical across encoders — the screen's control |
| `fusion`      | reciprocal rank fusion, 100 candidates out            | +0.0942 over the best single retriever           |
| `reranker`    | cross-encoder, `fuse` mix, top 50                     | +0.034 R@10, +0.013 P@5                          |
| `adjudicator` | LLM reorders the top 30 for the least-confident fifth | ceiling +0.046 R@10                              |
| `evaluator`   | micro and official macro side by side                 | 1.1e-16 from the organizers' script              |
| `submission`  | the organizers' `<Type>/<lang>/<id>.json` tree        | 50 ranked codes per record                       |

Label text is bilingual because every vocabulary entry is German and 58.7% of
gold assignments belong to English documents. The English is looked up from
`reference/label_translations.json`, never produced at run time:

```
Fachgebiet: Theoretische und Physikalische Chemie / Theoretical and Physical Chemistry
Schlagwort: Polymere / Polymers
Synonyme: Makropolymere; Hochpolymere; Polymer
```

The adjudicator is shown a numbered list and its answer is checked against
exactly that list — a model asked for GND codes freely invents plausible ones —
and a response naming anything else is rejected whole and logged. On dev it
routes 1,070 of 5,354 records, which are the hard ones: 0.0850 P@5 against the
split's 0.1567.

Confidence is read off the fused ranking, where it correlates +0.41 with
per-record P@5 (over the reranker's own relevance: −0.05, whose most confident
decile has 66.7% of records with no correct label at all). Declining the
least-confident records is outside the official metric, which has no
representation for abstention:

| dev coverage | P@5    | records with no hit in the top 5 | gold assignments still answered |
| ------------ | ------ | -------------------------------- | ------------------------------- |
| 100%         | 0.1567 | 39.2%                            | 100%                            |
| 50%          | 0.1991 | 26.7%                            | 53.4%                           |
| 10%          | 0.2627 | —                                | —                               |

## Evaluation

| aggregation               | weighting                                    | used for                                    |
| ------------------------- | -------------------------------------------- | ------------------------------------------- |
| micro                     | every gold assignment once                   | model selection                             |
| official macro-over-cells | one vote per `<record type> × language` cell | the headline, comparable to the leaderboard |

The organizers' arithmetic is reproduced exactly, rounding included;
`tests/test_evaluator.py` runs their script over a committed fixture and asserts
agreement at every k from 5 to 50, on every cell and both other sheets. The two
aggregations diverge because the cells are extreme: on test, 9 of 20 carry 52.5%
of the macro figure and seven of those hold nine records or fewer. Eleven of the
twenty are cells the organizers' own reader raises `KeyError` on (it seeds `de`
and `en` only), so the like-for-like row against the leaderboard is the 9-cell
one — which is the 0.6299 quoted above.

## Data

No dataset file is tracked in git: `TIBKAT_dataset/*.csv`, `GND_dataset/*.json`,
the `.cache/` clone and `artifacts/` are all ignored. Two commands fetch and
build everything from the official release, sparse-cloning
[https://github.com/jd-coderepos/llms4subjects](https://github.com/jd-coderepos/llms4subjects) into `.cache/` on first run:

```bash
python build_tibkat_csv.py                        # tib-core records + GND vocabulary
python build_tibkat_csv.py --subset all-subjects  # the second track, needed for rung 3
python scripts/verify_split_alignment.py          # attest both corpora before indexing
```

| written                                                                             | where             | size   |
| ----------------------------------------------------------------------------------- | ----------------- | ------ |
| `core_{train,dev,test}.csv`, `all_{train,dev,test}.csv`                             | `TIBKAT_dataset/` | 195 MB |
| `GND-Subjects-{tib-core,all}.json`, name-qualifier sidecars, `qualifier_stats.json` | `GND_dataset/`    | 82 MB  |
| sparse clone of the release                                                         | `.cache/`         | 1.9 GB |

The splits:

| file                     | records | unique labels | assignments |
| ------------------------ | ------- | ------------- | ----------- |
| `core_train.csv`         | 32,043  | 14,607        | 78,037      |
| `core_dev.csv`           | 5,354   | 5,563         | 13,085      |
| `core_test.csv`          | 4,910   | 5,189         | 11,798      |
| `all_train.csv` (rung 3) | 70,588  | —             | —           |

- `all_train.csv` is 70,633 rows under 70,588 records: 45 documents filed twice,
  20 of them with differing gold, merged into one carrying the union.
- **None of the 4,910 gold test records is in** `all_train`**.** Nine `core_dev`
  records are, so they are named in `reference/split_alignment.json` and dropped
  from every index — **70,579 documents indexed**.
- 19.1% of test labels never occur in train.
- The earlier CSVs were contaminated — 142 training records were also in the
  gold test set, 8,966 titles came from the wrong track — and were removed on
  2026-09-07. Numbers measured on them are not comparable; `legacy/` keeps their
  statistics and branch `initial_submission` keeps the files.

### Vocabulary

GND disambiguates homographs with a qualifier (`Interaktion,Naturwissenschaft`
against `Interaktion,Soziologie`). The files used until 2026-09-07 had them
stripped, merging distinct senses; the rebuild restores them and recovers 4,741
tib-core name boundaries the release itself flattens.

| mode            | rendering                 | notes                                     |
| --------------- | ------------------------- | ----------------------------------------- |
| `parenthetical` | `Verlegung (Ortswechsel)` | default; reads as language to an encoder  |
| `raw`           | `Verlegung Ortswechsel`   | the release string, untouched             |
| `stripped`      | `Verlegung`               | the pre-rebuild form, kept as an ablation |

| vocabulary | entries | with a qualifier | qualified terms | name   | synonym | related |
| ---------- | ------- | ---------------- | --------------- | ------ | ------- | ------- |
| tib-core   | 79,427  | 17,959 (22.6%)   | 22,068          | 4,746  | 12,463  | 4,859   |
| all        | 204,739 | 61,557 (30.1%)   | 77,583          | 24,772 | 40,600  | 12,211  |

## What the code refuses

Most of the project's discipline is a refusal rather than a convention:

| refusal                                                        | why                                                              |
| -------------------------------------------------------------- | ---------------------------------------------------------------- |
| `run_experiment.py --split core_test`                          | the gold split is opened once, by one script                     |
| a test row whose config moved since `reference/test_plan.json` | the plan is digested and committed before the split is read      |
| a second read of `core_test` without a justification           | `reference/test_run.json` is the receipt                         |
| indexing a corpus `verify_split_alignment.py` has not attested | contamination is checked, not assumed                            |
| a record contributing its own gold to its own candidates       | `tests/test_pipeline.py` perturbs gold and asserts nothing moves |
| a model the registry does not date before 2025-01-31           | checked before any weights are fetched                           |
| an adjudicator response naming a code outside the list shown   | models invent plausible GND identifiers                          |
| predictions outside the tib-core vocabulary                    | a rung-3 index must not widen the label universe                 |
| overwriting `reference/frequency_bands.json` without `--force` | bands must not drift between rungs                               |
| two screen configs differing in more than the encoder          | otherwise the screen ranks a retuning                            |

## Repository layout

```
llms4subjects/          the pipeline, one module per stage under stages/
  config.py             experiment configuration, loaded from configs/*.yaml
  contracts.py          the artifacts stages pass between each other
  corpus.py             the only module that reads the dataset
  splits.py             what may be indexed, and the held-out records dropped
  artifacts.py          the cached-artifact convention
  finetune.py           contrastive fine-tuning and the adapter it leaves
  hardware.py           device selection, and what to say when CUDA is absent
  pipeline.py           predict(...) — the seam tests assert against
  stages/               label_text, encoders, indexes, retrievers, fusion,
                        group_prior, reranker, adjudicator, evaluator, submission
baseline/               the rejected 2024 classifier, kept runnable as a results row
configs/                one committed YAML per rung of the experiment ladder
reference/              frozen artifacts: bands, model releases, translations, receipts
official_eval/          the organizers' scorer, unmodified
scripts/                the harnesses every table above came from
tests/                  contract and invariant tests
```

Stages take inputs as arguments and return named artifacts; none reads the
dataset or another stage's files. Everything except training runs on Apple
Silicon without CUDA — only the two rung-3 fine-tunes and the baseline re-run go
to the GPU host, which receives a negatives file and returns an adapter.

`baseline/` is the 2024 approach: `bert-base-multilingual-cased` with a dense
output layer over the 14,607 training labels. Its zero-shot recall is exactly
zero and `tests/test_baseline_classifier.py` asserts it with random scores — no
training run can change it. The original's graph component is not revived: it
modelled a hierarchy the vocabulary does not contain (zero `skos:broader`
triples), and its output was a batch-constant vector that could only shift the
head's bias.

## Commands

```bash
# setup — Mac, no CUDA
uv venv && uv pip install -r requirements/mac.txt -r requirements/dev.txt

# dataset: records and vocabulary, from the official release
python build_tibkat_csv.py
python build_tibkat_csv.py --subset all-subjects
python scripts/verify_split_alignment.py          # which corpora are clear to index

# run and score
python -m llms4subjects configs/rung2.yaml        # resolve a config: keys, device, cache hits
python scripts/run_experiment.py configs/rung1-knn.yaml
python scripts/run_experiment.py configs/rung1.yaml --limit 500 --submission out/

# the tables above, each from its own harness
python scripts/ablate_retrievers.py configs/rung1.yaml --tune-weights
python scripts/screen_encoders.py configs/rung1.yaml configs/rung1-e5-large.yaml \
    configs/rung1-bge-m3.yaml configs/rung1-gte-base.yaml
python scripts/compare_rungs.py reference/screens/rung1.json reference/screens/rung2.json
python scripts/rerank_report.py configs/rung1-rerank.yaml --sample 300
python scripts/adjudicate_report.py configs/rung1-adjudicate.yaml --dry-run
python scripts/coverage_curve.py configs/rung1.yaml --figure artifacts/coverage/rung1.png
python scripts/zero_shot_bound.py configs/test.yaml --split core_dev

# fine-tuning: mine on the Mac, train on the GPU host, score back here
python scripts/mine_hard_negatives.py configs/rung2-gte-base.yaml
ssh nlp2 '... scripts/train_encoder.py configs/rung3-gte-base.yaml \
    --negatives-from configs/rung2-gte-base.yaml'
python scripts/rung3_report.py --pair configs/rung3-untrained.yaml configs/rung3.yaml \
    --against reference/screens/rung2.json

# the test split, once
python scripts/final_test.py --rehearse           # the same harness, on dev
python scripts/final_test.py --fix-plan           # commit the plan digest first
python scripts/final_test.py --json artifacts/test/run.json
python scripts/zero_shot_bound.py configs/test.yaml --split core_test \
    --predictions artifacts/test/headline/submission

# the 2024 baseline, kept runnable
python -m baseline.train --smoke
python -m baseline.score artifacts/baseline/mbert-dense/predictions-core_dev.json

# checks
python scripts/verify_model_releases.py --offline # models against the 2025-01-31 cutoff
python scripts/freeze_bands.py                    # report the bands, refuse to overwrite
python scripts/translate_labels.py --report       # translation cache coverage
pytest
```
