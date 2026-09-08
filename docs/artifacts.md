# Artifacts, configuration and the two hosts

## Artifact directory convention

Every stage writes its output under:

    artifacts/<stage>/<key>/

`<key>` is a 12-character fingerprint of **only** the configuration sections that
stage's output actually depends on, plus the dataset revision. The mapping from
stage to sections is `STAGE_DEPENDENCIES` in
[`llms4subjects/artifacts.py`](../llms4subjects/artifacts.py), and it is what
makes the cache useful: swapping the reranker leaves every index and candidate
key untouched, so nothing is recomputed to answer a question it cannot affect.

Each directory also holds a `manifest.json` recording exactly what its key was
computed from, so an artifact found on disk months later still explains itself.

```python
from llms4subjects.artifacts import ArtifactStore
from llms4subjects.config import load_experiment
from llms4subjects.corpus import data_revision

store = ArtifactStore("artifacts", data_revision=data_revision())
config = load_experiment("configs/rung2.yaml")

store.path("document_index", config, "vectors.npy")
# artifacts/document_index/2c727d0b135f/vectors.npy — nothing created

if not store.exists("document_index", config, "vectors.npy"):
    store.prepare("document_index", config)   # mkdir + manifest, for writers
```

One stage of that table is not a stage of the pipeline but a cache of the most
expensive thing in it. `embeddings` depends on the encoder section alone, and
`ArtifactStore` is wrapped by `CachedEncoder`, which files one matrix per set of
texts under a digest of those texts:

```python
from llms4subjects.artifacts import CachedEncoder

encoder = CachedEncoder(encoder, store, config)   # same Encoder interface
```

Because the wrapper sits at the encoder rather than at the index builder, the
index corpus, the records being predicted and the label tower all reach the same
cache, and an ablation that changes anything downstream of the encoder re-encodes
nothing at all: scoring dev costs 106s the first time and 9s after that.

The label tower goes through the same door, which is what keys it by encoder and
by label-text revision without a second mechanism: the 79,427 rendered label
strings are the texts, so switching `label_text.qualifiers` or turning
`include_definition` on re-encodes the tower while leaving every document vector
in place, and a rung that changes only the index re-encodes neither. Encoding
the 79,427 label texts costs about two minutes on the M4 Pro, once per
rendering.

The digest is over the texts themselves, so adding a document to the index or
editing an abstract produces a different file rather than a stale hit. One file
holds one whole matrix, which makes the unit of reuse the exact set of texts: an
8,000-document sample shares nothing with the 32,043-document index it was drawn
from, because a per-text cache would spend more on stat calls than it saves.

The `reranked` stage is cached the same way, by `scripts/rerank_report.py`
rather than by the pipeline: one JSON file per split holding the reranked lists
with their scores and provenance, keyed by every section from the label text
through the reranker. Reranking dev costs a cross-encoder forward pass per
candidate — 535,400 of them at `input_k: 100` — so the confidence calibration
and the band tables read that file rather than paying for the pass again.
`--refresh` recomputes it.

The `adjudicated` stage holds the one artifact in this project that costs money
rather than time: `responses.jsonl`, one line per record with the raw text the
language model returned, keyed by every section including the model's pinned id,
the prompt revision and every knob that changes a prompt. It is appended to as
each response arrives rather than written at the end, so a run interrupted at
record 900 of 1,070 keeps the 900 it has already paid for, and a re-score of a
finished run bills nothing. `rejections.jsonl` sits beside it: one line per
response that named a code it was not offered, with the reason, so a constraint
violation leaves evidence rather than a silently unchanged ranking.

Asking for a path creates nothing, so a cache miss stays a miss; only
`prepare` writes. That asymmetry matters more than it looks: a `path()` that
created its directory would make the next run see a hit for an artifact that
was never produced.

To see what a run will read and write before it starts:

    python -m llms4subjects configs/rung2.yaml

## Frozen reference artifacts

`reference/` is the opposite of `artifacts/`: small, tracked, and never written
as a side effect of a run. It holds three files:

| file | what it is | written by |
|---|---|---|
| `frequency_bands.json` | the label frequency band assignment, frozen against tib-core train counts | `scripts/freeze_bands.py --force` |
| `label_translations.json` | English for all 79,224 distinct German label strings | `scripts/translate_labels.py` |
| `model_releases.json` | every model the project loads, its release date and the revision it is pinned to | `scripts/verify_model_releases.py --force` |

The distinction is the point. A cached artifact is a saved computation and can
be deleted at any time; a frozen reference is a *decision*, and recomputing it
would change the meaning of every results table that cites it. Nothing but the
script in the right-hand column writes any of the three, so a change of
reference leaves a commit rather than happening as a by-product of a run.

For the bands, that by-product would be adding data to the index:

    python scripts/freeze_bands.py            # report the bands, refuse to overwrite
    python scripts/freeze_bands.py --force    # rewrite the reference

`scripts/build_eval_fixture.py --force` is the same shape for the committed
evaluation fixture in `tests/fixtures/`, which pins the local evaluator to the
organizers' scorer.

### The translation cache

All 79,427 vocabulary entries are German and 58.7% of gold label assignments
belong to English documents, so `label_text.bilingual` renders each label's
preferred name and classification group in both languages. The English comes
from `reference/label_translations.json`, and the point of committing it is that
**translation costs a run nothing**: the pipeline looks strings up in a dict,
and no module on the rendering path can even import a translation model —
`tests/test_label_translations.py` asserts that against the source.

    python scripts/translate_labels.py --report   # coverage, writes nothing
    python scripts/translate_labels.py            # translate whatever is missing
    python scripts/translate_labels.py --force    # re-translate every string

It is a German-string to English-string map rather than a code-to-name one, so
the two senses of `Interaktion` share one translation of the word they have in
common while keeping the qualifier that tells them apart, and 79,427 entries
reduce to 79,224 strings. Which strings the cache must cover is
`label_text.translatable`, decided by the module that renders them rather than
by the script that fills them.

The default run translates only what is missing and writes keys sorted, so a
second run is a no-op and produces a byte-identical file. `produced_by` records
the model — `Helsinki-NLP/opus-mt-de-en`, 2020, inside the declared cutoff — and
nothing reads it: the cache is an input to the pipeline, not a dependency on the
model that wrote it. Filling it from scratch takes about two minutes on the M4
Pro, and `sentencepiece` in `requirements/base.txt` is there for this script
alone.

### The model registry

docs/spec.md restricts every component to models released on or before
2025-01-31, the close of the shared task's evaluation window. A model name does
not carry that claim: `intfloat/multilingual-e5-base` was created in May 2023
and had a commit landed on it in April 2026, so naming it without a revision
loads weights the cutoff never covered while every date in the sentence stays
true.

`reference/model_releases.json` is where the claim lives. Each entry records
when the repository was created, the newest commit dated on or before the
cutoff, and — for a model that ships its own modelling code — the repository and
commit of that code, since loading it executes it.

    python scripts/verify_model_releases.py            # re-check against the hub
    python scripts/verify_model_releases.py --offline   # check the file alone
    python scripts/verify_model_releases.py --force     # rewrite the reference

`llms4subjects.models.check_cutoff` reads it, and the two pipeline loaders that
fetch weights — `stages.encoders.resolve` and `stages.reranker.resolve` — call
it before any download, so a model the registry does not vouch for fails in a
sentence rather than after a gigabyte. Those two are also where a config
becomes the pinned revision actually loaded; `encoders.pinned` puts that
revision into the artifact key, so cached vectors are served only to the
weights that produced them.

Two paths are recorded in the registry without reading it. The rejected
baseline (`baseline/classifier.py`) loads `bert-base-multilingual-cased`
unpinned — it predates the registry, it is a results row rather than a
dependency, and re-running it needs the GPU host — and the translation cache is
written once by a script rather than by a run. Both are entries in the file, so
the writeup's model list is complete; gating the baseline is a loose end rather
than a claim.

Nothing else in the project reaches the hub API, and no run reads it:
re-verification is the deliberate command above.

`artifacts/` is ignored by git, along with the dataset itself
(`TIBKAT_dataset/*.csv`, `GND_dataset/*.json`) and the sparse clone the rebuild
fetches into `.cache/`. Nothing large is tracked; everything is rebuildable.

## Experiment configuration

A rung of the experiment ladder is a committed YAML file in
[`configs/`](../configs), never an edit to a script:

| file | index | models |
|---|---|---|
| `configs/rung1.yaml` | 8,000 documents, stratified | `multilingual-e5-base`, off the shelf |
| `configs/rung1-e5-large.yaml` | the same 8,000 | `multilingual-e5-large` — a screened encoder |
| `configs/rung1-bge-m3.yaml` | the same 8,000 | `bge-m3` — a screened encoder |
| `configs/rung1-gte-base.yaml` | the same 8,000 | `gte-multilingual-base` — a screened encoder |
| `configs/rung1-knn.yaml` | the same 8,000 | the neighbour retriever alone |
| `configs/rung1-dense.yaml` | unread — the label tower scores the vocabulary | the dense label retriever alone |
| `configs/rung1-dense-german.yaml` | the same | the same, label text German-only |
| `configs/rung1-lexical.yaml` | unread — BM25 over the labels' own strings | the lexical retriever alone |
| `configs/rung1-lexical-german.yaml` | the same | the same, label text German-only |
| `configs/rung1-rerank.yaml` | the same 8,000 | plus `bge-reranker-v2-m3` over the fused candidates |
| `configs/rung1-rerank-base.yaml` | the same 8,000 | plus `bge-reranker-base` — the smaller screened reranker |
| `configs/rung1-prior.yaml` | the same 8,000 | plus the 66-way classification-group prior |
| `configs/rung1-adjudicate.yaml` | the same 8,000 | plus `claude-3-5-sonnet-20241022` over the least-confident fifth |
| `configs/rung2.yaml` | 32,043 documents (tib-core train) | off-the-shelf encoder |
| `configs/rung3.yaml` | 70,588 documents (all-subjects train) | fine-tuned adapter, full pipeline |

The loader ([`llms4subjects/config.py`](../llms4subjects/config.py)) rejects
unknown keys rather than ignoring them, because a misspelled ablation flag
would otherwise read as "off" and the run would look like a measurement of
something it is not. Ablations are new files, not edits: copy a rung, change
one flag, keep both.

### The group-prior head

`group_prior.enabled` needs a fitted head, and the pipeline reads it from the
artifact store rather than accepting one from the caller — a harness that forgot
to pass it would score an unboosted run under a boosted config's name. Fit it
first:

    python scripts/train_group_prior.py configs/rung1-prior.yaml            # fit and score
    python scripts/train_group_prior.py configs/rung1-prior.yaml --report    # score, write nothing
    python scripts/train_group_prior.py configs/rung1-prior.yaml --force     # refit over an existing head

It writes `artifacts/group_prior/<key>/head.npz` — the per-group coefficients,
the intercepts and the group order they are in. Two arrays and a list of names
rather than a pickled estimator, so the artifact outlives the scikit-learn
version that fitted it, and a run months later reads it back.

The key is the encoder, index and group-prior sections. The encoder is in it
because the head is a linear layer over that encoder's document vectors and
means nothing over another's. The whole group-prior section is in it, including
the `weight` that only inference reads, because re-fitting is 11 s once the
vectors are cached and a narrower key would be a second mechanism to maintain.
Training reads every document of `index.corpora` and ignores `index.size`; see
[results.md](results.md) for why that is worth 9.7 points of accuracy.

An enabled prior with no fitted head raises `MissingGroupPrior` naming the
command above, rather than quietly skipping the boost.

## The two environments

| environment | file | what runs there |
|---|---|---|
| Local — Apple Silicon, no CUDA | `requirements/mac.txt` | everything except training: dataset rebuild, indexing, retrieval, fusion, reranking, adjudication, evaluation, submission |
| GPU — the `nlp2` host | `requirements/gpu.txt` | three training runs only: two rung-3 fine-tunes and the baseline classifier re-run |

Local:

    uv venv && uv pip install -r requirements/mac.txt -r requirements/dev.txt

The GPU host has `uv` but no system `pip`, and `uv` defaults to
first-index-match, which will not find torch's `+cu124` local version on PyPI:

    ssh nlp2
    cd ~/projects/llms4subjects
    uv venv --python 3.12 .venv
    uv pip install --index-strategy unsafe-best-match -r requirements/gpu.txt

With plain `pip`, `pip install -r requirements/gpu.txt` is enough.

No local stage requires `nlp2` to be reachable. The only inputs that come from
it are fine-tuned adapters, and a config naming one fails on a missing path
rather than by reaching over the network.

## Working with the GPU host

Send code, not data. The dataset is gitignored and rebuilt on the host with the
same one command used locally, so there is no hand-copied snapshot to go stale:

    # from the repo root, code only
    rsync -az --delete \
      --exclude .venv --exclude .git --exclude artifacts \
      --exclude TIBKAT_dataset --exclude GND_dataset \
      --exclude __pycache__ --exclude .cache \
      ./ nlp2:~/projects/llms4subjects/

    # on the host, rebuild the dataset from the official release
    ssh nlp2 'cd ~/projects/llms4subjects && .venv/bin/python build_tibkat_csv.py'

The rebuild must report the same record and label counts as the local one; that
equality is what lets a checkpoint trained there be scored here.

Bring training artifacts back the same way, into the gitignored artifact tree:

    rsync -az nlp2:~/projects/llms4subjects/artifacts/adapters/ artifacts/adapters/
    rsync -az nlp2:~/projects/llms4subjects/artifacts/baseline/ artifacts/baseline/

Scoring then happens locally through the shared evaluator, so every number in
the results table comes from one code path regardless of where the model was
trained.

The baseline re-run is the worked example of that split. On the host:

    ssh nlp2 'cd ~/projects/llms4subjects && \
      .venv/bin/python -u -m baseline.train --epochs 15 --device cuda \
      > artifacts/baseline/mbert-dense/train.log 2>&1'

It writes `artifacts/baseline/<run>/`, which is a plain run directory rather than
a keyed artifact: it has no `ExperimentConfig` to fingerprint, since the rejected
classifier is not a rung of the ladder.

| file | what it is |
|---|---|
| `labels.json` | the output layer in column order — part of the checkpoint, not a derivation of it |
| `checkpoint.pt` | the trained weights |
| `run.json` | configuration, per-epoch losses, wall clock, device, host |
| `predictions-<split>.json` | 50 ranked codes per record |
| `train.log` | the run's console output |

Then, locally:

    rsync -az nlp2:~/projects/llms4subjects/artifacts/baseline/ artifacts/baseline/
    python -m baseline.score artifacts/baseline/mbert-dense/predictions-core_dev.json --markdown

Nothing on the host computes a metric. `--markdown` prints the rows that go into
[docs/results.md](results.md), so the results document is a paste rather than a
transcription.
