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

Asking for a path creates nothing, so a cache miss stays a miss; only
`prepare` writes. That asymmetry matters more than it looks: a `path()` that
created its directory would make the next run see a hit for an artifact that
was never produced.

To see what a run will read and write before it starts:

    python -m llms4subjects configs/rung2.yaml

## Frozen reference artifacts

`reference/` is the opposite of `artifacts/`: small, tracked, and never written
as a side effect of a run. It holds `frequency_bands.json`, the label frequency
band assignment frozen against tib-core train counts.

The distinction is the point. A cached artifact is a saved computation and can
be deleted at any time; a frozen reference is a *decision*, and recomputing it
would change the meaning of every results table that cites it. Only
`scripts/freeze_bands.py --force` writes it, so a change of reference leaves a
commit rather than happening as a by-product of adding data to the index:

    python scripts/freeze_bands.py            # report the bands, refuse to overwrite
    python scripts/freeze_bands.py --force    # rewrite the reference

`scripts/build_eval_fixture.py --force` is the same shape for the committed
evaluation fixture in `tests/fixtures/`, which pins the local evaluator to the
organizers' scorer.

`artifacts/` is ignored by git, along with the dataset itself
(`TIBKAT_dataset/*.csv`, `GND_dataset/*.json`) and the sparse clone the rebuild
fetches into `.cache/`. Nothing large is tracked; everything is rebuildable.

## Experiment configuration

A rung of the experiment ladder is a committed YAML file in
[`configs/`](../configs), never an edit to a script:

| file | index | models |
|---|---|---|
| `configs/rung1.yaml` | 8,000 documents, stratified | off-the-shelf encoder |
| `configs/rung2.yaml` | 32,043 documents (tib-core train) | off-the-shelf encoder |
| `configs/rung3.yaml` | 70,588 documents (all-subjects train) | fine-tuned adapter, full pipeline |

The loader ([`llms4subjects/config.py`](../llms4subjects/config.py)) rejects
unknown keys rather than ignoring them, because a misspelled ablation flag
would otherwise read as "off" and the run would look like a measurement of
something it is not. Ablations are new files, not edits: copy a rung, change
one flag, keep both.

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
