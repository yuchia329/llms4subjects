"""The rejected classifier, re-run on the clean splits as a results row.

This is the approach docs/spec.md rejects: a dense output layer over the
training label set, which cannot reach the 19.1% of test labels that never
appear in training. Ticket 14 turns that from an argument into a measurement, so
the code stays executable — against the clean splits, and scored through the
shared evaluator:

    python -m baseline.train --smoke                  # first training step, laptop
    python -m baseline.train --epochs 15              # the real run, on nlp2
    python -m baseline.score artifacts/baseline/mbert-dense/predictions-core_dev.json

Nothing in `llms4subjects/` imports this package. The dependency runs one way:
the baseline reads the dataset through `llms4subjects.corpus` and is scored by
`llms4subjects.stages.evaluator`, so its row comes from the same code path as
every pipeline row.

## The modules

| module | what it holds |
|---|---|
| `labels.py` | the output layer: the train label set in frozen column order |
| `classifier.py` | mBERT, CLS pooling, one dense layer, BCE loss. No graph |
| `train.py` | the run: train, checkpoint, run.json, ranked predictions |
| `score.py` | local scoring through the shared evaluator and frozen bands |

## Defects found while moving the code, and what ticket 14 did about each

The measurement has to come from the classifier that was actually run, not from
a repaired version of it — but a defect that makes a number meaningless is not
a faithful detail, it is a broken measurement. Each was either fixed, or the
component carrying it was dropped for a reason of its own:

- `label_metadata.get_subject_metadata` tested `category_subject_mapping.get(label)`
  and wrote under `label_metadata["Classification Name"]`, so each classification
  group was overwritten down to one label and the `else` branch was unreachable:
  65 classification names over the 14,607 train labels, so the graph the original
  `train.py` built from that mapping was 65 two-node components covering 130
  labels, with the other 14,477 unconnected. **Dropped with the graph** — docs/spec.md excludes reviving it, since it modelled a hierarchy the
  vocabulary does not contain (zero `skos:broader` triples), and its output was a
  batch-constant vector that could only shift the head's bias.
- `data_handler.load_dev_data_one_hot` ordered its columns by iterating a set, so
  dev label indices varied between processes and did not line up with train.
  **Fixed**: `labels.output_layer` builds one sorted layer from `core_train`, saved
  beside the checkpoint and read back verbatim by every split.
- The original resplit `core_train` three ways with `train_test_split` and
  measured against a slice of it. **Fixed**: the clean official splits exist now,
  so it trains on `core_train` and validates on `core_dev`.
- `train.py` never saved a checkpoint and `eval_bert_rebota.py` expected one; the
  two were run by hand, in that order, with the save commented out. **Fixed**: the
  run writes `checkpoint.pt`, `labels.json` and `run.json`.
- `eval_bert_rebota.py` took `topk` over 768-wide CLS embeddings as though they
  were scores over the 14,607 labels, so its indices were not label ids.
  **Replaced** by `labels.rank_codes`, which ranks the head's own columns, and by
  `score.py`, which computes metrics with the shared evaluator instead of a
  private `compute_metrics`.

`train_nocsr.py`, `eval_bert_rebota.py`, `data_handler.py` and
`label_metadata.py` moved to `drafts/` beside the earlier drafts. They remain in
git history; nothing runnable imports them.
"""
