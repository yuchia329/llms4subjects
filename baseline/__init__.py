"""The rejected classifier, kept runnable as a results row.

This is the approach docs/spec.md rejects: a dense output layer over the
training label set, which cannot reach the 19.1% of test labels that never
appear in training. Its numbers belong in the results table as the reference
point (ticket 14), so the code stays executable against the clean splits.

Nothing in `llms4subjects/` imports this package.

Known defects in this code, found while moving it and left as they are because
the numbers it produces have to come from the classifier that was actually run,
not from a repaired version of it. Ticket 14 has to decide which of these to
fix before its measurement counts:

- `label_metadata.get_subject_metadata` tests `category_subject_mapping.get(label)`
  but writes under `label_metadata["Classification Name"]`, so each
  classification group ends up holding exactly one label and the `else` branch
  is unreachable. The label graph in `train.py` is built from that mapping.
- `data_handler.load_dev_data_one_hot` orders its columns by iterating a set,
  so dev label indices vary between processes and do not line up with train.
- `train.py` never saves a checkpoint, and `eval_bert_rebota.py` expects one;
  the two were run by hand, in that order, with the save commented out.
- `eval_bert_rebota.py` takes `topk` over 768-wide CLS embeddings as though
  they were scores over the 14,607 labels, so its indices are not label ids.
- The graph component is not revived by the current design: it modelled a
  hierarchy the vocabulary does not contain (zero `skos:broader` triples).
"""
