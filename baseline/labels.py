"""The classifier's output layer: the training label set, in a frozen order.

A dense head is positional — column 4,211 means one GND code and nothing else —
so the order the layer was built in is part of the checkpoint. It is saved beside
the weights and read back verbatim rather than rebuilt, because a layer rebuilt
differently would re-map every column while still looking like the same model.

This module is also where the approach's ceiling is stated in code: a code with
no column cannot be predicted at any score, so `unreachable_assignments` counts
the gold the layer cannot represent before a single epoch runs.

Fixes two of the defects `baseline/__init__.py` records: the ordering came from
iterating a set, and the dev columns were built independently of the train ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from llms4subjects.contracts import CODES_PER_RECORD, Code, Record


def output_layer(records: Iterable[Record]) -> tuple[Code, ...]:
    """Every code the training split assigns, sorted.

    Sorted rather than in encounter order so that two processes reading the same
    split agree on which column is which. `load_dev_data_one_hot` ordered its
    columns by iterating a `set`, whose order varies between processes for str
    keys, so its dev indices did not line up with the train indices the model
    had been fitted against.
    """
    return tuple(sorted({code for record in records for code in record.subjects}))


def save_labels(path: str | Path, codes: Sequence[Code]) -> Path:
    """Write the layer beside its checkpoint, one JSON array in column order."""
    path = Path(path)
    path.write_text(json.dumps(list(codes), ensure_ascii=False, indent=0))
    return path


def load_labels(path: str | Path) -> tuple[Code, ...]:
    """The layer as saved. Not re-sorted: the saved order is the head's order."""
    return tuple(json.loads(Path(path).read_text()))


def target_rows(
    records: Iterable[Record], codes: Sequence[Code]
) -> list[tuple[int, ...]]:
    """Gold column positions per record, as the loss's sparse targets.

    Codes with no column are dropped, which is the finding rather than a
    convenience: on the dev split those are the assignments the approach cannot
    reach, and `unreachable_assignments` is how many.
    """
    position = {code: index for index, code in enumerate(codes)}
    return [
        tuple(sorted(position[code] for code in record.subjects if code in position))
        for record in records
    ]


def unreachable_assignments(
    records: Iterable[Record], codes: Sequence[Code]
) -> tuple[int, int]:
    """(gold assignments, of those with no column in the layer).

    The ceiling of a closed-vocabulary classifier on a split, computable before
    training and unchanged by it.
    """
    known = set(codes)
    total = 0
    unreachable = 0
    for record in records:
        for code in record.subjects:
            total += 1
            if code not in known:
                unreachable += 1
    return total, unreachable


def rank_codes(
    scores, codes: Sequence[Code], k: int = CODES_PER_RECORD
) -> tuple[Code, ...]:
    """The k highest-scoring codes, best first.

    Takes scores over the layer's columns — logits are enough, since the sigmoid
    the loss applies is monotone and cannot reorder them. Returns fewer than k
    only when the layer itself holds fewer, which the real 14,607-column layer
    never does.

    Replaces `eval_bert_rebota.py`, which took `topk` over 768-wide CLS
    embeddings as though they were scores over the labels, so its indices were
    not label ids at all.
    """
    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) != len(codes):
        raise ValueError(
            f"{len(values)} scores for {len(codes)} columns; the scores must "
            "come from the same output layer as the codes"
        )
    order = np.argsort(-values, kind="stable")[:k]
    return tuple(codes[index] for index in order)
