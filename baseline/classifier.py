"""mBERT with a dense output layer over the training labels. No label graph.

The model the project rejects, stated in as few moving parts as the rejection
allows: `bert-base-multilingual-cased`, CLS pooling, one `Linear` to 14,607
columns, `BCEWithLogitsLoss`. Everything the original carried beyond that has
been removed rather than repaired, because each removal is either a defect
(`baseline/__init__.py` lists them) or a component the current design excludes.

What is gone, and why:

- **The GCN label-graph refiner.** Its edges came from
  `label_metadata.get_subject_metadata`, whose `category_subject_mapping` was
  keyed by classification name but tested by label, so each group was
  overwritten down to a single label: over the 14,607 train labels, 65
  classification names produced 65 two-node components covering 130 of them,
  and the other 14,477 labels had no edge at all. Repairing the keying would
  have built a real graph over classification groups — and
  docs/spec.md excludes reviving it regardless, since it modelled a hierarchy
  the vocabulary does not contain (zero `skos:broader` triples). Its output was
  in any case a batch-constant vector — the mean over all label embeddings,
  repeated across the batch — so it could shift the head's bias and nothing else.
- **The train/val/test resplit of the training file.** The original split
  `core_train` three ways with `train_test_split`, which measured against a
  slice of train. The clean official splits exist now: this trains on
  `core_train` and validates on `core_dev`.
- **`compute_metrics`.** Metrics come from `llms4subjects.stages.evaluator`, so
  the baseline row is produced by the same code path as every pipeline result.

The loss is the only place the closed vocabulary is visible during training:
targets are built from `baseline.labels.target_rows`, which drops codes with no
column, so the assignments this approach cannot reach are absent from the
gradient as well as from the prediction.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from llms4subjects.contracts import CODES_PER_RECORD, Code, Record

from .labels import rank_codes, target_rows

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 256

# Raised either way round: a loader holding more rows than there are ids, or
# fewer. Both mean the ranking is being filed under the wrong record.
_MISALIGNED = (
    "the loader and the {0} record ids are not the same length; a ranking "
    "would be filed under the wrong record"
)


class SubjectDataset(Dataset):
    """Records as token ids, with their gold columns as a multi-hot target.

    The target is built per item rather than held as a matrix: 32,043 records by
    14,607 columns is 1.9 GB dense, and every row has around three ones in it.
    The original reached for `scipy.sparse` to survive the same problem; a
    sparse row per item and a dense tensor per batch is the same fix with one
    fewer dependency.
    """

    def __init__(
        self,
        records: Sequence[Record],
        codes: Sequence[Code],
        tokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.texts = [record.text for record in records]
        self.targets = target_rows(records, codes)
        self.num_labels = len(codes)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        encoding = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = torch.zeros(self.num_labels, dtype=torch.float32)
        columns = self.targets[index]
        if columns:
            target[list(columns)] = 1.0
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": target,
        }


class SubjectClassifier(nn.Module):
    """A transformer encoder and one dense layer over the training labels."""

    def __init__(self, text_model, num_labels: int):
        super().__init__()
        self.text_model = text_model
        self.classifier = nn.Linear(text_model.config.hidden_size, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None) -> dict:
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def build_model(num_labels: int, model_name: str = MODEL_NAME):
    """The classifier over `num_labels` columns, on freshly pretrained weights."""
    from transformers import AutoModel

    return SubjectClassifier(AutoModel.from_pretrained(model_name), num_labels)


def build_tokenizer(model_name: str = MODEL_NAME):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def train_epoch(model, loader: DataLoader, optimizer, device) -> float:
    """One pass over the training split, returning its mean batch loss."""
    model.train()
    total = 0.0
    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        loss = outputs["loss"]
        total += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total / len(loader) if len(loader) else 0.0


@torch.no_grad()
def split_loss(model, loader: DataLoader, device) -> float:
    """Mean loss over a split, for watching the run rather than scoring it."""
    model.eval()
    total = 0.0
    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        total += outputs["loss"].item()
    return total / len(loader) if len(loader) else 0.0


def train(
    model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer,
    device,
    epochs: int,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Train for `epochs`, returning one record per epoch for the run summary.

    Wall clock is recorded per epoch as well as for the run, because the point
    of the row this produces is partly what the approach costs: a 14,607-way
    head over 32,043 documents is the reason this ticket needs the GPU host.
    """
    history = []
    for epoch in range(1, epochs + 1):
        started = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_seconds = time.time() - started

        started = time.time()
        dev_loss = split_loss(model, dev_loader, device)
        dev_seconds = time.time() - started

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "seconds": round(train_seconds + dev_seconds, 1),
            }
        )
        log(
            f"epoch {epoch}: train loss {train_loss:.6f} "
            f"({train_seconds:.1f}s), dev loss {dev_loss:.6f} ({dev_seconds:.1f}s)"
        )
    return history


@torch.no_grad()
def rank(
    model,
    loader: DataLoader,
    codes: Sequence[Code],
    device,
    record_ids: Sequence[str],
    k: int = CODES_PER_RECORD,
) -> dict[str, tuple[Code, ...]]:
    """Ranked codes per record, in the order `record_ids` gives them.

    The loader must not shuffle: the ids are zipped onto the batches in order.
    Ranking happens on the host in the same call as the forward pass, so no
    5,354 x 14,607 logit matrix is ever materialised.
    """
    model.eval()
    rankings: dict[str, tuple[Code, ...]] = {}
    position = 0
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )["logits"]
        for row in logits.detach().float().cpu().numpy():
            if position >= len(record_ids):
                raise ValueError(_MISALIGNED.format(len(record_ids)))
            rankings[record_ids[position]] = rank_codes(row, codes, k)
            position += 1
    if position != len(record_ids):
        raise ValueError(_MISALIGNED.format(len(record_ids)))
    return rankings
