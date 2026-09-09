"""Contrastive fine-tuning of a shortlisted encoder, and the adapter it leaves.

The rung-3 fine-tune teaches one of the two shortlisted encoders the task's own
document-to-label geometry: a record's text is pulled towards the text of each
of its gold headings and pushed away from the headings it is confused with. Two
kinds of negative do the pushing.

- **In-batch.** Every other pair's positive in the same batch, which is free —
  the vectors are computed anyway — and is most of the signal.
- **Mined.** The codes the *untrained* retriever ranked highest for this record
  and got wrong, from `scripts/mine_hard_negatives.py`. They are the confusions
  the model actually makes, and they are mined once on the Mac from indexes that
  already exist rather than recomputed on the GPU host.

Both kinds have the same hazard, and it is the reason half of this module is
bookkeeping: a "negative" that is in fact a gold heading for the record teaches
the model that a correct answer is wrong. A record carries 2.4 headings on
average and the vocabulary is long-tailed, so this is not hypothetical. It is
handled in three places — `hard_negatives` drops the record's own gold,
`batches` refuses to put one code in a batch twice, and `false_negative_mask`
removes what is left, which is another record's gold appearing as a column.

Training is the one thing in this project that requires CUDA, and it is
parameter-efficient: LoRA adapters over the attention and feed-forward
projections, so what comes back from the host is a few megabytes of deltas
rather than a checkpoint. `stages.encoders.load` applies them, and every index
and every metric is computed afterwards on Apple Silicon.

Outside `stages/` on purpose: the stage package is the prediction pipeline's
module boundaries (docs/spec.md), and training is not one of them. It sits
beside `corpus` and `splits`, which are the other things a run reads rather than
runs.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Iterator, Mapping, Sequence

from .contracts import Code, Record

# The peft artifact this project adds to what peft itself writes: which weights
# the deltas belong on, what they were trained over, and what it cost.
ADAPTER_MANIFEST = "training.json"

MANIFEST_SCHEMA = 1

TRAIN_HINT = (
    "Adapters are produced by `python scripts/train_encoder.py <config>` on the "
    "GPU host and copied back with the rsync in docs/artifacts.md."
)

# Which projections LoRA adapts, per architecture. A table rather than a default
# for the reason `encoders.PREFIXES` is one: a name that matches nothing trains
# nothing, the run still writes an adapter, and the result is reported as a
# fine-tune that did not happen. The two entries are the two encoders rung 2
# shortlisted; a third architecture adds a row after its module names have been
# read off the model.
LORA_TARGETS = {
    # XLM-RoBERTa: attention projections by name, and every feed-forward
    # `dense` (which also catches the attention output projection).
    "BAAI/bge-m3": ("query", "key", "value", "dense"),
    # GTE's own architecture: fused QKV, the attention output, and the gated
    # feed-forward's two halves.
    "Alibaba-NLP/gte-multilingual-base": (
        "qkv_proj",
        "o_proj",
        "up_gate_proj",
        "down_proj",
    ),
}


class AdapterMismatch(RuntimeError):
    """An adapter that is not deltas onto the weights about to be loaded."""


@dataclass(frozen=True)
class TrainingExample:
    """One (document, gold heading) pair, with the confusions mined for it."""

    record_id: str
    text: str
    positive: Code
    negatives: tuple[Code, ...] = ()


@dataclass(frozen=True)
class TrainingSettings:
    """Every knob of a fine-tune, so a run is described by one object.

    Recorded verbatim in the adapter's manifest: an adapter whose settings are
    not readable months later is a checkpoint of unknown provenance, and the
    whole reason to save it as an artifact is that indexing and evaluation can
    be re-run without retraining.
    """

    epochs: int = 2
    batch_size: int = 32
    # Mined negatives per pair, on top of the batch's own.
    negatives: int = 4
    learning_rate: float = 1e-4
    # The scale similarities are divided by before the softmax. 0.05 is what
    # the contrastive literature these encoders come from trains at.
    temperature: float = 0.05
    document_length: int = 512
    # Headings are short; giving them the document's 512 would spend most of
    # each batch on padding.
    label_length: int = 128
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    warmup: float = 0.1
    seed: int = 42
    # Trades compute for memory by recomputing activations in the backward
    # pass. Off by default; on when the host's GPU is shared.
    gradient_checkpointing: bool = False
    # `bf16` runs the forward pass in bfloat16 while the weights and the
    # optimiser stay in fp32, which is the A100's native format and roughly
    # halves activation memory. No loss scaler: bfloat16 has fp32's exponent
    # range, so there is nothing to scale. `fp32` is the fallback, and the only
    # thing that runs anywhere but CUDA.
    precision: str = "bf16"

    def as_dict(self) -> dict:
        import dataclasses

        return dataclasses.asdict(self)


# --- Mining ------------------------------------------------------------------


def hard_negatives(
    codes: Sequence[Code], gold: Collection[Code], count: int
) -> tuple[Code, ...]:
    """The retriever's best wrong answers for one record, in its own order.

    Its own order, because a hard negative is hard by rank: the codes it put
    first are the ones the untrained model cannot tell from the right answer.
    Gold codes are removed rather than kept and labelled, because everything
    downstream treats this list as things to push away from.
    """
    wrong = [code for code in codes if code not in gold]
    return tuple(wrong[:count])


def examples(
    records: Sequence[Record],
    mined: Mapping[str, Sequence[Code]],
    negatives: int,
    vocabulary: Collection[Code] | None = None,
) -> list[TrainingExample]:
    """One pair per gold assignment, each carrying that record's mined negatives.

    A record with three headings becomes three pairs over the same text: the
    model is being taught a document-to-label geometry, and a record is a point
    that has to be near all three.

    `vocabulary` restricts the positives to codes the label tower can render. A
    record from the all-subjects split carries headings outside tib-core, and a
    pair whose positive has no text is a pair with no positive.
    """
    built: list[TrainingExample] = []
    for record in records:
        gold = [
            code
            for code in dict.fromkeys(record.subjects)
            if vocabulary is None or code in vocabulary
        ]
        if not gold:
            continue
        # Mined against the record, so the whole of its gold comes out — not
        # just the positive of the pair being built.
        pushed = hard_negatives(mined.get(record.id, ()), set(gold), negatives)
        built.extend(
            TrainingExample(
                record_id=record.id,
                text=record.text,
                positive=code,
                negatives=pushed,
            )
            for code in gold
        )
    return built


# --- Batching ----------------------------------------------------------------


def batches(
    pairs: Sequence[TrainingExample], size: int, seed: int
) -> Iterator[list[TrainingExample]]:
    """Shuffled batches in which no code is the positive of two pairs.

    In-batch negatives say that every other pair's positive is wrong for this
    document. Two pairs with the same positive make that statement false about
    each other, and a record with three headings makes it likely: the same
    heading arrives from many records. A repeat is therefore deferred to a later
    batch rather than dropped, so an epoch still shows the model every pair.
    """
    order = list(pairs)
    random.Random(seed).shuffle(order)

    pending: list[TrainingExample] = []
    batch: list[TrainingExample] = []
    taken: set[Code] = set()

    for pair in order:
        if pair.positive in taken:
            pending.append(pair)
            continue
        batch.append(pair)
        taken.add(pair.positive)
        if len(batch) == size:
            yield batch
            batch, taken = [], set()

    # Whatever collided, in as many passes as the collisions need. A pass that
    # places nothing closes the batch it could not add to, which frees every
    # code it was holding — so the next pass places at least one pair, and the
    # loop cannot spin.
    while pending:
        deferred: list[TrainingExample] = []
        for pair in pending:
            if pair.positive in taken:
                deferred.append(pair)
                continue
            batch.append(pair)
            taken.add(pair.positive)
            if len(batch) == size:
                yield batch
                batch, taken = [], set()
        if len(deferred) == len(pending):
            if batch:
                yield batch
            batch, taken = [], set()
        pending = deferred

    if batch:
        yield batch


def columns(batch: Sequence[TrainingExample]) -> list[Code]:
    """The codes one batch scores against: its positives, then its negatives.

    Positives first and in batch order, so row `i`'s target is column `i` and
    the loss needs no index arithmetic. Negatives follow, deduplicated against
    the positives and each other — a code that is one row's positive and
    another's mined negative is one column, and masking is what keeps that
    honest.
    """
    ordered = [pair.positive for pair in batch]
    seen = set(ordered)
    for pair in batch:
        for code in pair.negatives:
            if code not in seen:
                seen.add(code)
                ordered.append(code)
    return ordered


def false_negative_mask(
    batch: Sequence[TrainingExample],
    scored: Sequence[Code],
    gold: Mapping[str, Collection[Code]],
) -> list[list[bool]]:
    """Which (row, column) scores the loss must not count as wrong.

    A column that is a gold heading for the row's record is a right answer
    sitting in the negatives, and counting it teaches the model that a correct
    assignment is a mistake. It happens two ways — another pair of the same
    record is in the batch, or another record's positive is also this record's
    gold — and both are the same fix.

    The row's own target is never masked, whatever else is true of it: it is the
    thing being learned.
    """
    masked = []
    for position, pair in enumerate(batch):
        assignments = gold.get(pair.record_id, ())
        masked.append(
            [
                code in assignments and index != position
                for index, code in enumerate(scored)
            ]
        )
    return masked


# --- The adapter artifact ----------------------------------------------------


def lora_targets(name: str) -> tuple[str, ...]:
    """The projection names LoRA adapts in this architecture."""
    try:
        return LORA_TARGETS[name]
    except KeyError:
        raise KeyError(
            f"no LoRA target modules are recorded for {name!r} (known: "
            f"{', '.join(sorted(LORA_TARGETS))}). Read them off the model and "
            "add a row; a name that matches no module trains nothing and still "
            "writes an adapter."
        ) from None


def adapter_manifest(
    *,
    base: str,
    revision: str,
    data_revision: str,
    negatives: str,
    hyperparameters: Mapping[str, object],
    losses: Sequence[float],
    seconds: float,
    host: str,
    examples: int = 0,
    trainable: int = 0,
    skipped: int = 0,
) -> dict:
    """What the adapter is deltas onto, and what produced it."""
    return {
        "schema": MANIFEST_SCHEMA,
        "base": base,
        "revision": revision,
        "data_revision": data_revision,
        # The artifact key of the mined negatives, so the training inputs are
        # identifiable and not merely described.
        "negatives": negatives,
        "hyperparameters": dict(hyperparameters),
        "examples": examples,
        # Pairs the run was given, and batches it could not fit. The second is
        # not a detail on a shared GPU: it is how much of the first was used.
        "trainable_parameters": trainable,
        "skipped_batches": skipped,
        "losses": [float(loss) for loss in losses],
        "seconds": float(seconds),
        "host": host,
    }


def read_adapter(directory: str | Path, base: str, revision: str) -> dict:
    """The adapter's manifest, if it belongs on the weights being loaded.

    LoRA deltas are trained against particular weights. Applied to another
    model, or to another revision of the same one, they are noise the run has no
    way to notice: the vectors come out normalised, the retrievers rank them,
    and the score is simply worse than it should be. So it is checked here, at
    the one place an adapter is opened.
    """
    path = Path(directory) / ADAPTER_MANIFEST
    if not path.exists():
        raise AdapterMismatch(
            f"{path} is missing, so nothing says which weights the adapter in "
            f"{directory} belongs on.\n{TRAIN_HINT}"
        )
    manifest = json.loads(path.read_text())

    if manifest.get("base") != base:
        raise AdapterMismatch(
            f"the adapter in {directory} was trained over "
            f"{manifest.get('base')!r} and the config names {base!r}; LoRA "
            "deltas are specific to the weights they were fitted against."
        )
    if manifest.get("revision") != revision:
        raise AdapterMismatch(
            f"the adapter in {directory} was trained over revision "
            f"{manifest.get('revision')!r} and this run pins {revision!r}."
        )
    return manifest


# --- Training ----------------------------------------------------------------


@dataclass
class TrainingRun:
    """What a fine-tune produced, beyond the weights themselves."""

    losses: list[float] = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    trainable: int = 0
    # Batches the GPU could not fit. Counted rather than swallowed: it is a
    # number that belongs in the adapter's manifest, because a run that skipped
    # a tenth of its data trained on a tenth less data.
    skipped: int = 0


def apply_lora(model, name: str, settings: TrainingSettings):
    """Wrap a `sentence-transformers` model's transformer in LoRA adapters.

    The adapters go on the transformer itself rather than on the whole
    `SentenceTransformer`, because the modules after it — pooling, and a
    normalisation layer — hold no weights to adapt, and peft would have to be
    told to ignore them.

    Under `bf16` the frozen base is held in bfloat16 and only the adapters stay
    in fp32. The base is never updated, so its precision costs nothing but the
    forward pass's — and it is 2.3 GB of the memory a batch has to fit beside,
    which on a shared GPU is the difference between a batch of 8 and one of 32.
    In-batch negatives are most of the training signal, so batch size is the
    setting worth buying.
    """
    import torch
    from peft import LoraConfig, get_peft_model

    transformer = model[0]
    if settings.precision == "bf16":
        transformer.auto_model.to(torch.bfloat16)
    transformer.auto_model = get_peft_model(
        transformer.auto_model,
        LoraConfig(
            r=settings.rank,
            lora_alpha=settings.alpha,
            lora_dropout=settings.dropout,
            bias="none",
            target_modules=list(lora_targets(name)),
        ),
    )
    if settings.precision == "bf16":
        # The adapters back to fp32: they are what AdamW keeps moments for, and
        # bfloat16's seven mantissa bits lose an update of the size this
        # learning rate makes.
        for parameter in transformer.auto_model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    if settings.gradient_checkpointing:
        transformer.auto_model.enable_input_require_grads()
        transformer.auto_model.gradient_checkpointing_enable()
    return model


def train(
    model,
    pairs: Sequence[TrainingExample],
    label_texts: Mapping[Code, str],
    gold: Mapping[str, Collection[Code]],
    settings: TrainingSettings,
    prefixes,
    device: str = "cuda",
    log=print,
) -> TrainingRun:
    """Fit the adapters, and report the loss the way a training log should.

    One optimiser step per batch, over an InfoNCE objective: the document is
    scored against every column of the batch, the scores that would punish a
    correct assignment are removed, and the row's own heading has to win.

    The two towers are encoded in one pass each rather than as one concatenated
    batch, because they are asymmetric — E5's convention puts a different prefix
    on each side, and the document tower reads 512 tokens where the label tower
    reads 128.
    """
    import time

    import torch
    from torch.nn import functional as functional_ops

    parameters = [p for p in model.parameters() if p.requires_grad]
    run = TrainingRun(trainable=sum(p.numel() for p in parameters))
    optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate)

    planned = sum(
        1 for _ in batches(pairs, settings.batch_size, settings.seed)
    ) * settings.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _schedule(step, planned, settings.warmup)
    )

    autocast = _autocast(settings.precision, device, torch)

    def step(batch: Sequence[TrainingExample]) -> float:
        """One optimiser step over one batch, returning its loss.

        The scores are computed in fp32 whatever the towers ran in: the matrix
        is `len(batch)` by however many columns, so widening it costs nothing,
        and the softmax over it is where a contrastive loss loses resolution.
        """
        scored = columns(batch)
        with autocast:
            documents = _encode(
                model,
                [prefixes.document + pair.text for pair in batch],
                settings.document_length,
                device,
            )
            labels = _encode(
                model,
                [prefixes.label + label_texts[code] for code in scored],
                settings.label_length,
                device,
            )

        logits = documents.float() @ labels.float().T / settings.temperature
        mask = torch.tensor(
            false_negative_mask(batch, scored, gold), device=logits.device
        )
        logits = logits.masked_fill(mask, float("-inf"))
        loss = functional_ops.cross_entropy(
            logits, torch.arange(len(batch), device=logits.device)
        )

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())

    model.train()
    started = time.perf_counter()
    for epoch in range(settings.epochs):
        total, counted = 0.0, 0
        # A different shuffle each epoch, or every epoch would present the same
        # in-batch negatives and only the mined ones would vary.
        for batch in batches(pairs, settings.batch_size, settings.seed + epoch):
            try:
                total += step(batch)
            except torch.OutOfMemoryError:
                # The GPU host is shared, and a co-tenant's allocation can grow
                # under a run that has already been going for an hour. One
                # oversized batch is worth less than the run, so it is dropped,
                # counted, and reported in the adapter's manifest — the
                # alternative is a two-hour traceback and no adapter.
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                run.skipped += 1
                continue

            counted += 1
            run.steps += 1
            if counted % 100 == 0:
                log(
                    f"  epoch {epoch + 1} step {counted}: "
                    f"loss {total / counted:.4f}"
                    + (f" ({run.skipped} skipped)" if run.skipped else "")
                )

        mean = total / max(counted, 1)
        run.losses.append(mean)
        log(f"epoch {epoch + 1}/{settings.epochs}: mean loss {mean:.4f}")

    run.seconds = time.perf_counter() - started
    model.eval()
    return run


def _autocast(precision: str, device: str, torch):
    """Mixed precision where the hardware has it, and nothing where it does not."""
    import contextlib

    if precision == "bf16" and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"unknown precision {precision!r}; expected bf16 or fp32")
    return contextlib.nullcontext()


def _encode(model, texts: Sequence[str], length: int, device: str):
    """One tower's vectors, L2-normalised, with gradients attached.

    `model.encode` is the inference path: it detaches, batches and moves things
    to numpy. Training needs the graph, so the two steps it wraps — tokenise,
    forward — are called directly.
    """
    import torch
    from torch.nn import functional as functional_ops

    model.max_seq_length = length
    features = model.tokenize(list(texts))
    features = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in features.items()
    }
    embedded = model(features)["sentence_embedding"]
    return functional_ops.normalize(embedded, p=2, dim=1)


def _schedule(step: int, planned: int, warmup: float) -> float:
    """Linear warmup then linear decay, as a multiplier on the learning rate."""
    if planned <= 0:
        return 1.0
    warm = max(int(planned * warmup), 1)
    if step < warm:
        return step / warm
    return max(0.0, (planned - step) / max(planned - warm, 1))


def save_adapter(model, directory: str | Path, manifest: Mapping[str, object]) -> Path:
    """Write the deltas and the manifest that says what they belong on."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    model[0].auto_model.save_pretrained(str(path))
    (path / ADAPTER_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return path
