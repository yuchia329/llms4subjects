"""Cross-encoder scoring of a document against each candidate's label text.

Reduces the top 100 to the final 50, and reports a per-record confidence so the
adjudicator can be routed only the least-confident records.

## Why a cross-encoder at all

Everything upstream scores a document and a label in separate forward passes and
compares the two vectors, which is what makes 79,427 labels affordable. The
price is that the document is summarised before it ever meets the label, so two
labels whose names differ in one qualifying word — the pairs the vocabulary is
full of — land in nearly the same place. A cross-encoder reads the pair jointly
and can tell them apart, and it can only be afforded because fusion has already
cut the vocabulary to 100 candidates.

Reranking is a reordering and a truncation, never a source: a code that no
retriever proposed cannot appear here, and every surviving candidate keeps the
per-retriever provenance fusion gave it, because the project's claim is about
attribution and a reranked list that forgot where its entries came from could
not support one.

The score written back is the model's own relevance in [0, 1] rather than a
rank, and it replaces the fused reciprocal-rank score. That is deliberate: the
confidence measure below reads these scores, so they have to be comparable
between records, which a reciprocal of a rank is not.

Implemented by ticket 12.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from ..config import RerankerConfig
from ..contracts import Candidate, CandidateList, Code, Record
from ..models import ModelRelease, UnverifiedRevision, check_cutoff

# How many of a record's top candidates its confidence is read from. Five,
# because P@5 is the metric a suggest-and-confirm workflow lives on, so the
# question the number answers — "how good is what this record would be shown?"
# — is asked over the same window that answers it in the results table.
CONFIDENCE_K = 5

# Positions are 0-based and reciprocal rank fusion is defined over 1-based
# ranks, as in `stages.fusion`.
RANK_OFFSET = 1

# Above this many pairs, a silent reranking pass looks like a hang: dev is
# 5,354 records × 100 candidates through a model that reads both sides jointly.
PROGRESS_THRESHOLD = 2048


class CrossEncoder(Protocol):
    """A model that scores document-label pairs jointly.

    One method, taking every pair for every record at once, because batching is
    the model's business and a per-record call would spend the batch dimension
    on 100 short pairs.

    Scores are relevance in [0, 1], not logits: `confidence` averages them and
    the calibration analysis reads them as probabilities, so an adapter that
    returned raw logits would make a confidence figure that means something
    different for every model.
    """

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...


class SentenceTransformerCrossEncoder:
    """A `sentence-transformers` `CrossEncoder` behind the protocol above.

    Pairs are scored longest-first in length-sorted order and the scores are put
    back where they belong, which is not a micro-optimisation here: a batch is
    padded to its longest member, TIBKAT abstracts run from one line to two
    thousand characters, and 100 candidates for one record all carry the same
    document — so an unsorted batch pads a short record's pairs to a long one's
    length. Measured over 400 dev pairs on this laptop's MPS backend, sorting is
    worth 3 to 4 times the throughput: 4 to 17 pairs a second for the larger of
    the two screened rerankers, 15 to 48 for the smaller.
    """

    def __init__(self, model, activation, batch_size: int):
        self._model = model
        self._activation = activation
        self._batch_size = batch_size

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)

        order = sorted(
            range(len(pairs)), key=lambda index: -_pair_length(pairs[index])
        )
        scored = self._model.predict(
            [pairs[index] for index in order],
            batch_size=self._batch_size,
            activation_fct=self._activation,
            convert_to_numpy=True,
            show_progress_bar=len(pairs) > PROGRESS_THRESHOLD,
        )
        scores = np.empty(len(pairs), dtype=np.float32)
        scores[order] = np.asarray(scored, dtype=np.float32).reshape(-1)
        return scores


def _pair_length(pair: tuple[str, str]) -> int:
    """Characters in a pair, which is what the batch's padding is driven by.

    Characters rather than tokens on purpose: tokenising twice to save padding
    would cost more than the padding does, and the two agree closely enough to
    put similar pairs in the same batch.
    """
    return len(pair[0]) + len(pair[1])


@dataclass(frozen=True)
class Resolved:
    """What a reranker config resolves to once the registry has had its say.

    Separate from `load` for the same reason the encoder's is: the revision a run
    pins is the whole of the cutoff claim, and it should be assertable without
    568M parameters being fetched to assert it.
    """

    release: ModelRelease
    revision: str


def resolve(config: RerankerConfig) -> Resolved:
    """Check the reranker against the 2025-01-31 cutoff and pin its revision.

    The revision comes from the registry rather than from the config, on the
    same terms as the encoder's (`stages.encoders.resolve`): both screened
    rerankers predate the cutoff by more than a year, but a name alone loads
    whatever the hub serves today, and a bare SHA in a config carries no date
    that anything has checked. `reranker.revision` may restate the pin and may
    not disagree with it.
    """
    if not config.model:
        raise ValueError("reranker.model names no model to load")
    entry = check_cutoff(config.model)
    if config.revision is not None and config.revision != entry.revision:
        raise UnverifiedRevision(
            f"{config.model} is pinned to {entry.revision} by "
            f"reference/model_releases.json, and reranker.revision asks for "
            f"{config.revision}, which nothing has dated against the "
            "2025-01-31 cutoff. Register that commit, or drop the override."
        )
    return Resolved(release=entry, revision=entry.revision)


def pinned(config):
    """The experiment config with its reranker revision resolved to the pin.

    The counterpart of `stages.encoders.pinned`, and there for the same reason:
    an artifact keyed on the `reranker` section as a config file carries it —
    with no revision in it — would serve one model's scores to any revision of
    it, and re-pinning the reranker would read back the old ones and report
    them as the new. Harnesses key on this.

    Takes and returns an `ExperimentConfig`; typed loosely to keep this module
    off the config module's inner shape.
    """
    import dataclasses

    resolved = resolve(config.reranker)
    if config.reranker.revision == resolved.revision:
        return config
    return dataclasses.replace(
        config,
        reranker=dataclasses.replace(config.reranker, revision=resolved.revision),
    )


def load(config: RerankerConfig, device: str) -> CrossEncoder:
    """Build the reranker named by the config, on the given device.

    Called by the pipeline rather than by `rerank`, for the same reason the
    encoder is: a test must be able to assert this stage's contract without
    downloading half a gigabyte of weights.
    """
    resolved = resolve(config)

    # Imported here rather than at module scope: `torch` and `transformers`
    # cost seconds to import, and every test that reranks a fixture with a fake
    # model would pay it.
    import torch
    from sentence_transformers import CrossEncoder as SentenceTransformerModel

    model = SentenceTransformerModel(
        config.model,
        revision=resolved.revision,
        max_length=config.max_length,
        device=device,
    )
    labels = int(getattr(model.config, "num_labels", 1))
    if labels != 1:
        # A per-logit sigmoid over a multi-class head is not a relevance, and
        # the confidence measure would read it as one.
        raise ValueError(
            f"{config.model} has {labels} output labels; the reranker interface "
            "is a single relevance score per pair"
        )
    return SentenceTransformerCrossEncoder(
        model, torch.nn.Sigmoid(), config.batch_size
    )


def rerank(
    records: Sequence[Record],
    candidates: Sequence[CandidateList],
    label_texts: Mapping[Code, str],
    config: RerankerConfig,
    model: CrossEncoder,
) -> list[CandidateList]:
    """Reorder each record's top `input_k` candidates, keeping `output_k`.

    `records` and `candidates` are matched by position, and a mismatch is an
    error rather than a positional guess: scoring one record's text against
    another's candidates would be invisible in the output and fatal to every
    number taken from it.

    Every pair for every record goes to the model in one call, so the batch
    dimension is spent on the whole pass rather than on 100 short pairs at a
    time. Which side of each pair is the query is `reranker.query`: these models
    are asymmetric, and this task fits neither of their two roles obviously.
    """
    if not records:
        return []
    _check_alignment(records, candidates)

    document_first = config.query == "document"
    pairs: list[tuple[str, str]] = []
    heads: list[tuple[Candidate, ...]] = []
    for record, result in zip(records, candidates):
        head = result.candidates[: config.input_k]
        heads.append(head)
        # Read once per record rather than once per candidate: `Record.text`
        # builds its string on every access, and a dev pass would otherwise
        # hold 100 copies of every abstract at once — 907 MB against 9 MB.
        text = record.text
        for candidate in head:
            label = _label_text(label_texts, candidate.code)
            pairs.append((text, label) if document_first else (label, text))

    scores = np.asarray(model.score(pairs), dtype=np.float32).reshape(-1)
    if len(scores) != len(pairs):
        raise ValueError(
            f"the reranker returned {len(scores)} scores for {len(pairs)} "
            "pairs; the pairs and the scores are matched by position"
        )

    reranked: list[CandidateList] = []
    start = 0
    for result, head in zip(candidates, heads):
        reranked.append(
            _ranked(
                result.record_id,
                head,
                scores[start : start + len(head)],
                config,
            )
        )
        start += len(head)
    return reranked


def confidence(candidates: CandidateList, k: int = CONFIDENCE_K) -> float:
    """Per-record confidence, used only for routing.

    The mean score of the top `k` candidates. Not the top one: a record whose
    first suggestion is certain and whose next four are noise is not a record to
    leave alone, and 2.40 gold labels per record is what a k of five is being
    asked about. Not the whole 50 either — every record's fiftieth candidate is
    weak, so including the tail would measure the candidate depth every record
    shares rather than what distinguishes them.

    Defined over any scored ranking rather than over reranked lists alone, and
    that generality is load-bearing rather than incidental: on reranked lists it
    is a mean relevance in [0, 1], on fused ones a mean reciprocal-rank score,
    and ticket 12 measured the second to be the calibrated one (docs/results.md)
    — so the routing signal ticket 13 reads comes through this function whether
    or not a reranker runs. Both are comparable between records, which is the
    only property routing needs, and neither is comparable to the other.

    A record the retrievers found nothing for scores 0.0 rather than raising:
    routing has to be defined for it, and it is exactly the case a router
    should send onward.
    """
    head = candidates.candidates[:k]
    if not head:
        return 0.0
    return float(sum(candidate.score for candidate in head) / len(head))


def _ranked(
    record_id: str,
    head: Sequence[Candidate],
    scores: np.ndarray,
    config: RerankerConfig,
) -> CandidateList:
    """One record's candidates reordered and cut to `output_k`.

    Under `mix: replace` the order is the model's, which is the pipeline
    contract in docs/spec.md. Under `mix: fuse` the model's order and the order
    it was handed are combined by reciprocal rank, because ticket 12 measured
    them to be right about different records: the cross-encoder is the only
    component that has moved the zero-shot band and the worst thing that has
    happened to the head band.

    Ties are broken by code either way, so no ranking depends on the order a
    dict happened to be in.
    """
    by_model = sorted(
        zip(range(len(head)), (float(score) for score in scores)),
        key=lambda pair: (-pair[1], head[pair[0]].code),
    )

    if config.mix == "replace":
        ordered = [(position, score) for position, score in by_model]
    else:
        weight, rrf_k = config.mix_weight, config.mix_rrf_k
        mixed = {
            position: weight / (rrf_k + model_rank + RANK_OFFSET)
            + 1.0 / (rrf_k + position + RANK_OFFSET)
            for model_rank, (position, _) in enumerate(by_model)
        }
        ordered = sorted(
            mixed.items(), key=lambda pair: (-pair[1], head[pair[0]].code)
        )

    return CandidateList(
        record_id=record_id,
        candidates=tuple(
            Candidate(
                code=head[position].code,
                score=score,
                # The retrievers that proposed it, unchanged. The reranker is
                # not a candidate source and does not claim to be one.
                sources=dict(head[position].sources),
            )
            for position, score in ordered[: config.output_k]
        ),
    )


def _label_text(label_texts: Mapping[Code, str], code: Code) -> str:
    try:
        return label_texts[code]
    except KeyError:
        raise KeyError(
            f"no label text for candidate {code!r}; the reranker reads the same "
            "rendering the label tower does, so every candidate needs one"
        ) from None


def _check_alignment(
    records: Sequence[Record], candidates: Sequence[CandidateList]
) -> None:
    if len(records) != len(candidates):
        raise ValueError(
            f"{len(records)} records against {len(candidates)} candidate lists; "
            "reranking is positional"
        )
    for record, result in zip(records, candidates):
        if record.id != result.record_id:
            raise ValueError(
                f"record {record.id!r} was handed the candidates of "
                f"{result.record_id!r}; reranking is positional and both "
                "sequences come from the same pass"
            )
