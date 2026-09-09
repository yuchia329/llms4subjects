"""Uniform interface over the four candidate encoders.

Hides their differing pooling, prefix and instruction conventions, so an
encoder swap is a config change. Emits dense vectors and, where the model
supports it, sparse term weights.

Vectors are L2-normalised on the way out, so every retriever downstream can
treat a dot product as cosine similarity and no stage has to know which model
produced the numbers.

Tickets 04 and 05 need one encoder and ticket 09 screens all four, which is why
`PREFIXES` is a table rather than a branch. A model absent from that table gets
no prefixes at all rather than E5's, which is the right default — a convention
guessed for the wrong family costs more than none — but it does mean adding an
encoder means adding its row, or its label tower is encoded under the document
convention. Ticket 06 declares the sparse interface a model may also offer; the
adapter that implements it arrives with the encoder that has it.

Nothing is loaded that `reference/model_releases.json` does not vouch for.
`resolve` checks the model against the 2025-01-31 cutoff and settles the
revision before any weights are fetched, so an encoder swap cannot quietly
carry the comparison outside the window the shared task closed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..config import EncoderConfig
from ..models import ModelRelease, UnverifiedRevision, check_cutoff


class Encoder(Protocol):
    """What every encoder adapter offers the rest of the pipeline."""

    @property
    def dimensions(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_labels(self, texts: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class SparseEncoder(Protocol):
    """An encoder whose forward pass also emits sparse term weights.

    BGE-M3 is the one of the four candidates that does. Where a model offers
    them, they are the lexical retriever's scores, since they arrive from the
    pass the dense tower has already paid for; where it does not, BM25 over the
    label strings is built instead.
    """

    def encode_sparse_documents(
        self, texts: Sequence[str]
    ) -> Sequence[Mapping[str, float]]: ...

    def encode_sparse_labels(
        self, texts: Sequence[str]
    ) -> Sequence[Mapping[str, float]]: ...


@runtime_checkable
class SizedEncoder(Protocol):
    """An encoder that can say how many parameters it has.

    Nothing in the pipeline needs this — it is the other half of a cost claim,
    read by the rung-1 screen, which reports each candidate's wall clock beside
    its size. Asked of the adapter rather than recorded in the registry, so the
    number describes the weights that actually ran.
    """

    @property
    def parameters(self) -> int: ...


def parameter_count(encoder: Encoder) -> int | None:
    """How large the model is, where the adapter can say, and nothing if not."""
    return encoder.parameters if isinstance(encoder, SizedEncoder) else None


def sparse_weights(encoder: Encoder) -> SparseEncoder | None:
    """The encoder itself if it emits sparse term weights, otherwise nothing.

    Asked rather than configured, so that switching to a model that has them
    switches the lexical retriever over with it and a model that has not is
    never asked for something it cannot do.
    """
    return encoder if isinstance(encoder, SparseEncoder) else None


@dataclass(frozen=True)
class Prefixes:
    """The instruction strings a family expects in front of its two inputs.

    E5 was trained with `query:` and `passage:` and degrades measurably without
    them. Document-to-document retrieval is symmetric, so both sides of it get
    the document prefix; only the label tower is the asymmetric case.
    """

    document: str = ""
    label: str = ""


# Matched as a substring of the lowercased model name, longest match first, so a
# fine-tuned checkpoint named after its base model inherits its conventions.
PREFIXES = {
    "e5": Prefixes(document="query: ", label="passage: "),
}

NO_PREFIXES = Prefixes()

# Above this many texts, encoding is slow enough that silence looks like a hang.
PROGRESS_THRESHOLD = 2048


def prefixes_for(name: str) -> Prefixes:
    """The prefix convention for a model name, or none if it has no convention."""
    lowered = name.lower()
    matches = sorted(
        (key for key in PREFIXES if key in lowered), key=len, reverse=True
    )
    return PREFIXES[matches[0]] if matches else NO_PREFIXES


class SentenceTransformerEncoder:
    """A `sentence-transformers` model behind the `Encoder` protocol."""

    def __init__(self, model, prefixes: Prefixes, batch_size: int):
        self._model = model
        self._prefixes = prefixes
        self._batch_size = batch_size

    @property
    def dimensions(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def parameters(self) -> int:
        return sum(tensor.numel() for tensor in self._model.parameters())

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self._prefixes.document)

    def encode_labels(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self._prefixes.label)

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        vectors = self._model.encode(
            [prefix + text for text in texts],
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > PROGRESS_THRESHOLD,
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass(frozen=True)
class Resolved:
    """What a config resolves to once the model registry has had its say.

    Separate from `load` so that what gets loaded is assertable without weights
    being fetched: the revision a run pins, and whether it is about to execute
    modelling code shipped from the hub, are the two facts the cutoff claim
    rests on, and both are decided here.
    """

    release: ModelRelease
    revision: str
    prefixes: Prefixes
    trust_remote_code: bool = False
    code_revision: str | None = None

    def kwargs(self) -> dict[str, object]:
        """The `SentenceTransformer` arguments this resolution implies."""
        arguments: dict[str, object] = {"revision": self.revision}
        if self.trust_remote_code:
            arguments["trust_remote_code"] = True
            # Remote code is fetched twice, by two different calls: the
            # configuration class by `AutoConfig` and the model class by
            # `AutoModel`. Both need the pin, or half the executed code floats
            # at whatever its repository serves today — which is the half that
            # decides what the other half is.
            pinned_code = {"code_revision": self.code_revision}
            arguments["model_kwargs"] = dict(pinned_code)
            arguments["config_kwargs"] = dict(pinned_code)
        return arguments


def resolve(config: EncoderConfig) -> Resolved:
    """Check the model against the cutoff and settle what will be loaded.

    The revision comes from the registry, not from the config. `encoder.revision`
    may restate the registry's pin — a config is allowed to be explicit — but it
    may not name a different one: the cutoff is a claim about the commit that
    gets loaded, and a bare SHA carries no date, so a config that could override
    the pin could carry the whole comparison past the window while every
    recorded date stayed true. Registering the commit is what makes it
    checkable, and that is one line in `scripts/verify_model_releases.py`.
    """
    entry = check_cutoff(config.name)
    if config.revision is not None and config.revision != entry.revision:
        raise UnverifiedRevision(
            f"{config.name} is pinned to {entry.revision} by "
            f"reference/model_releases.json, and encoder.revision asks for "
            f"{config.revision}, which nothing has dated against the "
            "2025-01-31 cutoff. Register that commit, or drop the override."
        )
    return Resolved(
        release=entry,
        revision=entry.revision,
        prefixes=prefixes_for(config.name),
        trust_remote_code=entry.trust_remote_code,
        code_revision=entry.code_revision,
    )


def pinned(config):
    """The experiment config with its encoder revision resolved to the pin.

    Every artifact key that depends on the encoder is a fingerprint of the
    `encoder` section, and the section a config file carries has no revision in
    it — the pin lives in the registry. Keyed on that section as written, one
    encoder's vectors would be served to any revision of it, so re-pinning a
    model to different weights would read back the old ones and report them as
    the new. Harnesses therefore key on this rather than on the config as
    loaded, and the pin appears in every manifest.

    Takes and returns an `ExperimentConfig`; typed loosely to keep this module
    off the config module's inner shape.
    """
    import dataclasses

    resolved = resolve(config.encoder)
    if config.encoder.revision == resolved.revision:
        return config
    return dataclasses.replace(
        config,
        encoder=dataclasses.replace(config.encoder, revision=resolved.revision),
    )


def load(config: EncoderConfig, device: str) -> Encoder:
    """Build the adapter named by the config, on the given device."""
    resolved = resolve(config)

    # Checked before any weights are fetched. An adapter trained over another
    # model would otherwise be discovered after a 2 GB download, and — worse —
    # an adapter that merely *looks* applicable is checked here too, which is
    # the case nothing downstream could notice.
    if config.adapter is not None:
        from ..finetune import read_adapter

        read_adapter(config.adapter, config.name, resolved.revision)

    # Imported here rather than at module scope: `transformers` and `torch` cost
    # seconds to import, and every test of this package that does not encode
    # anything would pay it.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.name, device=device, **resolved.kwargs())
    model.max_seq_length = config.max_length
    if config.adapter is not None:
        model = _apply_adapter(model, config.adapter)
    return SentenceTransformerEncoder(
        model, resolved.prefixes, config.batch_size
    )


def _apply_adapter(model, directory: str):
    """Fold a fine-tune's LoRA deltas into the loaded weights.

    Merged rather than kept as a live peft wrapper: every use of the adapter in
    this project is inference over hundreds of thousands of texts, and a merged
    model is the base model's own forward pass — same shape, same speed, no
    adapter layers in the loop. It also means nothing downstream can tell a
    trained encoder from an untrained one by its type, which is what makes the
    trained and untrained rows of rung 3 the same code path.
    """
    from peft import PeftModel

    transformer = model[0]
    adapted = PeftModel.from_pretrained(transformer.auto_model, str(directory))
    transformer.auto_model = adapted.merge_and_unload()
    return model
