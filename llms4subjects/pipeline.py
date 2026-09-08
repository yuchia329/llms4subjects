"""The `predict` seam: the whole pipeline behind one call.

Everything below it — indexing, the three retrievers, fusion, the group prior,
reranking, adjudication — is an implementation detail reachable only through
configuration. Tests assert contract invariants here rather than reaching into
stages whose behaviour is defined by external model weights.

Wiring lands with the stages it wires; see docs/spec.md, "Testing Decisions".
Tickets 04 and 05 wire the neighbour and dense retrievers; the calls the lexical
retriever and the late stages will hang from are here as refusals, so an
experiment that enables one of them fails rather than reporting an ablation that
never ran.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .artifacts import ArtifactStore, CachedEncoder
from .config import RETRIEVER_NAMES, ExperimentConfig
from .contracts import (
    CODES_PER_RECORD,
    CandidateList,
    Code,
    Record,
    VocabularyEntry,
)
from .stages import encoders, fusion, indexes, label_text, retrievers
from .stages.encoders import Encoder


def predict(
    records: Sequence[Record],
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    index_records: Sequence[Record],
    store: ArtifactStore,
    device: str = "auto",
    encoder: Encoder | None = None,
    name_qualifiers: Mapping[str, str] | None = None,
) -> list[CandidateList]:
    """Rank vocabulary codes for each record.

    `records` are the records being predicted and `index_records` the corpus
    being retrieved from; they are separate arguments so that no record can
    contribute its own gold subjects to its own candidate set.

    `encoder` is for tests, which must not download model weights to assert an
    invariant that holds for any encoder. Left unset, the configured one is
    loaded onto `device`.
    """
    enabled = [name for name in RETRIEVER_NAMES if config.retrievers[name].enabled]
    _refuse_unbuilt_stages(config, enabled)

    if not enabled:
        # An ablation that turns everything off is a row in the results table.
        return [CandidateList(record.id, ()) for record in records]

    if encoder is None:
        from .hardware import select_device

        encoder = encoders.load(config.encoder, select_device(device))
    encoder = CachedEncoder(encoder, store, config)

    per_retriever = {
        name: _restrict(
            _build(name, config, vocabulary, index_records, encoder, name_qualifiers)
            .retrieve(records),
            vocabulary,
        )
        for name in enabled
    }

    combined = _combine(per_retriever, config)
    return [candidates.top(CODES_PER_RECORD) for candidates in combined]


def _refuse_unbuilt_stages(
    config: ExperimentConfig, enabled: Sequence[str]
) -> None:
    """Refuse anything the configuration enables and this code cannot do.

    Checked before any encoding, so an experiment that asks for a stage nobody
    has built fails in a second rather than after indexing. The alternative is
    worse than slow: a flag that is silently ignored reports an ablation that
    never ran. Every refusal here is deleted by the ticket it names.
    """
    if len(enabled) > 1:
        # All three retrievers exist as of ticket 06, and nothing combines them
        # yet. Refused here rather than inside `_combine` so that a config
        # asking for two of them fails before the vocabulary is embedded.
        raise NotImplementedError(
            "ticket 07: reciprocal rank fusion; one retriever at a time until "
            f"then, and {', '.join(enabled)} are all enabled"
        )
    if not config.label_text.bilingual:
        # The default is bilingual and nothing reads the flag yet, so the
        # German-only side of the ablation would otherwise score identically to
        # the bilingual one and be reported as "translation does not help".
        raise NotImplementedError("ticket 08: German-only label text")
    if config.group_prior.enabled:
        raise NotImplementedError("ticket 11: group prior")
    if config.reranker.enabled:
        raise NotImplementedError("ticket 12: cross-encoder reranking")
    if config.adjudication.enabled:
        raise NotImplementedError("ticket 13: LLM adjudication")


def _build(
    name: str,
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    index_records: Sequence[Record],
    encoder: Encoder,
    name_qualifiers: Mapping[str, str] | None,
) -> retrievers.Retriever:
    settings = config.retrievers[name]

    if name == "knn":
        selected = indexes.select_documents(index_records, config.index)
        index = indexes.build_document_index(selected, encoder, vocabulary)
        return retrievers.NeighbourRetriever(index, encoder, settings)

    texts = _label_texts(config, vocabulary, name_qualifiers)

    if name == "dense":
        index = indexes.build_label_index(
            list(texts), list(texts.values()), encoder
        )
        return retrievers.DenseLabelRetriever(index, encoder, settings)

    variants = _label_variants(config, vocabulary, name_qualifiers)
    return retrievers.LexicalLabelRetriever(
        _lexical_index(texts, variants, encoder), settings
    )


def _lexical_index(
    texts: Mapping[Code, str],
    variants: Mapping[Code, tuple[str, ...]],
    encoder: Encoder,
) -> indexes.LexicalMatcher:
    """The encoder's own term weights where it emits them, BM25 where it does not.

    A model that emits sparse weights gives two retrievers for one forward pass,
    so it is asked first, and asked over `texts` — the same label rendering the
    dense tower reads. BM25 instead indexes `variants`, the labels' own surface
    strings, which is what the 53.7%-against-23.0% verbatim measurement in
    docs/spec.md was made over.
    """
    sparse = encoders.sparse_weights(encoder)
    if sparse is not None:
        return indexes.build_sparse_term_index(
            list(texts), list(texts.values()), sparse
        )
    return indexes.build_lexical_index(list(variants), list(variants.values()))


def _label_texts(
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    name_qualifiers: Mapping[str, str] | None,
) -> dict[Code, str]:
    rendered = label_text.render(
        vocabulary.values(),
        config.label_text.qualifiers,
        name_qualifiers,
        config.label_text.include_definition,
    )
    return {entry.code: entry.text for entry in rendered}


def _label_variants(
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    name_qualifiers: Mapping[str, str] | None,
) -> dict[Code, tuple[str, ...]]:
    return label_text.variants(
        vocabulary.values(), config.label_text.qualifiers, name_qualifiers
    )


def _restrict(
    candidates: Sequence[CandidateList], vocabulary: Mapping[Code, VocabularyEntry]
) -> list[CandidateList]:
    """Drop anything outside the vocabulary being predicted over.

    Predictions are restricted to the tib-core vocabulary regardless of index
    composition (docs/spec.md, "Scope of the benchmark"), and this is the one
    place that holds for every retriever, present and future. The document index
    filters too, so that an out-of-vocabulary subject does not spend a candidate
    slot before reaching here; that is an efficiency, and this is the guarantee.
    """
    return [
        CandidateList(
            record_id=result.record_id,
            candidates=tuple(
                candidate
                for candidate in result.candidates
                if candidate.code in vocabulary
            ),
        )
        for result in candidates
    ]


def _combine(
    per_retriever: Mapping[str, Sequence[CandidateList]],
    config: ExperimentConfig,
) -> list[CandidateList]:
    """One ranked list per record, from however many retrievers ran.

    A single retriever needs no fusion, and saying so here keeps ticket 04's
    number free of a rank-fusion transform that has nothing to fuse.
    """
    if len(per_retriever) == 1:
        only = next(iter(per_retriever.values()))
        return [result.top(config.fusion.candidates) for result in only]

    weights = {
        name: config.retrievers[name].weight for name in per_retriever
    }
    return fusion.fuse(per_retriever, weights, config.fusion)
