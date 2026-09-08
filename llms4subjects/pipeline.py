"""The `predict` seam: the whole pipeline behind one call.

Everything below it — indexing, the three retrievers, fusion, the group prior,
reranking, adjudication — is an implementation detail reachable only through
configuration. Tests assert contract invariants here rather than reaching into
stages whose behaviour is defined by external model weights.

Wiring lands with the stages it wires; see docs/spec.md, "Testing Decisions".
Every stage in the spec is now wired, and each is off by default, so a rung is
described by which flags its config turns on rather than by which code path it
takes.

`predict` returns the fused top 100 with per-retriever provenance rather than
the 50 the submission takes. The extra 50 are what the reranker reads and what
the recall ceiling is measured at; `CandidateList.as_prediction` is where a
ranking becomes a submission.
"""

from __future__ import annotations

from typing import Mapping, MutableMapping, Sequence

from .artifacts import ArtifactStore, CachedEncoder, load_responses, log_rejections
from .config import RETRIEVER_NAMES, ExperimentConfig
from .contracts import CandidateList, Code, Record, VocabularyEntry
from .stages import (
    adjudicator,
    encoders,
    fusion,
    group_prior,
    indexes,
    label_text,
    reranker,
    retrievers,
)
from .stages.adjudicator import LanguageModel
from .stages.encoders import Encoder
from .stages.reranker import CrossEncoder

# The retrievers that render the vocabulary. `knn` is not one of them: it
# harvests the gold subjects of neighbouring documents, so no label-text flag —
# qualifiers, definitions, translation — is a variable of a kNN-only run.
READS_LABEL_TEXT = ("dense", "lexical")


def predict(
    records: Sequence[Record],
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    index_records: Sequence[Record],
    store: ArtifactStore,
    device: str = "auto",
    encoder: Encoder | None = None,
    cross_encoder: CrossEncoder | None = None,
    language_model: LanguageModel | None = None,
    name_qualifiers: Mapping[str, str] | None = None,
    translations: Mapping[str, str] | None = None,
    prior: group_prior.GroupPrior | None = None,
    responses: MutableMapping[str, str] | None = None,
) -> list[CandidateList]:
    """Rank vocabulary codes for each record.

    `records` are the records being predicted and `index_records` the corpus
    being retrieved from; they are separate arguments so that no record can
    contribute its own gold subjects to its own candidate set.

    `encoder`, `cross_encoder` and `language_model` are for tests, which must
    not download model weights — or spend an API budget — to assert an
    invariant that holds for any model. Left unset, the configured ones are
    loaded onto `device`. `prior` is the same for the group-prior head, and
    `responses` for the adjudicator's cache; both are otherwise read from the
    artifact store.

    With reranking off this returns the fused top `fusion.candidates`; with it
    on, the reranked `reranker.output_k`, which is the submission's 50.
    Adjudication reorders the routed records and changes neither length. The
    recall ceiling is therefore a property of the candidate stage and is
    measured through `retrieve`, not here.
    """
    if config.reranker.enabled and cross_encoder is None:
        # Checked before any encoding: an unregistered or misspelt reranker
        # would otherwise fail after the index, the label tower and all three
        # retrievers had been paid for, which on rung 2 is hours.
        reranker.resolve(config.reranker)

    if config.adjudication.enabled and language_model is None:
        # The same check, and one more reason for it: this stage's model is
        # reached over an API, so an unregistered name, a model outside the
        # cutoff without the appendix flag, an unknown provider or a missing
        # key should all cost a sentence rather than a full retrieval pass.
        # The client itself is built after retrieval, where it is used; what
        # cannot wait is whether it could be built at all.
        adjudicator.resolve(config.adjudication)
        adjudicator.check_credentials(config.adjudication)

    if config.group_prior.enabled:
        # After the two resolutions above and before any encoding: a boosted
        # config whose head has not been fitted would otherwise index, retrieve
        # and fuse, and only then discover that the stage it was run for cannot
        # run — and this branch loads the encoder, so every check that can be
        # made without weights has to be made before it.
        if prior is None:
            from .artifacts import load_group_prior

            prior = load_group_prior(store, config)
        if encoder is None:
            # Loaded here rather than inside `retrieve` so the prior and the
            # retrievers share one set of weights: the prior reads the same
            # document vectors the retrievers do, and the cache serves them to
            # whichever asks second.
            from .hardware import select_device

            encoder = encoders.load(config.encoder, select_device(device))

    if config.reranker.enabled or config.adjudication.enabled:
        # Resolved once, before retrieval, and handed down: the reranker reads
        # the same label rendering the label tower does, so a kNN-only run with
        # reranking on has a label-text reader after all, and a translation map
        # loaded twice would be 3 MB read twice for one vocabulary.
        translations = _translations(
            config,
            translations,
            [name for name in RETRIEVER_NAMES if config.retrievers[name].enabled],
            also_read=True,
        )

    per_retriever = retrieve(
        records,
        config,
        vocabulary,
        index_records,
        store,
        device=device,
        encoder=encoder,
        name_qualifiers=name_qualifiers,
        translations=translations,
    )
    if not per_retriever:
        # An ablation that turns everything off is a row in the results table.
        fused = [CandidateList(record.id, ()) for record in records]
    else:
        fused = combine(per_retriever, config)

    if config.group_prior.enabled:
        fused = _boost(
            records, fused, config, vocabulary, store, device, encoder, prior
        )

    ranked = fused
    if config.reranker.enabled:
        ranked = _rerank(
            records,
            fused,
            config,
            vocabulary,
            device,
            cross_encoder,
            name_qualifiers,
            translations or {},
        )

    if not config.adjudication.enabled:
        return ranked
    return _adjudicate(
        records,
        ranked,
        fused,
        config,
        vocabulary,
        store,
        language_model,
        responses,
        name_qualifiers,
        translations or {},
    )


def retrieve(
    records: Sequence[Record],
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    index_records: Sequence[Record],
    store: ArtifactStore,
    device: str = "auto",
    encoder: Encoder | None = None,
    name_qualifiers: Mapping[str, str] | None = None,
    translations: Mapping[str, str] | None = None,
) -> dict[str, list[CandidateList]]:
    """Each enabled retriever's own ranked lists, before anything combines them.

    Public because the ablation table is a deliverable rather than a debugging
    aid: `scripts/ablate_retrievers.py` runs the retrievers once and fuses every
    subset of them, which is seven rows for one pass over the index instead of
    seven passes. `predict` is this followed by `combine`, so no row in that
    table comes from a code path the pipeline does not use.
    """
    enabled = [name for name in RETRIEVER_NAMES if config.retrievers[name].enabled]
    if not enabled:
        return {}

    translations = _translations(config, translations, enabled)

    # The embedding cache is keyed on the config with its encoder revision
    # resolved to the registry's pin, not on the config as written: a config
    # file carries no revision, so every revision of one model would otherwise
    # share one cache entry and a re-pinned model would read back the old
    # weights' vectors and report them as the new ones. An encoder handed in by
    # a caller is exempt — it may not be the model the config names at all, so
    # the registry has nothing to say about what produced those vectors.
    keyed_as = config
    if encoder is None:
        from .hardware import select_device

        encoder = encoders.load(config.encoder, select_device(device))
        keyed_as = encoders.pinned(config)
    encoder = CachedEncoder(encoder, store, keyed_as)

    return {
        name: _restrict(
            _build(
                name, config, vocabulary, index_records, encoder,
                name_qualifiers, translations,
            ).retrieve(records),
            vocabulary,
        )
        for name in enabled
    }


def _boost(
    records: Sequence[Record],
    fused: Sequence[CandidateList],
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    store: ArtifactStore,
    device: str,
    encoder: Encoder | None,
    prior: group_prior.GroupPrior,
) -> list[CandidateList]:
    """The fused candidates, reordered towards the groups the prior finds likely.

    Applied after fusion and before reranking, so the prior adjusts candidate
    scores rather than the candidate set: it cannot change the recall ceiling,
    only which of those candidates the reranker and the submission see first.

    The head is resolved by the caller — `predict` reads it from the artifact
    store before retrieving, for the reason the translation cache is loaded
    rather than requested: a harness that forgot it would score an unboosted
    run under a boosted config's name.
    """
    if encoder is None:
        from .hardware import select_device

        encoder = encoders.load(config.encoder, select_device(device))
    cached = CachedEncoder(encoder, store, config)

    # The same texts the two embedding retrievers were handed, so this is a
    # cache hit wherever either of them ran. A lexical-only run embeds no
    # record, and there the boost pays for the split itself.
    vectors = cached.encode_documents([record.text for record in records])
    return group_prior.apply_prior(
        fused,
        prior.distributions(vectors),
        group_prior.group_of_code(vocabulary.values()),
        config.group_prior,
    )


def _rerank(
    records: Sequence[Record],
    fused: Sequence[CandidateList],
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    device: str,
    cross_encoder: CrossEncoder | None,
    name_qualifiers: Mapping[str, str] | None,
    translations: Mapping[str, str],
) -> list[CandidateList]:
    """The fused candidates, reordered by a cross-encoder and cut to 50.

    The label text is rendered here from the same flags the label tower reads,
    rather than being handed down from retrieval: with kNN alone no retriever
    renders anything, and the reranker still has to be told what a code says.
    """
    if cross_encoder is None:
        from .hardware import select_device

        cross_encoder = reranker.load(config.reranker, select_device(device))

    return reranker.rerank(
        records,
        fused,
        reranker_label_texts(config, vocabulary, name_qualifiers, translations),
        config.reranker,
        cross_encoder,
    )


def _adjudicate(
    records: Sequence[Record],
    ranked: Sequence[CandidateList],
    fused: Sequence[CandidateList],
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    store: ArtifactStore,
    language_model: LanguageModel | None,
    responses: MutableMapping[str, str] | None,
    name_qualifiers: Mapping[str, str] | None,
    translations: Mapping[str, str],
) -> list[CandidateList]:
    """The rankings, with the least-confident share reordered by a language model.

    Routing reads the confidence of the *fused* ranking rather than of the
    ranking being reordered, and that is the whole of the decision. Ticket 12
    measured the three available signals against per-record precision: the fused
    ranking's mean reciprocal-rank score correlates +0.41, the fused-with-
    reranker ranking +0.37, and the cross-encoder's own relevance −0.05 — its
    most confident decile has two thirds of its records with no correct label at
    all. Under `reranker.mix: replace` the scores on `ranked` *are* that bare
    relevance, so routing on the ranking in hand would send the model the
    records the cross-encoder was surest about, which are the ones it ruined.

    The model is shown the same field-marked label text the label tower renders,
    for the reason the reranker is: the codes a run offers and the text it
    offers them as should be one decision, not three.

    The response cache and the rejection log are keyed on the configuration with
    the model's pin resolved, so re-pinning the adjudicator is a new cache
    rather than the old answers under a new model's name. A caller that handed
    in its own model is exempt: the registry has nothing to say about what
    produced those responses.
    """
    keyed_as = config
    if language_model is None:
        language_model = adjudicator.load(config.adjudication)
        keyed_as = adjudicator.pinned(config)
    if responses is None:
        responses = load_responses(store, keyed_as)

    routed = set(
        adjudicator.route(
            ranked,
            config.adjudication,
            confidences=[reranker.confidence(result) for result in fused],
        )
    )
    by_id = {record.id: record for record in records}
    lists = [result for result in ranked if result.record_id in routed]
    results = adjudicator.adjudicate(
        [by_id[result.record_id] for result in lists],
        lists,
        label_texts(config, vocabulary, name_qualifiers, translations),
        config.adjudication,
        model=language_model,
        responses=responses,
    )
    # Written before the rankings are applied, so an interrupted run still
    # leaves the evidence of what the model was refused for.
    log_rejections(store, keyed_as, results)
    return adjudicator.apply(ranked, results, config.adjudication)


def reranker_label_texts(
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    name_qualifiers: Mapping[str, str] | None,
    translations: Mapping[str, str],
) -> dict[Code, str]:
    """What the cross-encoder is shown for a candidate, per `reranker.label_form`.

    A separate decision from what the label tower reads, and public for the same
    reason `label_texts` is: `scripts/rerank_report.py` screens both forms, and a
    screen that rendered them its own way would be measuring a text no run uses.
    """
    if config.reranker.label_form == "name":
        rendered = label_text.render_names(
            vocabulary.values(),
            config.label_text.qualifiers,
            name_qualifiers,
            translations,
        )
        return {entry.code: entry.text for entry in rendered}
    return label_texts(config, vocabulary, name_qualifiers, translations)


def _translations(
    config: ExperimentConfig,
    supplied: Mapping[str, str] | None,
    enabled: Sequence[str],
    also_read: bool = False,
) -> Mapping[str, str]:
    """The English half of the label text, or nothing when nobody reads it.

    The frozen cache is loaded here rather than asked of the caller, because a
    caller that forgot it would render German-only under `bilingual: true` and
    report the German-only numbers as the bilingual ones — the failure the
    ticket-08 refusal existed to prevent, arriving by omission instead. Passing
    a map explicitly overrides the load.

    It is skipped when nothing would read it: `bilingual: false`, and a run of
    the neighbour retriever alone, which harvests subjects from documents and
    never renders a label. So the ablation runs on a checkout that has no cache,
    and a kNN rung does not open a 3 MB file to ignore it.

    `also_read` is how a reader outside the retrievers says so. The reranker is
    one: it scores the document against a candidate's label text, so a kNN-only
    run with reranking on renders the vocabulary after all.
    """
    if not config.label_text.bilingual:
        return {}
    if not also_read and not [name for name in enabled if name in READS_LABEL_TEXT]:
        return {}
    if supplied is not None:
        return supplied

    from .corpus import load_label_translations

    return load_label_translations()


def _build(
    name: str,
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    index_records: Sequence[Record],
    encoder: Encoder,
    name_qualifiers: Mapping[str, str] | None,
    translations: Mapping[str, str],
) -> retrievers.Retriever:
    settings = config.retrievers[name]

    if name == "knn":
        # The one retriever that reads no label text: it harvests the subjects
        # of neighbouring documents, so translating the vocabulary cannot reach
        # it and `bilingual` is not a variable of a kNN-only run.
        selected = indexes.select_documents(index_records, config.index)
        index = indexes.build_document_index(selected, encoder, vocabulary)
        return retrievers.NeighbourRetriever(index, encoder, settings)

    texts = label_texts(config, vocabulary, name_qualifiers, translations)

    if name == "dense":
        index = indexes.build_label_index(
            list(texts), list(texts.values()), encoder
        )
        return retrievers.DenseLabelRetriever(index, encoder, settings)

    variants = _label_variants(config, vocabulary, name_qualifiers, translations)
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


def label_texts(
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    name_qualifiers: Mapping[str, str] | None,
    translations: Mapping[str, str],
) -> dict[Code, str]:
    """The vocabulary rendered under this config's label-text flags.

    Public because `scripts/rerank_report.py` reranks a candidate set it fused
    itself and has to show the cross-encoder the same rendering the pipeline
    would: a report that rendered labels its own way would be measuring a label
    text no run uses.
    """
    rendered = label_text.render(
        vocabulary.values(),
        config.label_text.qualifiers,
        name_qualifiers,
        config.label_text.include_definition,
        translations,
    )
    return {entry.code: entry.text for entry in rendered}


def _label_variants(
    config: ExperimentConfig,
    vocabulary: dict[str, VocabularyEntry],
    name_qualifiers: Mapping[str, str] | None,
    translations: Mapping[str, str],
) -> dict[Code, tuple[str, ...]]:
    return label_text.variants(
        vocabulary.values(),
        config.label_text.qualifiers,
        name_qualifiers,
        translations,
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


def combine(
    per_retriever: Mapping[str, Sequence[CandidateList]],
    config: ExperimentConfig,
) -> list[CandidateList]:
    """One ranked list per record, from however many retrievers ran.

    A single retriever needs no fusion, and saying so here keeps ticket 04's
    number free of a rank-fusion transform that has nothing to fuse: reciprocal
    rank fusion over one list is a monotone re-scoring, so it cannot reorder
    anything, but it would replace that retriever's own scores — which the
    confidence analysis and the reranker read — with a reciprocal of a rank.
    """
    if len(per_retriever) == 1:
        only = next(iter(per_retriever.values()))
        return [result.top(config.fusion.candidates) for result in only]

    weights = {
        name: config.retrievers[name].weight for name in per_retriever
    }
    return fusion.fuse(per_retriever, weights, config.fusion)
