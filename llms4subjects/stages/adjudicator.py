"""LLM selection over a candidate list, with hard constraint validation.

The last stage, and the only one that costs money per record, so it runs on the
records where it can pay: the least-confident `route_fraction` of the split, as
measured by `reranker.confidence` over whatever ranking the pipeline produced.
Everything else is returned exactly as it arrived.

## The constraint is the stage

A generative model asked for GND codes freely produces identifiers that look
entirely plausible and do not exist — `gnd:4043751-6` is as convincing as
`gnd:4043744-9` and is nobody's subject heading. So the model is never asked to
author an identifier. It is shown the top `candidates` codes with their label
text and asked to choose among them, and its answer is checked against the set
it was offered: a response naming anything else is rejected whole, logged, and
the record keeps the ranking it arrived with. Rejecting the whole response
rather than the offending code is deliberate — a model that invented one
identifier is not evidence about the ones it did not invent.

That makes the stage a reordering and nothing else. It cannot introduce a code,
cannot drop one, and cannot change the length of the list, so the 50-code output
contract survives whatever the model says, including nothing at all.

## What is cached, and why it has to be

Responses are cached by record id under a key that covers the model, the prompt
revision and every knob that changes the prompt, so a re-score of a run that has
already been paid for costs nothing. The cache is written through as each
response arrives rather than at the end of the pass: an interrupted run of a
thousand records must not throw away the nine hundred it has already bought.

Implemented by ticket 13.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Protocol, Sequence

from ..config import AdjudicationConfig
from ..contracts import Candidate, CandidateList, Code, Record
from ..models import MODEL_CUTOFF, ModelRelease, ModelTooRecent, Registry, release
from .reranker import confidence

# Which prompt a run uses. A revision is part of the cache key, so adding one
# here is what lets a reworded prompt be measured against the old one instead of
# silently reusing its answers. `config.prompt_revision` is checked against this
# tuple, which is why it lives at module scope rather than inside the builder.
PROMPT_REVISIONS = ("v1",)

# Where each provider's chat endpoint is, and which environment variable carries
# the key. Recorded here rather than in a config file: they are facts about the
# provider, not knobs of an experiment.
PROVIDERS = {
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "key": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "key": "OPENAI_API_KEY",
    },
}

ANTHROPIC_VERSION = "2023-06-01"

TIMEOUT_SECONDS = 120

# Transient HTTP statuses worth asking again about: rate limits and the
# provider's own overload and gateway errors. A pass is a thousand calls, so a
# single 429 must not cost the run — and everything else is a real answer.
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})

RETRIES = 4

# Seconds before the first retry; doubled each time.
BACKOFF_SECONDS = 2.0


class ConstraintViolation(ValueError):
    """The model returned an identifier that was not in the candidate set."""


class MalformedResponse(ValueError):
    """The response holds no list of codes at all."""


class MissingApiKey(RuntimeError):
    """The provider's key is not in the environment, so nothing can be asked."""


@dataclass(frozen=True)
class Adjudication:
    """What the model said about one record, or why it was not taken.

    `codes` is the accepted ordering of the offered candidates. A rejected
    adjudication carries no codes and a `reason` that names what was wrong, so
    the run's log says which records were affected and by what.
    """

    record_id: str
    codes: tuple[Code, ...]
    rejected: bool = False
    reason: str = ""


@dataclass(frozen=True)
class Resolved:
    """What an adjudication config resolves to once the registry has had its say.

    `appendix` is the one place in this project where a post-cutoff model is
    allowed, and it is allowed only because it is declared: docs/spec.md permits
    a single clearly-labelled appendix row quantifying what model progress adds
    to an otherwise identical pipeline, in this stage alone.
    """

    release: ModelRelease
    revision: str
    provider: str
    appendix: bool


def resolve(
    config: AdjudicationConfig, registry: Registry | None = None
) -> Resolved:
    """Check the adjudicator against the 2025-01-31 cutoff, or against the flag.

    Refuses a model the registry does not vouch for, on the same terms as the
    encoder's and the reranker's: a name alone carries no date, and the headline
    number is only comparable to a January-2025 leaderboard if every model in
    the pipeline predates it.

    A model outside the cutoff needs `adjudication.appendix`, and a model inside
    it may not carry that flag — otherwise the appendix row and the headline row
    would be told apart by prose rather than by configuration.
    """
    from ..config import ConfigError

    if not config.model:
        raise ValueError("adjudication.model names no model to call")

    entry = release(config.model, registry)
    cutoff = registry.cutoff if registry is not None else MODEL_CUTOFF
    inside = entry.created <= cutoff and entry.revision_date <= cutoff

    if inside and config.appendix:
        raise ConfigError(
            f"{config.model} was released {entry.created.isoformat()}, inside "
            f"the {cutoff.isoformat()} cutoff, so it is a headline model. "
            "adjudication.appendix marks the one row allowed to use a model "
            "the cutoff does not cover; drop it."
        )
    if not inside and not config.appendix:
        raise ModelTooRecent(
            f"{config.model} was released {entry.created.isoformat()}, after "
            f"the {cutoff.isoformat()} model cutoff docs/spec.md declares. "
            "Only the appendix row may use a current model, and only in this "
            "stage: set adjudication.appendix true and label the row."
        )
    if not entry.provider:
        raise ConfigError(
            f"{config.model} is registered without an api provider, so nothing "
            "knows where to send a prompt. Register it as an API model."
        )
    if entry.provider not in PROVIDERS:
        # Otherwise this is a `KeyError` on an endpoint lookup, raised after
        # the index, the retrievers and the reranker have all been paid for.
        raise ConfigError(
            f"{config.model} is registered against provider "
            f"{entry.provider!r}, which this stage cannot call "
            f"(known: {', '.join(sorted(PROVIDERS))})"
        )
    return Resolved(
        release=entry,
        revision=entry.revision,
        provider=entry.provider,
        appendix=not inside,
    )


def pinned(config):
    """The experiment config with its adjudicator revision resolved to the pin.

    The counterpart of `stages.reranker.pinned`. The response cache is keyed on
    the `adjudication` section, and a config file carries no revision, so
    without this every revision of one model would share one set of answers.

    Takes and returns an `ExperimentConfig`; typed loosely to keep this module
    off the config module's inner shape.
    """
    import dataclasses

    resolved = resolve(config.adjudication)
    if config.adjudication.revision == resolved.revision:
        return config
    return dataclasses.replace(
        config,
        adjudication=dataclasses.replace(
            config.adjudication, revision=resolved.revision
        ),
    )


class LanguageModel(Protocol):
    """A model that answers one prompt with one string.

    One record per call, because that is the unit the cache and the constraint
    are defined over: a batched call that failed halfway would leave responses
    that cannot be attributed to a record, and a response that mixed two
    records' candidates could not be validated against either.
    """

    def complete(self, prompt: str) -> str: ...


class HttpLanguageModel:
    """A provider's chat endpoint behind the protocol above, over the standard
    library.

    No SDK: the request is one JSON POST and the response one string, and a
    pinned vendor client would be a dependency this project does not otherwise
    need on either host. The payload and the extraction are separated from the
    call so both are testable without a key or a network.
    """

    def __init__(self, provider: str, model: str, config: AdjudicationConfig):
        self._provider = provider
        self._model = model
        self._config = config
        self._key = _api_key(provider)

    def complete(self, prompt: str) -> str:
        """One prompt, one answer, retrying only what is worth retrying.

        A rate limit or a provider overload is not an answer, so it is asked
        again with a doubling backoff; anything else — a bad key, a model name
        the provider does not know, a malformed envelope — stops the pass.
        That asymmetry is deliberate: a failed call bought nothing and is not a
        constraint violation, so turning one into a rejection would file a
        systemic failure as a thousand records the model got wrong. Everything
        already paid for is in the cache, and a re-run resumes from it.
        """
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            PROVIDERS[self._provider]["url"],
            data=json.dumps(
                _payload(self._provider, self._model, prompt, self._config)
            ).encode(),
            headers=_headers(self._provider, self._key),
            method="POST",
        )
        for attempt in range(RETRIES):
            try:
                with urllib.request.urlopen(
                    request, timeout=TIMEOUT_SECONDS
                ) as response:
                    return _text(self._provider, json.loads(response.read()))
            except urllib.error.HTTPError as error:
                if error.code not in RETRY_STATUSES or attempt == RETRIES - 1:
                    raise
            except urllib.error.URLError:
                if attempt == RETRIES - 1:
                    raise
            time.sleep(BACKOFF_SECONDS * 2**attempt)
        raise AssertionError("unreachable: the loop returns or raises")


def check_credentials(config: AdjudicationConfig) -> None:
    """Fail now if a pass would fail on its first call.

    Called by the pipeline before any encoding. The model itself is built after
    retrieval, where it is needed, but the key it will read is in the
    environment or is not, and discovering that after an hour of indexing is the
    same mistake as discovering an unregistered model name there.

    Skipped when nothing will be routed: `route_fraction: 0` is the free
    ablation — the stage runs, bills nothing and changes no ranking — and it
    should not need a credential to do nothing.
    """
    if config.route_fraction <= 0:
        return
    _api_key(resolve(config).provider)


def load(config: AdjudicationConfig) -> LanguageModel:
    """Build the adjudicator named by the config.

    Called by the pipeline rather than by `adjudicate`, for the same reason the
    reranker's loader is: a test must be able to assert this stage's contract
    without a key, a network or a bill.
    """
    resolved = resolve(config)
    return HttpLanguageModel(resolved.provider, resolved.revision, config)


def _api_key(provider: str) -> str:
    variable = PROVIDERS[provider]["key"]
    key = os.environ.get(variable)
    if not key:
        raise MissingApiKey(
            f"{variable} is not set, and the adjudication stage calls "
            f"{provider}. Export it, or run with adjudication.enabled false."
        )
    return key


def _headers(provider: str, key: str) -> dict[str, str]:
    if provider == "anthropic":
        return {
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
    return {"content-type": "application/json", "authorization": f"Bearer {key}"}


def _payload(
    provider: str, model: str, prompt: str, config: AdjudicationConfig
) -> dict:
    """The request body, as data, so a test can assert it without a network."""
    body = {
        "model": model,
        "temperature": config.temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if provider == "anthropic":
        body["max_tokens"] = config.max_output_tokens
    else:
        body["max_completion_tokens"] = config.max_output_tokens
    return body


def _text(provider: str, response: Mapping) -> str:
    """The assistant's text, or a `MalformedResponse` naming what came back."""
    try:
        if provider == "anthropic":
            return "".join(
                block["text"]
                for block in response["content"]
                if block.get("type") == "text"
            )
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as error:
        raise MalformedResponse(
            f"the {provider} response carries no message text: {error}"
        ) from None


def route(
    candidates: Sequence[CandidateList],
    config: AdjudicationConfig,
    confidences: Sequence[float] | None = None,
) -> list[str]:
    """Record ids of the least-confident `route_fraction` of records.

    The count is floored rather than rounded, because the fraction is a budget:
    a run asked for 20% must not bill for 21%.

    `confidences` overrides the measure taken from `candidates` themselves, and
    that is not a convenience. Ticket 12 measured the fused ranking's confidence
    to correlate +0.41 with per-record P@5 and the cross-encoder's own relevance
    −0.05 — its most confident decile has two thirds of its records with no
    correct label at all — so a pipeline that reranks should still be able to
    route on the ranking that calibrates. Ties break by record id, so the routed
    set does not depend on the order a split happened to be read in.
    """
    if not candidates or config.route_fraction <= 0:
        return []

    measured = (
        list(confidences)
        if confidences is not None
        else [confidence(result) for result in candidates]
    )
    if len(measured) != len(candidates):
        raise ValueError(
            f"{len(measured)} confidences against {len(candidates)} candidate "
            "lists; routing is positional"
        )

    ranked = sorted(
        zip(candidates, measured), key=lambda pair: (pair[1], pair[0].record_id)
    )
    # Floored, with a tolerance: `0.29 * 100` is 28.999999999999996 in binary
    # floating point, and a budget that quietly routed 28 records where 29 were
    # asked for would be a rounding bug reported as a routing decision.
    routed = math.floor(config.route_fraction * len(candidates) + 1e-9)
    return [result.record_id for result, _ in ranked[:routed]]


def prompt(
    record: Record,
    offered: Sequence[Candidate],
    label_texts: Mapping[Code, str],
    config: AdjudicationConfig,
) -> str:
    """What the model is shown for one record, per `adjudication.prompt_revision`."""
    return _PROMPTS[config.prompt_revision](record, offered, label_texts, config)


def _prompt_v1(
    record: Record,
    offered: Sequence[Candidate],
    label_texts: Mapping[Code, str],
    config: AdjudicationConfig,
) -> str:
    """The record, the candidates, and the rule the answer is checked against.

    The candidates are numbered and carry their label text, because a bare code
    says nothing a model can reason over; the codes are what the answer must be
    written in, because they are what the submission takes.
    """
    listed = "\n".join(
        f"{position}. {candidate.code}  {_one_line(label_texts, candidate.code)}"
        for position, candidate in enumerate(offered, start=1)
    )
    return (
        "You are a subject librarian assigning GND subject headings to a "
        "technical library record.\n\n"
        "Record\n"
        f"  Type:     {record.type}\n"
        f"  Language: {record.lang}\n"
        f"  Title:    {record.title}\n"
        f"  Abstract: {_clipped(record.abstract, config.document_chars)}\n\n"
        f"Candidate subject headings, in the order a retrieval system ranked "
        f"them:\n{listed}\n\n"
        f"Choose the {config.select} candidates that best describe this "
        "record, most relevant first. Prefer headings that name what the "
        "record is about over headings that merely appear in its text.\n\n"
        "Answer with a JSON array of the chosen codes and nothing else, for "
        'example ["gnd:4043744-9", "gnd:4030309-2"]. Every code must be copied '
        "exactly from the list above; do not write a code that is not on it."
    )


_PROMPTS = {"v1": _prompt_v1}


def _one_line(label_texts: Mapping[Code, str], code: Code) -> str:
    """A candidate's label text on one line of the prompt.

    The rendering is field-marked and multi-line for an encoder; a prompt is a
    numbered list, so the fields are joined rather than reformatted — the model
    reads the same text the retrievers scored.
    """
    try:
        text = label_texts[code]
    except KeyError:
        raise KeyError(
            f"no label text for candidate {code!r}; the adjudicator shows the "
            "same rendering the label tower reads, so every candidate needs one"
        ) from None
    return " · ".join(line.strip() for line in text.splitlines() if line.strip())


def _clipped(text: str, limit: int) -> str:
    """The abstract, cut to what a prompt pays for.

    TIBKAT abstracts run to a few thousand characters and the routed fifth of a
    split is a thousand prompts, so the tail of a long abstract is a cost
    decision rather than an oversight. It is a config knob and part of the
    cache key, so changing it is a new measurement rather than a quiet one.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + " […]"


def parse(response: str) -> tuple[Code, ...]:
    """The codes a response names, or `MalformedResponse` if it names none.

    Prose around the array and a markdown fence around that are both tolerated:
    neither is a constraint violation, and refusing them would reject responses
    whose content is perfectly valid.

    Every `[` is tried as the start of an array rather than the first one being
    assumed to be it, because prose brackets are common — "[see below]", a
    citation, a bracketed aside — and a regular expression that took the first
    pair would read the aside and reject the answer beside it. A rejection is
    cached like any other response, so that mistake would be permanent for the
    configuration that made it.
    """
    decoder = json.JSONDecoder()
    for index, character in enumerate(response):
        if character != "[":
            continue
        try:
            value, _ = decoder.raw_decode(response, index)
        except ValueError:
            continue
        if isinstance(value, list) and all(
            isinstance(code, str) for code in value
        ):
            return tuple(value)
    raise MalformedResponse(
        f"no JSON array of codes in the response: {response[:200]!r}"
    )


def validate(offered: Sequence[Code], returned: Sequence[Code]) -> tuple[Code, ...]:
    """Return `returned` if it is a subset of `offered`, else raise.

    The one check the whole stage exists for.
    """
    outside = [code for code in returned if code not in set(offered)]
    if outside:
        raise ConstraintViolation(
            f"the model returned {len(outside)} identifier(s) it was not "
            f"offered: {', '.join(sorted(set(outside)))}"
        )
    return tuple(returned)


def adjudicate(
    records: Sequence[Record],
    candidates: Sequence[CandidateList],
    label_texts: Mapping[Code, str],
    config: AdjudicationConfig,
    model: LanguageModel | None = None,
    responses: MutableMapping[str, str] | None = None,
) -> list[Adjudication]:
    """Select from the offered candidates, rejecting out-of-set identifiers.

    `records` and `candidates` are the routed subset, matched by position; a
    mismatch is an error rather than a positional guess, as everywhere else in
    the pipeline.

    `responses` is the cache: a response per record id. A record already in it
    is replayed and never re-billed, and a response the model returns is written
    back as it arrives — including a rejected one, which was paid for and would
    otherwise be bought again to be told the same thing.
    """
    if not records:
        return []
    _check_alignment(records, candidates)
    responses = {} if responses is None else responses

    results: list[Adjudication] = []
    for record, offered in zip(records, candidates):
        head = offered.candidates[: config.candidates]
        response = responses.get(record.id)
        if response is None:
            if model is None:
                raise ValueError(
                    f"record {record.id!r} has no cached response and no model "
                    "to ask; adjudication needs one or the other"
                )
            response = model.complete(prompt(record, head, label_texts, config))
            responses[record.id] = response
        results.append(_adjudication(record.id, head, response))
    return results


def _adjudication(
    record_id: str, offered: Sequence[Candidate], response: str
) -> Adjudication:
    """One response, checked. A failure of either check rejects the whole thing."""
    codes = [candidate.code for candidate in offered]
    try:
        returned = validate(codes, parse(response))
    except (ConstraintViolation, MalformedResponse) as error:
        return Adjudication(record_id, (), rejected=True, reason=str(error))
    # A repeated code is sloppiness rather than an invented identifier, so it
    # costs its duplicate and not the response.
    return Adjudication(record_id, tuple(dict.fromkeys(returned)))


def apply(
    candidates: Sequence[CandidateList],
    adjudications: Sequence[Adjudication],
    config: AdjudicationConfig,
) -> list[CandidateList]:
    """The rankings, with each adjudicated record reordered and the rest as they
    were.

    `adjudications` covers the routed subset only, so this is where a routed
    ranking rejoins the split it came from. A record the model was not shown, or
    whose response was rejected, is returned untouched.

    Scores are the incoming scores read positionally: the model returns an
    order, not a relevance, and the ranking is the only thing this stage has an
    opinion about. Keeping the incoming score sequence keeps the list
    non-increasing and keeps the confidence measure in the units the coverage
    curve and the routing threshold read it in.
    """
    by_record = {
        result.record_id: result for result in adjudications if not result.rejected
    }
    return [
        _reordered(result, by_record[result.record_id].codes, config)
        if result.record_id in by_record
        else result
        for result in candidates
    ]


def _reordered(
    result: CandidateList, chosen: Sequence[Code], config: AdjudicationConfig
) -> CandidateList:
    """One record's candidates, with the chosen ones brought to the front.

    Nothing is added and nothing is dropped: the codes the model chose lead, in
    its order; the rest of the offered window follows in the order it had; the
    candidates below the window are untouched, because the model never saw them
    and has no opinion to apply.
    """
    if not chosen:
        return result

    by_code = {candidate.code: candidate for candidate in result.candidates}
    offered = [candidate.code for candidate in result.candidates[: config.candidates]]
    lead = [code for code in chosen if code in by_code]
    order = lead + [code for code in offered if code not in set(lead)]
    order += [candidate.code for candidate in result.candidates[config.candidates :]]

    scores = [candidate.score for candidate in result.candidates]
    return CandidateList(
        record_id=result.record_id,
        candidates=tuple(
            Candidate(
                code=code,
                score=score,
                # The retrievers that proposed it, unchanged. The adjudicator
                # is not a candidate source and does not claim to be one.
                sources=dict(by_code[code].sources),
            )
            for code, score in zip(order, scores)
        ),
    )


def _check_alignment(
    records: Sequence[Record], candidates: Sequence[CandidateList]
) -> None:
    if len(records) != len(candidates):
        raise ValueError(
            f"{len(records)} records against {len(candidates)} candidate lists; "
            "adjudication is positional"
        )
    for record, result in zip(records, candidates):
        if record.id != result.record_id:
            raise ValueError(
                f"record {record.id!r} was handed the candidates of "
                f"{result.record_id!r}; adjudication is positional and both "
                "sequences come from the same routing pass"
            )
