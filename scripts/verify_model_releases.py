"""Verify every model this project loads against the 2025-01-31 cutoff.

    python scripts/verify_model_releases.py            # re-check against the hub
    python scripts/verify_model_releases.py --offline   # check the file alone
    python scripts/verify_model_releases.py --force     # rewrite the reference

docs/spec.md restricts every component to models released on or before
2025-01-31 (story 2). This is where that stops being a comment: it reads the
creation date and the commit history of each model from the Hugging Face API,
pins each one to the newest commit dated on or before the cutoff, and writes
`reference/model_releases.json`, which `llms4subjects.models` reads and every
loader checks.

Pinning the revision rather than only recording the release date is the part
that matters. `intfloat/multilingual-e5-base` was created in May 2023 and had a
commit landed on it in April 2026; a run that names the model without a revision
loads the 2026 state of it, and the cutoff claim would be false while every date
in the registry stayed true.

Nothing else in the project reaches the hub API, and no run reads it: the
registry is a frozen reference, like the bands and the translation cache, so
re-verification is a deliberate command that leaves a commit rather than a
by-product of an experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llms4subjects.models import (  # noqa: E402
    MODEL_CUTOFF,
    SCHEMA,
    MissingModelRegistry,
    load_registry,
)
from llms4subjects.paths import MODEL_RELEASES_FILE  # noqa: E402

API = "https://huggingface.co/api/models"
HUB = "https://huggingface.co"

TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class Candidate:
    """A model the project loads, and what it is loaded for.

    This tuple is the list of models the project is allowed to use; the registry
    is derived from it, so adding a model means adding a line here and running
    the script, which is a commit rather than an edit to generated data.
    """

    name: str
    role: str
    notes: str


# The four encoders ticket 09 screens, plus the models the rest of the
# repository already loads, all of them fetched from the hub. `role` is what the
# model is loaded for, not a capability of the model. The adjudicator's hosted
# models (ticket 13) are declared in `API_MODELS` instead, since they have no
# repository to read.
CANDIDATES = (
    Candidate(
        "intfloat/multilingual-e5-base",
        "encoder",
        "Screening baseline through tickets 04-08. E5's `query:`/`passage:` "
        "prefixes are handled by `stages.encoders.PREFIXES`.",
    ),
    Candidate(
        "intfloat/multilingual-e5-large",
        "encoder",
        "The same family at 2x the parameters, so the screen separates the "
        "family from the size.",
    ),
    Candidate(
        "BAAI/bge-m3",
        "encoder",
        "Named in docs/idea.md. Also the one candidate whose forward pass can "
        "emit sparse term weights; ticket 09 screens its dense tower only, so "
        "all four encoders face the same BM25 lexical retriever.",
    ),
    Candidate(
        "Alibaba-NLP/gte-multilingual-base",
        "encoder",
        "Named in docs/idea.md. Ships its own modelling code, so both the "
        "weights and that code are pinned.",
    ),
    Candidate(
        "Helsinki-NLP/opus-mt-de-en",
        "translation",
        "Wrote reference/label_translations.json once (ticket 08). An input to "
        "the pipeline rather than a dependency of it.",
    ),
    Candidate(
        "bert-base-multilingual-cased",
        "baseline",
        "The rejected classifier's base model (ticket 14). A results row, kept "
        "here so the cutoff covers every number the writeup quotes.",
    ),
    Candidate(
        "BAAI/bge-reranker-v2-m3",
        "reranker",
        "The larger of the two off-the-shelf rerankers ticket 12 screens. Same "
        "family as the bge-m3 encoder, so document and label are scored by "
        "models trained on the same multilingual mixture.",
    ),
    Candidate(
        "BAAI/bge-reranker-base",
        "reranker",
        "The smaller screened reranker (ticket 12), at half the parameters, so "
        "the screen separates the reranker's quality from its size.",
    ),
)


@dataclass(frozen=True)
class ApiModel:
    """A hosted model, which has no weights and no commits to pin.

    The adjudicator (ticket 13) calls a model over an API, so the evidence for
    its date cannot be a hub commit: there is nothing to fetch. What stands in
    its place is the provider's own dated model id — `claude-3-5-sonnet-20241022`
    is the exact string the request names, and the date in it is the release the
    cutoff is about — plus the announcement it was released in, recorded here as
    `source` so a reader can check the claim rather than take it.

    These entries are declared rather than derived, and `origin: api` in the
    registry says so. That is weaker evidence than a fetched commit date, and
    the writeup should say which kind each model has rather than presenting one
    as the other.
    """

    name: str
    provider: str
    created: str
    source: str
    notes: str
    # The one entry allowed to postdate the cutoff, and only for the
    # adjudicator's appendix row (docs/spec.md). Declaring it here is what keeps
    # `main` from refusing to write the registry over it.
    appendix: bool = False

    def entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "role": "adjudicator",
            "created": self.created,
            # The id is the pin: it is what the request names, and the provider
            # holds it to one set of weights.
            "revision": self.name,
            "revision_date": self.created,
            "source": self.source,
            "origin": "api",
            "provider": self.provider,
            "notes": self.notes,
        }
        if self.appendix:
            entry["appendix"] = True
        return entry


# The hosted models the adjudication stage may call. The headline row must use
# one released on or before the cutoff; an appendix run that quantifies model
# progress registers its current model here too, and `adjudication.appendix` is
# what allows it to be used (see `stages.adjudicator.resolve`).
API_MODELS = (
    ApiModel(
        "claude-3-5-sonnet-20241022",
        "anthropic",
        "2024-10-22",
        "https://www.anthropic.com/news/3-5-models-and-computer-use",
        "The headline adjudicator (ticket 13). Released 2024-10-22, inside the "
        "cutoff, and the date is carried by the model id the API request names.",
    ),
    ApiModel(
        "gpt-4o-2024-08-06",
        "openai",
        "2024-08-06",
        "https://openai.com/index/introducing-structured-outputs-in-the-api/",
        "The second provider path for the adjudicator, registered so that "
        "`adjudication.model` can be swapped without the registry becoming the "
        "reason it cannot. Also inside the cutoff, and dated by its id.",
    ),
)


class HubUnreachable(RuntimeError):
    """The API could not be read, so nothing can be verified against it."""


class NoRevisionInCutoff(RuntimeError):
    """The hub was read fine, and the model has nothing inside the cutoff.

    Separate from `HubUnreachable` because the two need opposite responses:
    this one is an answer, and `--offline` would not help.
    """


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="rewrite reference/model_releases.json"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="check the committed file against the cutoff without the network",
    )
    parser.add_argument("--output", help="write somewhere else (for testing)")
    args = parser.parse_args(argv)

    if args.offline:
        if args.force:
            print("--offline cannot --force: rewriting needs the hub's dates.")
            return 1
        return _offline()

    try:
        models = {
            candidate.name: _describe(candidate) for candidate in CANDIDATES
        }
        # Merged rather than fetched: a hosted model has no commit history to
        # walk, so its dates are the declarations in `API_MODELS`.
        models.update({model.name: model.entry() for model in API_MODELS})
    except NoRevisionInCutoff as error:
        print(f"{error}\nRemove it from CANDIDATES rather than recording it.")
        return 1
    except HubUnreachable as error:
        print(f"{error}\nRe-run with --offline to check the committed file alone.")
        return 1

    document = {
        "schema": SCHEMA,
        "cutoff": MODEL_CUTOFF.isoformat(),
        "verified_at": datetime.now(timezone.utc).date().isoformat(),
        "models": models,
    }

    # An appendix model postdates the cutoff on purpose; every other one that
    # does is a mistake. Without that exception the appendix row docs/spec.md
    # allows could never be registered, because this is the only writer.
    outside = [
        name
        for name, entry in models.items()
        if date.fromisoformat(entry["created"]) > MODEL_CUTOFF
        and not entry.get("appendix")
    ]
    _report(models, outside)

    if outside:
        print(
            f"\n{len(outside)} model(s) postdate the cutoff and cannot be used; "
            "remove them rather than recording them. A current model is "
            "registrable only as the adjudicator's appendix row, by declaring "
            "`appendix=True` on its API_MODELS entry."
        )
        return 1

    destination = Path(args.output) if args.output else MODEL_RELEASES_FILE
    if not args.force:
        print(f"\n{destination} not written. Re-run with --force to rewrite it.")
        return _diff(document, destination)

    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {destination}")
    return 0


def _offline() -> int:
    """Check the committed registry against the cutoff, reading no network."""
    try:
        registry = load_registry()
    except MissingModelRegistry as error:
        print(error)
        return 1

    print(f"cutoff {registry.cutoff.isoformat()}, verified {registry.verified_at}\n")
    failures = 0
    for name, entry in registry.models.items():
        late = (
            max(entry.created, entry.revision_date) > registry.cutoff
            and not entry.appendix
        )
        failures += late
        pinned = (
            f"{entry.revision} (declared, {entry.provider})"
            if entry.origin == "api"
            else f"{entry.revision[:12]} ({entry.revision_date.isoformat()})"
        )
        flag = "FAIL" if late else ("APDX" if entry.appendix else "ok  ")
        print(
            f"  {flag}  {name:<45} created "
            f"{entry.created.isoformat()}  pinned {pinned}"
        )
    if failures:
        print(f"\n{failures} entry/entries postdate the cutoff.")
    return 1 if failures else 0


def _report(models: dict[str, dict[str, Any]], outside: list[str]) -> None:
    hosted = sum(entry.get("origin") == "api" for entry in models.values())
    print(
        f"cutoff {MODEL_CUTOFF.isoformat()}, {len(models) - hosted} model(s) "
        f"from the hub and {hosted} declared hosted model(s)\n"
    )
    for name, entry in models.items():
        flag = "FAIL" if name in outside else (
            "APDX" if entry.get("appendix") else "ok  "
        )
        pinned = (
            f"{entry['revision']} (declared)"
            if entry.get("origin") == "api"
            else f"{entry['revision'][:12]} ({entry['revision_date']})"
        )
        code = ""
        if entry.get("trust_remote_code"):
            code = (
                f"  remote code {entry['code_repository']}@"
                f"{(entry.get('code_revision') or '?')[:12]}"
            )
        print(
            f"  {flag}  {name:<45} {entry['role']:<11} created "
            f"{entry['created']}  pinned {pinned}{code}"
        )


def _diff(document: dict[str, Any], destination: Path) -> int:
    """Say whether the committed file already says what the hub says."""
    if not destination.exists():
        print(f"{destination} does not exist yet.")
        return 1

    committed = json.loads(destination.read_text(encoding="utf-8"))
    # `verified_at` moves on every run and is provenance, not a fact about a
    # model, so a re-verification that changes only it is agreement.
    compared = ("cutoff", "models")
    if all(committed.get(key) == document[key] for key in compared):
        print("the committed registry agrees with the hub.")
        return 0

    for name, entry in document["models"].items():
        before = committed.get("models", {}).get(name)
        if before != entry:
            print(f"  changed: {name}")
            for field, value in sorted(entry.items()):
                if before is None or before.get(field) != value:
                    was = "absent" if before is None else repr(before.get(field))
                    print(f"      {field}: {was} -> {value!r}")
    for name in sorted(set(committed.get("models", {})) - set(document["models"])):
        print(f"  no longer a candidate: {name}")
    return 1


def _describe(candidate: Candidate) -> dict[str, Any]:
    """One registry entry, every field of it derived from the hub."""
    info = _api(f"{API}/{candidate.name}")
    revision, revision_date = _pin(candidate.name)

    entry: dict[str, Any] = {
        "role": candidate.role,
        "created": info["createdAt"][:10],
        "revision": revision,
        "revision_date": revision_date,
        "source": f"{HUB}/{candidate.name}/commit/{revision}",
        "notes": candidate.notes,
    }

    repository = _code_repository(_config(candidate.name, revision))
    if repository is not None:
        code_revision, _ = _pin(repository)
        entry["trust_remote_code"] = True
        entry["code_repository"] = repository
        entry["code_revision"] = code_revision
    return entry


def _pin(name: str) -> tuple[str, str]:
    """The newest commit dated on or before the cutoff, and its date.

    The hub serves commits newest first, fifty to a page, so this walks pages
    until one falls inside the cutoff. Paging matters rather than being
    thorough: a busy repository can carry more than fifty commits since
    2025-01-31, and reading only the first page would report that none of them
    predate it — a false "no revision is inside the cutoff" on exactly the
    models whose provenance is hardest to establish.
    """
    url = f"{API}/{name}/commits/main"
    seen = 0
    while url:
        page, url = _page(url)
        for commit in page:
            seen += 1
            when = commit["date"][:10]
            if date.fromisoformat(when) <= MODEL_CUTOFF:
                return commit["id"], when
    raise NoRevisionInCutoff(
        f"all {seen} commits on {name} postdate {MODEL_CUTOFF.isoformat()}, so "
        "no revision of it is inside the cutoff and it cannot be used"
    )


def _config(name: str, revision: str) -> dict[str, Any]:
    """The model's `config.json` at the pinned revision.

    Read at the revision rather than at `main`, so the remote-code repository
    recorded here is the one the weights a run loads would execute.
    """
    return _api(f"{HUB}/{name}/raw/{revision}/config.json")



def _code_repository(config: dict[str, Any]) -> str | None:
    """Which repository ships the modelling code, when it is not `transformers`.

    `auto_map` entries look like `Alibaba-NLP/new-impl--modeling.NewModel`; the
    part before `--` is the repository whose code gets executed.
    """
    for target in (config.get("auto_map") or {}).values():
        if "--" in str(target):
            return str(target).split("--", 1)[0]
    return None


def _api(url: str) -> Any:
    return _page(url)[0]


def _page(url: str) -> tuple[Any, str | None]:
    """One response body, and the URL of the next page if the hub offers one."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read()), _next_link(response.headers)
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise HubUnreachable(f"cannot read {url}: {error}") from None


def _next_link(headers) -> str | None:
    """The `rel="next"` URL from an RFC 5988 `Link` header, if there is one."""
    for part in (headers.get("Link") or "").split(","):
        target, _, attributes = part.partition(">")
        if 'rel="next"' in attributes and "<" in target:
            return target[target.index("<") + 1:]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
