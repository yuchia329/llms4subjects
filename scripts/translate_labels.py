"""Translate the vocabulary's German label strings to English, once.

    python scripts/translate_labels.py            # translate whatever is missing
    python scripts/translate_labels.py --report   # say what is missing, write nothing
    python scripts/translate_labels.py --force    # re-translate every string

All label text in the release is German, and 58.7% of gold label assignments
belong to English documents, so document-to-label matching is cross-lingual for
most of the benchmark. Translating the labels once and committing the result is
what turns that from a per-run cost into a lookup:
`llms4subjects.stages.label_text` reads the cache and never loads a model, so
indexing and evaluation pay nothing for the second language.

The cache is a plain German-string to English-string map. Not keyed by GND code,
because two senses of one word — `Interaktion,Naturwissenschaft` and
`Interaktion,Soziologie` — share the word they disambiguate, and the qualifier
that tells them apart is a string of its own. 79,427 entries reduce to 79,224
strings that way, and the renderer composes them back under whichever qualifier
mode is configured.

Which strings are needed is `label_text.translatable`, not a decision taken
here: the module that renders the cache decides what the cache has to cover.

Idempotent by construction. The default run translates only the strings the
cache does not already hold, so a second run finds nothing to do and writes
nothing. `--force` is the deliberate act that replaces a cache built by one
model with one built by another, and leaves a commit saying so.

The translating model is an input, not a dependency: `produced_by` in the file
records which one wrote it, and nothing in the pipeline reads that block.
Running this needs `sentencepiece`, which is why it is in requirements/base.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llms4subjects.corpus import (  # noqa: E402
    MissingDataset,
    MissingTranslations,
    load_label_translations,
    load_name_qualifiers,
    load_vocabulary,
)
from llms4subjects.paths import LABEL_TRANSLATIONS_FILE  # noqa: E402
from llms4subjects.stages.label_text import translatable  # noqa: E402

# Predictions are restricted to the tib-core vocabulary whatever the index
# holds, so the all-subjects names are never rendered and never translated.
VOCABULARY = "tib-core"

# Opus-MT de-en: a 74M-parameter supervised MT model published in 2020, well
# inside the 2025-01-31 model cutoff docs/spec.md declares, and small enough to
# translate the whole vocabulary on the laptop in two minutes. An instruction
# model would translate short headings no better and would put the cache behind
# an API key.
MODEL = "Helsinki-NLP/opus-mt-de-en"

# The exact commit, not a branch: `--force` on another machine has to reproduce
# these strings rather than whatever `main` holds that day.
MODEL_REVISION = "1a922f3b32a8e809e17a47d4b32142d8105924e5"

SCHEMA = 1

# Label strings are short — 15.4 characters on average, 139 at the longest — so
# the batch is large and the generation is greedy. Beam search on a two-word
# noun phrase changes almost nothing and costs four times the wall clock.
BATCH_SIZE = 256
MAX_NEW_TOKENS = 64


def translate(strings: list[str], device: str) -> dict[str, str]:
    """German to English for each string, in one pass over a batched model.

    Sorted by length first, so a batch holds strings of similar size and the
    padding does not dominate: unsorted, one 139-character heading pads 255
    two-word ones to its length.
    """
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    model = MarianMTModel.from_pretrained(MODEL, revision=MODEL_REVISION)
    model = model.to(device).eval()

    ordered = sorted(strings, key=lambda string: (len(string), string))
    translations: dict[str, str] = {}
    started = time.perf_counter()

    for start in range(0, len(ordered), BATCH_SIZE):
        batch = ordered[start : start + BATCH_SIZE]
        with torch.no_grad():
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            generated = model.generate(
                **encoded, max_new_tokens=MAX_NEW_TOKENS, num_beams=1
            )
        for german, english in zip(
            batch, tokenizer.batch_decode(generated, skip_special_tokens=True)
        ):
            translations[german] = english.strip()

        done = start + len(batch)
        elapsed = time.perf_counter() - started
        rate = done / max(elapsed, 1e-9)
        print(
            f"  {done:>7,}/{len(ordered):,}  {elapsed:>6.0f}s  {rate:>5.0f}/s  "
            f"eta {(len(ordered) - done) / max(rate, 1e-9):>5.0f}s",
            flush=True,
        )
    return translations


def read_cache(path: Path) -> dict[str, str]:
    """The cached translations, or an empty map when there is no cache yet.

    The loader the pipeline uses, with its refusal turned into the empty map
    this script starts from: an absent cache is the normal first run here and
    the reported failure there.
    """
    try:
        return dict(load_label_translations(path))
    except MissingTranslations:
        return {}


def write_cache(path: Path, translations: dict[str, str]) -> None:
    """Write the cache, keys sorted, so regenerating it is byte-identical.

    Nothing in the document varies with when or where it ran — no timestamp, no
    device — because `--force` producing a diff of one line would make a real
    change of translations indistinguishable from a re-run at a glance. The date
    is in the commit; the model is what matters and it is pinned above.

    `produced_by` is provenance and nothing reads it: the pipeline asks for
    `translations` alone, which is what keeps the artifact independent of the
    model that produced it.
    """
    document = {
        "schema": SCHEMA,
        "produced_by": {
            "model": MODEL,
            "revision": MODEL_REVISION,
            "vocabulary": VOCABULARY,
            "source": "preferred names, their qualifiers, and classification names",
        },
        "strings": len(translations),
        "translations": dict(sorted(translations.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=1, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-translate every string, replacing the committed cache",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="say what the cache covers and what is missing; write nothing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="translate only the first N missing strings; leaves the cache "
        "incomplete, which tests/test_label_translations.py fails on",
    )
    parser.add_argument("--device", default="auto", help="mps, cuda, cpu or auto")
    args = parser.parse_args(argv)

    try:
        vocabulary = load_vocabulary(VOCABULARY)
        name_qualifiers = load_name_qualifiers(VOCABULARY)
    except MissingDataset as error:
        print(error)
        return 1

    required = translatable(vocabulary.values(), name_qualifiers)
    cached = {} if args.force else read_cache(LABEL_TRANSLATIONS_FILE)
    missing = sorted(required - set(cached))

    print(f"vocabulary:  {len(vocabulary):,} {VOCABULARY} entries")
    print(f"required:    {len(required):,} distinct German strings")
    print(f"cached:      {len(cached):,}")
    print(f"missing:     {len(missing):,}")

    if args.report:
        return 0
    if not missing:
        print("\nnothing to translate; the cache is up to date.")
        return 0
    if args.limit:
        missing = missing[: args.limit]
        print(
            f"translating: {len(missing):,} (--limit) — the cache will stay "
            "incomplete; run again without --limit before committing it"
        )

    from llms4subjects.hardware import describe_device, select_device

    device = select_device(args.device)
    print(describe_device(device))
    print(f"\ntranslating with {MODEL}:")

    cached.update(translate(missing, device))

    # Anything the vocabulary no longer names is dropped rather than carried
    # along: a cache holding strings the renderer cannot ask for would make the
    # coverage figure above unreadable after a vocabulary rebuild.
    kept = {german: cached[german] for german in sorted(required & set(cached))}
    write_cache(LABEL_TRANSLATIONS_FILE, kept)
    print(f"\nwrote {LABEL_TRANSLATIONS_FILE} ({len(kept):,} strings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
