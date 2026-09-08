"""Vocabulary entry to field-marked label text.

GND disambiguates homographs with a qualifier. The shared-task release writes it into
synonyms and related terms as a comma with no following space —
``Interaktion,Naturwissenschaft`` — and into preferred names as a bare suffix,
``Muster Struktur``, whose boundary survives only in the release's ``*_dnb-skos.ttl``
files as ``Muster (Struktur)``. `build_tibkat_csv.py` recovers those boundaries into a
sidecar map, which arrives here as ``name_qualifiers``.

How a qualified term becomes text is a flag with three modes:

    parenthetical   Muster (Struktur)   default; reads as natural language to an encoder
    raw             Muster Struktur     the release string, untouched
    stripped        Muster              the qualifier discarded, merging distinct senses

``stripped`` reproduces the lossy vocabulary this project used before, so the cost of
losing the sense distinction stays measurable rather than asserted.

All of that text is German, and 58.7% of gold label assignments belong to English
documents, so a second flag renders labels bilingually:

    Fachgebiet: Organische Chemie / Organic chemistry
    Schlagwort: Polymerisation / Polymerization
    Synonyme: Polyreaktion; Kettenpolymerisation

The English comes from ``reference/label_translations.json``, a frozen
German-string to English-string map written once by
``scripts/translate_labels.py``. This module never translates anything: it looks
strings up, so bilingual rendering costs a dictionary hit and the pipeline never
loads a translation model. Preferred names, their qualifiers and the 66
classification names are translated; the 71,570 synonyms are not, because they
are alternate German surface forms of a name whose English is already on the
line above.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from ..contracts import LabelText, VocabularyEntry

QUALIFIER_MODES = ("parenthetical", "raw", "stripped")

DEFAULT_MODE = "parenthetical"

# The fields a qualifier can appear in, under their names in the release schema.
QUALIFIED_FIELDS = ("Name", "Alternate Name", "Related Subjects")


def _check_mode(mode: str) -> None:
    if mode not in QUALIFIER_MODES:
        raise ValueError(f"unknown qualifier mode {mode!r}; expected one of {QUALIFIER_MODES}")


def split_qualifier(term: str) -> tuple[str, str | None]:
    """Split ``Term,Qualifier`` at the first comma that is not followed by a space.

    A comma *with* a space is ordinary punctuation inside a heading — "Fachkraft im
    Geld-, Wert- und Sicherheitstransport" — never a qualifier boundary.
    """
    for index, char in enumerate(term):
        if char == "," and index + 1 < len(term) and not term[index + 1].isspace():
            return term[:index], term[index + 1 :]
    return term, None


def _split_term(term: str, qualifier: str | None) -> tuple[str, str | None, str]:
    """Base, qualifier and the separator ``raw`` mode has to put back.

    ``qualifier`` supplies a boundary the string itself no longer carries — the preferred
    name, which the release has already flattened from ``Muster,Struktur`` to
    ``Muster Struktur``. It applies only when the term really ends with it, so a stale or
    mismatched sidecar entry cannot truncate a name.
    """
    if qualifier is not None and term.endswith(f" {qualifier}"):
        return term[: -len(qualifier) - 1], qualifier, " "
    base, found = split_qualifier(term)
    return base, found, ","


def _join_term(base: str, found: str | None, separator: str, mode: str) -> str:
    """Put a base and its qualifier back together under one qualifier mode.

    `separator` is what `raw` has to restore, and it differs by where the
    boundary came from: a comma when the string carried it, a space when the
    sidecar supplied one the release had already flattened away.
    """
    if found is None:
        return base
    if mode == "raw":
        return f"{base}{separator}{found}"
    if mode == "stripped":
        return base
    return f"{base} ({found})"


def render_term(term: str, mode: str = DEFAULT_MODE, qualifier: str | None = None) -> str:
    """Render one vocabulary term under the chosen qualifier mode."""
    _check_mode(mode)
    return _join_term(*_split_term(term, qualifier), mode)


def translate_term(
    term: str,
    translations: Mapping[str, str],
    mode: str = DEFAULT_MODE,
    qualifier: str | None = None,
) -> str | None:
    """The English rendering of one term, or ``None`` when it is not in the cache.

    The cache holds bare German strings, so a qualified name is looked up in two
    pieces: ``Muster (Struktur)`` is never a key, ``Muster`` and ``Struktur`` are.
    That is what lets the two senses of ``Interaktion`` share one translation of
    the word they have in common while keeping the qualifier that tells them
    apart. A qualifier missing from the cache stays German rather than being
    dropped, because dropping it would merge those senses in the English half of
    the line after the German half took care to separate them.

    The qualifier mode reaches the English rendering unchanged, separator
    included, so the two halves of a bilingual line are the same shape: ``raw``
    gives ``Muster Struktur / Pattern Structure`` and ``stripped`` gives
    ``Muster / Pattern``. Marking a boundary in English that the mode has just
    unmarked in German would make the flag an ablation of two things.

    ``None`` rather than the German string, so the caller can tell "no English"
    from "English that happens to match", and render one string instead of two.
    """
    _check_mode(mode)
    base, found, separator = _split_term(term, qualifier)
    english = translations.get(base)
    if not english:
        return None
    return _join_term(
        english, translations.get(found, found) if found else None, separator, mode
    )


def bilingual_text(german: str, english: str | None) -> str:
    """``German / English``, or the German alone when there is nothing to add.

    A translation identical to its source adds a repetition rather than a second
    language — thousands of GND names are proper nouns, chemical formulae or
    loanwords the translator returns intact — so it is dropped, which also keeps
    those labels' vectors identical to their German-only ones.
    """
    return german if not english or english == german else f"{german} / {english}"


def label_text(
    entry: VocabularyEntry,
    qualifier_mode: str = DEFAULT_MODE,
    name_qualifiers: Mapping[str, str] | None = None,
    include_definition: bool = False,
    translations: Mapping[str, str] | None = None,
) -> str:
    """Render one vocabulary entry as field-marked text.

        Fachgebiet: Organische Chemie / Organic chemistry
        Schlagwort: Polymerisation / Polymerization
        Synonyme: Polyreaktion; Kettenpolymerisation

    ``Definition`` is excluded by default: it holds cataloguing instructions to librarians
    rather than descriptions of meaning, and is present on a minority of the vocabulary.
    ``Related Subjects`` name neighbouring headings rather than this one, so they are not
    part of the text at all.

    An empty or absent ``translations`` renders German-only, which is the ablation
    ``label_text.bilingual: false`` selects. It is the same code path either way,
    so the two halves of that comparison differ in the text and in nothing else.
    """
    _check_mode(qualifier_mode)
    qualifier = (name_qualifiers or {}).get(entry.code)
    translations = translations or {}

    lines = []
    if entry.classification_name:
        english = translations.get(entry.classification_name)
        lines.append(
            f"Fachgebiet: {bilingual_text(entry.classification_name, english)}"
        )
    if entry.name:
        german = render_term(entry.name, qualifier_mode, qualifier)
        english = translate_term(entry.name, translations, qualifier_mode, qualifier)
        lines.append(f"Schlagwort: {bilingual_text(german, english)}")
    synonyms = [render_term(name, qualifier_mode) for name in entry.alternate_names]
    if synonyms:
        lines.append("Synonyme: " + "; ".join(synonyms))
    if include_definition and entry.definition:
        lines.append(f"Definition: {entry.definition}")
    return "\n".join(lines)


def render(
    entries: Iterable[VocabularyEntry],
    qualifier_mode: str = DEFAULT_MODE,
    name_qualifiers: Mapping[str, str] | None = None,
    include_definition: bool = False,
    translations: Mapping[str, str] | None = None,
) -> list[LabelText]:
    """Render a vocabulary under one set of flags, in the order given."""
    return [
        LabelText(
            code=entry.code,
            text=label_text(
                entry, qualifier_mode, name_qualifiers, include_definition, translations
            ),
        )
        for entry in entries
    ]


def label_variants(
    entry: VocabularyEntry,
    qualifier_mode: str = DEFAULT_MODE,
    name_qualifiers: Mapping[str, str] | None = None,
    translations: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """The label's own surface strings: its preferred name and its synonyms.

    Not the field-marked rendering. This is what a *lexical* match is against —
    the measurement the lexical retriever exists for is that a label's own name
    or one of its synonyms appears verbatim in the record text, for 53.7% of
    German gold assignments against 23.0% of English ones (docs/spec.md). The
    surrounding fields would only dilute it: ``Fachgebiet: Chemie`` is shared by
    thousands of labels and describes none of them, and ``Definition`` holds
    instructions to librarians.

    Each string is returned separately rather than joined, so a label with nine
    synonyms is not scored as one long document. Duplicates are dropped: a
    synonym that renders to the preferred name under ``stripped`` would
    otherwise count twice.

    The English name, where one is cached, is one more surface string rather
    than an addition to the German one: a lexical match is against a whole
    string, so ``Kraft / Force`` would match neither word. That is the half of
    the bilingual flag this retriever reads, and it is the half aimed at the
    23.0% verbatim rate on English documents against 53.7% on German ones.
    """
    _check_mode(qualifier_mode)
    qualifier = (name_qualifiers or {}).get(entry.code)
    translations = translations or {}

    strings = []
    if entry.name:
        strings.append(render_term(entry.name, qualifier_mode, qualifier))
        strings.append(
            translate_term(entry.name, translations, qualifier_mode, qualifier)
        )
    strings += [render_term(name, qualifier_mode) for name in entry.alternate_names]
    return tuple(dict.fromkeys(string for string in strings if string))


def variants(
    entries: Iterable[VocabularyEntry],
    qualifier_mode: str = DEFAULT_MODE,
    name_qualifiers: Mapping[str, str] | None = None,
    translations: Mapping[str, str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Every entry's surface strings, keyed by code, in the order given."""
    return {
        entry.code: label_variants(entry, qualifier_mode, name_qualifiers, translations)
        for entry in entries
    }


def translatable(
    entries: Iterable[VocabularyEntry],
    name_qualifiers: Mapping[str, str] | None = None,
) -> set[str]:
    """Every distinct German string a bilingual rendering has to look up.

    The inputs to `scripts/translate_labels.py`, computed here rather than there
    so that what the cache is asked to cover is decided by the code that renders
    it. Preferred names split into base and qualifier, plus the 66 classification
    names; synonyms and related subjects are not translated (see the module
    docstring).

    A set, so the two senses of ``Interaktion`` cost one translation of the word
    they share, and 79,427 entries reduce to 79,224 strings.
    """
    name_qualifiers = name_qualifiers or {}
    strings: set[str] = set()
    for entry in entries:
        if entry.classification_name:
            strings.add(entry.classification_name)
        if not entry.name:
            continue
        base, qualifier, _ = _split_term(entry.name, name_qualifiers.get(entry.code))
        strings.add(base)
        if qualifier:
            strings.add(qualifier)
    return strings


def count_qualifiers(
    entries: Sequence[VocabularyEntry] | Iterable[VocabularyEntry],
    name_qualifiers: Mapping[str, str] | None = None,
) -> dict:
    """Count how much of the vocabulary the qualifier mode actually moves.

    Recorded at rebuild time so the difference between the three modes is a measured
    number rather than an assumption.
    """
    name_qualifiers = name_qualifiers or {}
    by_field = {field: 0 for field in QUALIFIED_FIELDS}
    entries_affected = 0
    total = 0

    for entry in entries:
        total += 1
        affected = False
        if entry.code in name_qualifiers or split_qualifier(entry.name)[1] is not None:
            by_field["Name"] += 1
            affected = True
        for field, terms in (
            ("Alternate Name", entry.alternate_names),
            ("Related Subjects", entry.related_subjects),
        ):
            for term in terms:
                if split_qualifier(term)[1] is not None:
                    by_field[field] += 1
                    affected = True
        entries_affected += affected

    return {
        "entries": total,
        "entries_with_qualifier": entries_affected,
        "terms_with_qualifier": sum(by_field.values()),
        "by_field": by_field,
    }
