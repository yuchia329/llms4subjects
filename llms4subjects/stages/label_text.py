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


def render_term(term: str, mode: str = DEFAULT_MODE, qualifier: str | None = None) -> str:
    """Render one vocabulary term under the chosen qualifier mode.

    ``qualifier`` supplies a boundary the string itself no longer carries — the preferred
    name, which the release has already flattened from ``Muster,Struktur`` to
    ``Muster Struktur``. It applies only when the term really ends with it, so a stale or
    mismatched sidecar entry cannot truncate a name.
    """
    _check_mode(mode)
    separator = ","
    if qualifier is not None and term.endswith(f" {qualifier}"):
        term, found, separator = term[: -len(qualifier) - 1], qualifier, " "
    else:
        term, found = split_qualifier(term)

    if found is None:
        return term
    if mode == "raw":
        return f"{term}{separator}{found}"
    if mode == "stripped":
        return term
    return f"{term} ({found})"


def label_text(
    entry: VocabularyEntry,
    qualifier_mode: str = DEFAULT_MODE,
    name_qualifiers: Mapping[str, str] | None = None,
    include_definition: bool = False,
) -> str:
    """Render one vocabulary entry as field-marked text.

        Fachgebiet: Organische Chemie
        Schlagwort: Polymerisation
        Synonyme: Polyreaktion; Kettenpolymerisation

    ``Definition`` is excluded by default: it holds cataloguing instructions to librarians
    rather than descriptions of meaning, and is present on a minority of the vocabulary.
    ``Related Subjects`` name neighbouring headings rather than this one, so they are not
    part of the text at all.
    """
    _check_mode(qualifier_mode)
    qualifier = (name_qualifiers or {}).get(entry.code)

    lines = []
    if entry.classification_name:
        lines.append(f"Fachgebiet: {entry.classification_name}")
    if entry.name:
        lines.append(f"Schlagwort: {render_term(entry.name, qualifier_mode, qualifier)}")
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
) -> list[LabelText]:
    """Render a vocabulary under one set of flags, in the order given."""
    return [
        LabelText(
            code=entry.code,
            text=label_text(entry, qualifier_mode, name_qualifiers, include_definition),
        )
        for entry in entries
    ]


def label_variants(
    entry: VocabularyEntry,
    qualifier_mode: str = DEFAULT_MODE,
    name_qualifiers: Mapping[str, str] | None = None,
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
    """
    _check_mode(qualifier_mode)
    qualifier = (name_qualifiers or {}).get(entry.code)

    strings = [render_term(entry.name, qualifier_mode, qualifier)] if entry.name else []
    strings += [render_term(name, qualifier_mode) for name in entry.alternate_names]
    return tuple(dict.fromkeys(string for string in strings if string))


def variants(
    entries: Iterable[VocabularyEntry],
    qualifier_mode: str = DEFAULT_MODE,
    name_qualifiers: Mapping[str, str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Every entry's surface strings, keyed by code, in the order given."""
    return {
        entry.code: label_variants(entry, qualifier_mode, name_qualifiers)
        for entry in entries
    }


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
