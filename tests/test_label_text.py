"""Behaviour of the label-text builder: qualifier rendering and field marking."""

import pytest

from llms4subjects.contracts import VocabularyEntry
from llms4subjects.stages.label_text import (
    QUALIFIER_MODES,
    count_qualifiers,
    label_text,
    render,
    render_term,
    split_qualifier,
)

ENTRY = VocabularyEntry(
    code="gnd:4768168-8",
    name="Muster Struktur",
    classification_name="Allgemeine Mathematik",
    classification_number="31.1",
    alternate_names=("Pattern", "Interaktion,Naturwissenschaft"),
    related_subjects=("Amplifikation,Psychologie",),
    definition="Verknüpfe mit Anwendungsgebiet",
)

PLAIN = VocabularyEntry(code="gnd:4003694-7", name="Kraft", classification_name="Physik")

NAME_QUALIFIERS = {"gnd:4768168-8": "Struktur"}


class TestSplitQualifier:
    def test_comma_without_space_separates_term_from_qualifier(self):
        assert split_qualifier("Interaktion,Naturwissenschaft") == (
            "Interaktion",
            "Naturwissenschaft",
        )

    def test_comma_followed_by_space_is_part_of_the_term(self):
        term = "Fachkraft im Geld-, Wert- und Sicherheitstransport"
        assert split_qualifier(term) == (term, None)

    def test_only_the_first_comma_separates(self):
        assert split_qualifier("Fliese,Verlegung,Technik") == ("Fliese", "Verlegung,Technik")

    def test_unqualified_term_has_no_qualifier(self):
        assert split_qualifier("Polymerisation") == ("Polymerisation", None)


class TestRenderTerm:
    def test_default_mode_is_parenthetical(self):
        assert render_term("Interaktion,Naturwissenschaft") == "Interaktion (Naturwissenschaft)"

    def test_raw_mode_leaves_the_release_string_untouched(self):
        assert (
            render_term("Interaktion,Naturwissenschaft", mode="raw")
            == "Interaktion,Naturwissenschaft"
        )

    def test_stripped_mode_drops_the_qualifier(self):
        assert render_term("Interaktion,Naturwissenschaft", mode="stripped") == "Interaktion"

    def test_unqualified_term_is_identical_in_every_mode(self):
        assert {render_term("Polymerisation", mode=m) for m in QUALIFIER_MODES} == {
            "Polymerisation"
        }

    def test_rendering_is_idempotent(self):
        once = render_term("Interaktion,Naturwissenschaft")
        assert render_term(once) == once

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            render_term("Interaktion,Naturwissenschaft", mode="parenthesised")

    def test_explicit_qualifier_marks_a_boundary_the_string_does_not_carry(self):
        assert render_term("Muster Struktur", qualifier="Struktur") == "Muster (Struktur)"
        assert render_term("Muster Struktur", qualifier="Struktur", mode="raw") == "Muster Struktur"
        assert render_term("Muster Struktur", qualifier="Struktur", mode="stripped") == "Muster"

    def test_explicit_qualifier_is_ignored_when_the_term_does_not_end_with_it(self):
        assert render_term("Musterbeispiel", qualifier="Struktur") == "Musterbeispiel"


class TestLabelText:
    def test_default_rendering_is_field_marked_and_parenthetical(self):
        text = label_text(ENTRY, name_qualifiers=NAME_QUALIFIERS)
        assert text.splitlines() == [
            "Fachgebiet: Allgemeine Mathematik",
            "Schlagwort: Muster (Struktur)",
            "Synonyme: Pattern; Interaktion (Naturwissenschaft)",
        ]

    def test_raw_mode_keeps_the_release_strings(self):
        text = label_text(ENTRY, qualifier_mode="raw", name_qualifiers=NAME_QUALIFIERS)
        assert "Schlagwort: Muster Struktur" in text
        assert "Interaktion,Naturwissenschaft" in text

    def test_stripped_mode_reproduces_the_lossy_form(self):
        text = label_text(ENTRY, qualifier_mode="stripped", name_qualifiers=NAME_QUALIFIERS)
        assert "Schlagwort: Muster" in text
        assert "Struktur" not in text
        assert "Synonyme: Pattern; Interaktion" in text

    def test_name_qualifiers_are_optional(self):
        assert "Schlagwort: Muster Struktur" in label_text(ENTRY)

    def test_definition_is_excluded_by_default_and_available_as_a_flag(self):
        assert "Verknüpfe" not in label_text(ENTRY)
        assert "Definition: Verknüpfe mit Anwendungsgebiet" in label_text(
            ENTRY, include_definition=True
        )

    def test_missing_optional_fields_produce_no_empty_lines(self):
        assert label_text(PLAIN).splitlines() == ["Fachgebiet: Physik", "Schlagwort: Kraft"]

    def test_related_subjects_are_not_part_of_the_default_text(self):
        assert "Amplifikation" not in label_text(ENTRY)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            label_text(ENTRY, qualifier_mode="nope")


class TestRender:
    def test_returns_one_label_text_per_entry_keyed_by_code(self):
        rendered = render([ENTRY, PLAIN], name_qualifiers=NAME_QUALIFIERS)
        assert [item.code for item in rendered] == [ENTRY.code, PLAIN.code]
        assert rendered[0].text == label_text(ENTRY, name_qualifiers=NAME_QUALIFIERS)

    def test_flags_reach_every_entry(self):
        rendered = render([ENTRY], qualifier_mode="stripped", name_qualifiers=NAME_QUALIFIERS)
        assert "Struktur" not in rendered[0].text


class TestCountQualifiers:
    def test_counts_entries_and_terms_by_field(self):
        stats = count_qualifiers([ENTRY, PLAIN], name_qualifiers=NAME_QUALIFIERS)
        assert stats["entries"] == 2
        assert stats["entries_with_qualifier"] == 1
        assert stats["terms_with_qualifier"] == 3
        assert stats["by_field"] == {"Name": 1, "Alternate Name": 1, "Related Subjects": 1}

    def test_name_qualifiers_are_optional(self):
        stats = count_qualifiers([ENTRY])
        assert stats["by_field"]["Name"] == 0
        assert stats["terms_with_qualifier"] == 2

    def test_a_comma_qualified_name_is_counted_without_the_sidecar(self):
        entry = VocabularyEntry(code="gnd:1", name="Interaktion,Naturwissenschaft")
        assert count_qualifiers([entry])["by_field"]["Name"] == 1
