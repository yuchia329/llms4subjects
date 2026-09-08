"""Behaviour of the label-text builder: qualifier rendering and field marking."""

import pytest

from llms4subjects.contracts import VocabularyEntry
from llms4subjects.stages.label_text import (
    QUALIFIER_MODES,
    bilingual_text,
    count_qualifiers,
    label_text,
    label_variants,
    render,
    render_term,
    split_qualifier,
    translatable,
    translate_term,
    variants,
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

# The translation cache is a German-string to English-string map, so a term and
# its qualifier are separate entries and homographs share one translation.
TRANSLATIONS = {
    "Muster": "Pattern",
    "Struktur": "Structure",
    "Allgemeine Mathematik": "General mathematics",
    "Kraft": "Force",
    "Physik": "Physics",
    # Deliberately absent: "Interaktion", "Naturwissenschaft", "Pattern".
}


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


class TestBilingualText:
    """Joining a German string to its English translation, or leaving it alone."""

    def test_a_translation_is_appended_after_a_slash(self):
        assert bilingual_text("Polymerisation", "Polymerization") == (
            "Polymerisation / Polymerization"
        )

    def test_an_untranslated_string_is_left_as_it_is(self):
        assert bilingual_text("Polymerisation", None) == "Polymerisation"

    def test_a_translation_equal_to_its_source_is_not_repeated(self):
        """Thousands of GND names are proper nouns the translator returns intact."""
        assert bilingual_text("Aluminium", "Aluminium") == "Aluminium"

    def test_an_empty_translation_is_treated_as_absent(self):
        assert bilingual_text("Polymerisation", "") == "Polymerisation"


class TestTranslateTerm:
    def test_an_unqualified_term_is_looked_up_whole(self):
        assert translate_term("Muster", TRANSLATIONS) == "Pattern"

    def test_term_and_qualifier_are_translated_separately(self):
        """The cache holds bare strings, so "Muster (Struktur)" is never a key."""
        assert (
            translate_term("Muster Struktur", TRANSLATIONS, qualifier="Struktur")
            == "Pattern (Structure)"
        )

    def test_the_qualifier_mode_reaches_the_english_rendering(self):
        """Separator included: both halves of a bilingual line are one shape."""
        assert (
            translate_term("Muster Struktur", TRANSLATIONS, "raw", "Struktur")
            == "Pattern Structure"
        )
        assert (
            translate_term("Interaktion,Naturwissenschaft", {"Interaktion": "I"}, "raw")
            == "I,Naturwissenschaft"
        )
        assert (
            translate_term("Muster Struktur", TRANSLATIONS, "stripped", "Struktur")
            == "Pattern"
        )

    def test_an_untranslated_term_has_no_english_rendering(self):
        assert translate_term("Interaktion,Naturwissenschaft", TRANSLATIONS) is None

    def test_an_untranslated_qualifier_keeps_the_german_sense(self):
        """A missing qualifier must not merge two senses into one English string."""
        translations = {"Interaktion": "Interaction"}
        assert (
            translate_term("Interaktion,Naturwissenschaft", translations)
            == "Interaction (Naturwissenschaft)"
        )

    def test_no_translations_means_no_english_rendering(self):
        assert translate_term("Muster", {}) is None

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            translate_term("Muster", TRANSLATIONS, mode="nope")


class TestBilingualLabelText:
    def test_name_and_classification_render_bilingually(self):
        text = label_text(
            ENTRY, name_qualifiers=NAME_QUALIFIERS, translations=TRANSLATIONS
        )
        assert text.splitlines() == [
            "Fachgebiet: Allgemeine Mathematik / General mathematics",
            "Schlagwort: Muster (Struktur) / Pattern (Structure)",
            "Synonyme: Pattern; Interaktion (Naturwissenschaft)",
        ]

    def test_synonyms_stay_german(self):
        """Only preferred names are translated; the 71,570 synonyms are not."""
        text = label_text(PLAIN, translations={"Kraft": "Force", "Physik": "Physics"})
        assert "Synonyme" not in text

    def test_without_translations_the_rendering_is_german_only(self):
        assert label_text(ENTRY, name_qualifiers=NAME_QUALIFIERS) == label_text(
            ENTRY, name_qualifiers=NAME_QUALIFIERS, translations={}
        )

    def test_an_untranslated_entry_renders_german_only(self):
        entry = VocabularyEntry(code="gnd:1", name="Nichtübersetzt")
        assert label_text(entry, translations=TRANSLATIONS) == "Schlagwort: Nichtübersetzt"

    def test_render_passes_translations_to_every_entry(self):
        rendered = render(
            [ENTRY, PLAIN], name_qualifiers=NAME_QUALIFIERS, translations=TRANSLATIONS
        )
        assert "Pattern (Structure)" in rendered[0].text
        assert "Kraft / Force" in rendered[1].text


class TestBilingualVariants:
    """What a lexical match is against, when the labels have English names."""

    def test_the_english_name_joins_the_surface_strings(self):
        """Directly after the German name, ahead of the synonyms."""
        assert label_variants(
            ENTRY, name_qualifiers=NAME_QUALIFIERS, translations=TRANSLATIONS
        ) == (
            "Muster (Struktur)",
            "Pattern (Structure)",
            "Pattern",
            "Interaktion (Naturwissenschaft)",
        )

    def test_the_english_name_is_a_string_of_its_own(self):
        """Not appended to the German one: a joined string matches neither."""
        strings = label_variants(PLAIN, translations=TRANSLATIONS)
        assert strings == ("Kraft", "Force")

    def test_an_english_name_equal_to_a_german_one_is_not_duplicated(self):
        entry = VocabularyEntry(code="gnd:1", name="Aluminium")
        assert label_variants(entry, translations={"Aluminium": "Aluminium"}) == (
            "Aluminium",
        )

    def test_an_english_name_equal_to_a_synonym_is_not_duplicated(self):
        """"Pattern" is already a synonym of "Muster Struktur" under stripped."""
        entry = VocabularyEntry(code="gnd:1", name="Muster", alternate_names=("Pattern",))
        assert label_variants(entry, translations=TRANSLATIONS) == ("Muster", "Pattern")

    def test_without_translations_the_strings_are_german_only(self):
        assert label_variants(ENTRY, name_qualifiers=NAME_QUALIFIERS) == (
            "Muster (Struktur)",
            "Pattern",
            "Interaktion (Naturwissenschaft)",
        )

    def test_variants_passes_translations_to_every_entry(self):
        built = variants([ENTRY, PLAIN], translations=TRANSLATIONS)
        assert built[PLAIN.code] == ("Kraft", "Force")


class TestTranslatable:
    """Which German strings a bilingual rendering needs, and no others."""

    def test_names_qualifiers_and_classifications_only(self):
        assert translatable([ENTRY, PLAIN], NAME_QUALIFIERS) == {
            "Muster",
            "Struktur",
            "Allgemeine Mathematik",
            "Kraft",
            "Physik",
        }

    def test_synonyms_and_related_subjects_are_not_translated(self):
        strings = translatable([ENTRY], NAME_QUALIFIERS)
        assert "Pattern" not in strings
        assert "Interaktion" not in strings
        assert "Amplifikation" not in strings

    def test_a_comma_qualified_name_is_split_without_the_sidecar(self):
        entry = VocabularyEntry(code="gnd:1", name="Interaktion,Soziologie")
        assert translatable([entry]) == {"Interaktion", "Soziologie"}

    def test_homographs_share_one_entry_for_the_word_they_share(self):
        entries = [
            VocabularyEntry(code="gnd:1", name="Interaktion,Soziologie"),
            VocabularyEntry(code="gnd:2", name="Interaktion,Naturwissenschaft"),
        ]
        assert translatable(entries) == {
            "Interaktion",
            "Soziologie",
            "Naturwissenschaft",
        }

    def test_an_entry_with_no_name_contributes_nothing(self):
        assert translatable([VocabularyEntry(code="gnd:1", name="")]) == set()
