"""Recovery of preferred-name qualifiers from the release's SKOS files."""

import unicodedata

from build_tibkat_csv import (
    SUBSET_PREFIXES,
    clone_commands,
    name_qualifiers,
    parse_pref_labels,
)
from llms4subjects.stages.label_text import render_term

TTL = """@prefix gnd: <https://d-nb.info/gnd/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

gnd:4768168-8 a skos:Concept;
  skos:altLabel "Pattern"@de;
  skos:prefLabel "Muster (Struktur)"@de.

gnd:4003694-7 a skos:Concept;
  skos:prefLabel "Ausbreitung"@de.

gnd:4056138-0 a skos:Concept;
  skos:prefLabel "Sequenz (Biochemie)"@de.

gnd:4130333-4 a skos:Concept;
  skos:prefLabel "Zitat \\"Wort\\" (Sprache)"@de.
"""


class TestParsePrefLabels:
    def test_reads_one_preferred_label_per_concept(self):
        labels = parse_pref_labels(TTL)
        assert labels["gnd:4768168-8"] == "Muster (Struktur)"
        assert labels["gnd:4003694-7"] == "Ausbreitung"

    def test_alternate_labels_are_not_mistaken_for_preferred_ones(self):
        assert "Pattern" not in parse_pref_labels(TTL).values()

    def test_escaped_quotes_survive(self):
        assert parse_pref_labels(TTL)["gnd:4130333-4"] == 'Zitat "Wort" (Sprache)'


class TestNameQualifiers:
    def test_recovers_the_boundary_the_json_flattened(self):
        entries = [{"Code": "gnd:4768168-8", "Name": "Muster Struktur"}]
        assert name_qualifiers(entries, parse_pref_labels(TTL)) == {"gnd:4768168-8": "Struktur"}

    def test_unqualified_names_are_absent(self):
        entries = [{"Code": "gnd:4003694-7", "Name": "Ausbreitung"}]
        assert name_qualifiers(entries, parse_pref_labels(TTL)) == {}

    def test_a_name_that_does_not_reconstruct_is_left_alone(self):
        entries = [{"Code": "gnd:4056138-0", "Name": "Sequenz"}]
        assert name_qualifiers(entries, parse_pref_labels(TTL)) == {}

    def test_unicode_normalisation_does_not_block_a_match(self):
        entries = [{"Code": "gnd:1", "Name": "Exposé Marketing"}]
        labels = {"gnd:1": "Exposé (Marketing)"}
        assert name_qualifiers(entries, labels) == {"gnd:1": "Marketing"}

    def test_codes_missing_from_the_skos_file_are_skipped(self):
        entries = [{"Code": "gnd:9999", "Name": "Etwas Anderes"}]
        assert name_qualifiers(entries, parse_pref_labels(TTL)) == {}


class TestCloneCommands:
    def test_the_clone_is_sparse_shallow_and_blobless(self):
        clone, sparse = clone_commands(".cache/release")
        assert clone[:2] == ["git", "clone"]
        assert {"--filter=blob:none", "--sparse", "--depth"} <= set(clone)
        assert clone[-1] == ".cache/release"

    def test_the_sparse_step_can_be_re_run_on_its_own(self, tmp_path):
        """An interrupted clone is finished by re-running, not by deleting the directory."""
        repo = tmp_path / "release"
        (repo / ".git").mkdir(parents=True)
        clone, sparse = clone_commands(str(repo))
        assert sparse[:3] == ["git", "-C", str(repo)]

    def test_only_the_shared_task_datasets_are_checked_out(self):
        _, sparse = clone_commands(".cache/release")
        assert sparse == ["git", "-C", ".cache/release", "sparse-checkout", "set",
                          "shared-task-datasets"]


class TestSubsetPrefixes:
    def test_each_track_writes_its_own_csvs(self):
        assert SUBSET_PREFIXES == {"tib-core-subjects": "core", "all-subjects": "all"}
