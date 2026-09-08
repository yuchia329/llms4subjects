"""The frozen translation cache: committed, complete, and model-independent.

The pipeline renders labels bilingually by looking strings up, never by
translating them, so these tests are about the artifact rather than about
translation quality. What they hold is that the cache covers the vocabulary the
renderer will ask about, that reading it costs nothing a run cannot afford, and
that nothing downstream can come to depend on which model wrote it.
"""

import ast
import json
from pathlib import Path

import pytest

from llms4subjects.corpus import (
    MissingDataset,
    MissingTranslations,
    load_label_translations,
    load_name_qualifiers,
    load_vocabulary,
)
from llms4subjects.paths import LABEL_TRANSLATIONS_FILE
from llms4subjects.stages.label_text import label_text, translatable

PACKAGE = Path(__file__).resolve().parent.parent / "llms4subjects"

# Everything the renderer reads. `produced_by` is provenance for a human, and
# a test that let the pipeline read it would make the cache model-dependent.
READ_BY_THE_PIPELINE = "translations"


@pytest.fixture(scope="module")
def document():
    return json.loads(LABEL_TRANSLATIONS_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def translations():
    return load_label_translations()


@pytest.fixture(scope="module")
def vocabulary():
    try:
        return load_vocabulary("tib-core")
    except MissingDataset as error:
        pytest.skip(str(error))


def test_the_cache_is_committed():
    """`reference/` is tracked on purpose; a run must never have to write it."""
    assert LABEL_TRANSLATIONS_FILE.is_file()


def test_the_loader_returns_the_mapping_alone(document, translations):
    assert translations == document[READ_BY_THE_PIPELINE]
    assert "produced_by" not in translations


def test_the_file_records_which_model_produced_it(document):
    """Provenance the artifact carries and the pipeline never reads."""
    assert document["produced_by"]["model"]
    assert document["schema"] == 1


def test_a_missing_cache_says_how_to_write_it(tmp_path):
    with pytest.raises(MissingTranslations, match="scripts/translate_labels.py"):
        load_label_translations(tmp_path / "absent.json")


def test_every_string_the_renderer_asks_for_is_cached(vocabulary, translations):
    """The acceptance criterion: no preferred name renders half-translated."""
    required = translatable(vocabulary.values(), load_name_qualifiers("tib-core"))
    assert not required - set(translations)


def test_the_cache_holds_nothing_the_renderer_cannot_ask_for(vocabulary, translations):
    """Otherwise the coverage figure above says less than it looks like it does."""
    required = translatable(vocabulary.values(), load_name_qualifiers("tib-core"))
    assert not set(translations) - required


def test_every_preferred_name_reaches_the_rendered_text(vocabulary, translations):
    """Coverage of the cache is not coverage of the rendering.

    A name whose English is identical to its German renders once rather than
    twice, which is correct and is why this counts entries that gained a second
    string rather than entries that were looked up.
    """
    qualifiers = load_name_qualifiers("tib-core")
    entries = list(vocabulary.values())
    bilingual = sum(
        1
        for entry in entries
        if label_text(entry, "parenthetical", qualifiers, False, translations)
        != label_text(entry, "parenthetical", qualifiers, False)
    )
    # Roughly three quarters; the rest are proper nouns, formulae and loanwords
    # the translator returns unchanged. A collapse here means the cache stopped
    # matching the strings the renderer splits out.
    assert bilingual > 0.6 * len(entries)


# The translating model reaches exactly one file, and it is not importable
# from the pipeline: `scripts/` is not on the package path.
TRANSLATOR_IMPORTS = ("transformers", "sentencepiece", "MarianMTModel")


@pytest.mark.parametrize("module", ["stages/label_text", "corpus", "pipeline"])
def test_the_rendering_path_never_imports_a_translator(module):
    """"Translation adds no per-run cost" is a claim about imports, not speed.

    Bilingual rendering is a dictionary hit. If a translation model could be
    reached from the rendering path, an indexing run could come to pay for one
    — and the "no per-run cost" line in docs/artifacts.md would be a hope.
    """
    source = (PACKAGE / f"{module}.py").read_text()
    tree = ast.parse(source)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            imported.update(alias.name for alias in node.names)

    assert not imported & set(TRANSLATOR_IMPORTS), module


class TestRegeneration:
    """"Regenerating it is idempotent" — the ticket-08 criterion, as a test.

    `scripts/` is not importable as a package, so the script is loaded by path.
    Its write is the thing under test: everything else about a re-run depends on
    a translation model these tests must not download.
    """

    @pytest.fixture(scope="class")
    def script(self):
        import importlib.util

        path = PACKAGE.parent / "scripts" / "translate_labels.py"
        spec = importlib.util.spec_from_file_location("translate_labels", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_rewriting_the_committed_cache_changes_no_byte(self, script, tmp_path):
        """Nothing in the document varies with when or where it ran."""
        before = LABEL_TRANSLATIONS_FILE.read_bytes()
        copy = tmp_path / "label_translations.json"
        script.write_cache(copy, script.read_cache(LABEL_TRANSLATIONS_FILE))
        assert copy.read_bytes() == before

    def test_the_write_order_does_not_follow_the_input_order(self, script, tmp_path):
        """Keys sorted, so a re-run that finds strings in another order agrees."""
        path = tmp_path / "cache.json"
        script.write_cache(path, {"Zug": "Train", "Apfel": "Apple"})
        other = tmp_path / "other.json"
        script.write_cache(other, {"Apfel": "Apple", "Zug": "Train"})
        assert path.read_bytes() == other.read_bytes()

    def test_reading_an_absent_cache_starts_from_empty(self, script, tmp_path):
        """A first run has no cache; only the pipeline treats that as an error."""
        assert script.read_cache(tmp_path / "absent.json") == {}

    def test_the_model_is_pinned_to_a_commit_not_a_branch(self, script):
        """A branch would make `--force` reproduce whatever it holds that day."""
        assert len(script.MODEL_REVISION) == 40
        assert script.MODEL_REVISION.isalnum()

    def test_the_committed_cache_names_the_model_that_is_pinned(
        self, script, document
    ):
        produced = document["produced_by"]
        assert produced["model"] == script.MODEL
        assert produced["revision"] == script.MODEL_REVISION
