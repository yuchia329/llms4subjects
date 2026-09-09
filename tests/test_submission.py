"""The submission writer, tested only for round-trip through the official reader.

The layout is the contract, so the assertion that matters is that the
organizers' own reader gets back exactly the lists that went in. Anything else
would be testing our opinion of their format against itself.
"""

import json

import official_scorer as scorer
import pytest

from llms4subjects.contracts import CODES_PER_RECORD, Prediction, Record
from llms4subjects.stages.submission import (
    SUBJECT_FIELD,
    write_gold_tree,
    write_submission,
)


def codes(count=CODES_PER_RECORD, prefix="gnd:"):
    return tuple(f"{prefix}{n:07d}-0" for n in range(count))


@pytest.fixture(scope="module")
def entries():
    return scorer.fixture()["records"]


@pytest.fixture
def written(entries, tmp_path):
    predictions = [
        Prediction(entry["id"], tuple(entry["predictions"])) for entry in entries
    ]
    paths = write_submission(predictions, scorer.cells(entries), tmp_path / "run")
    return predictions, paths, tmp_path / "run"


def test_one_file_per_record_in_the_organizers_layout(entries, written):
    predictions, paths, root = written

    assert len(paths) == len(predictions)
    for entry in entries:
        expected = root / entry["type"] / entry["lang"] / f"{entry['id']}.json"
        assert expected in paths
        assert expected.exists()


def test_each_file_holds_exactly_fifty_ranked_codes(written):
    predictions, paths, _ = written

    for prediction, path in zip(predictions, paths):
        payload = json.loads(path.read_text())
        assert list(payload) == [SUBJECT_FIELD]
        assert payload[SUBJECT_FIELD] == list(prediction.codes)
        assert len(payload[SUBJECT_FIELD]) == CODES_PER_RECORD


def test_the_official_reader_round_trips_the_input_lists(written):
    predictions, _, root = written
    module = scorer.official()

    read_back = module.read_gnd_files(str(root), False)

    recovered = {
        filename.removesuffix(".json"): value
        for languages in read_back.values()
        for files in languages.values()
        for filename, value in files.items()
    }
    assert recovered == {
        prediction.record_id: list(prediction.codes) for prediction in predictions
    }


def test_the_official_validator_accepts_the_tree_against_its_own_gold(entries, written):
    _, _, root = written
    module = scorer.official()

    gold_tree = scorer.write_gold_tree(entries, root.parent / "gold")
    true_dict = module.read_gnd_files(str(gold_tree), True)
    pred_dict = module.read_gnd_files(str(root), False)

    assert module.validate_directory_structure(true_dict, pred_dict)


def test_a_ranking_that_is_not_fifty_codes_long_is_refused(tmp_path):
    short = [Prediction("r", codes(49))]

    with pytest.raises(ValueError, match="50"):
        write_submission(short, {"r": ("Book", "en")}, tmp_path)


def test_a_ranking_with_a_repeated_code_is_refused(tmp_path):
    repeated = codes(CODES_PER_RECORD - 1) + ("gnd:0000000-0",)
    assert len(set(repeated)) < CODES_PER_RECORD

    with pytest.raises(ValueError, match="duplicate"):
        write_submission([Prediction("r", repeated)], {"r": ("Book", "en")}, tmp_path)


def test_a_record_with_no_cell_is_refused_rather_than_written_somewhere(tmp_path):
    with pytest.raises(KeyError):
        write_submission([Prediction("r", codes())], {}, tmp_path)


def test_a_blank_record_type_or_language_is_refused(tmp_path):
    """A blank cell would collapse two path segments into one.

    The dev split has four records whose language field is empty, so this is a
    real row in the data rather than a defensive flourish.
    """
    with pytest.raises(ValueError, match="cell"):
        write_submission([Prediction("r", codes())], {"r": ("Book", "")}, tmp_path)


def test_the_same_record_twice_is_refused_rather_than_overwritten(tmp_path):
    twice = [Prediction("r", codes()), Prediction("r", codes(prefix="gnd:x"))]

    with pytest.raises(ValueError, match="duplicate record"):
        write_submission(twice, {"r": ("Book", "en")}, tmp_path)


def test_writing_a_language_the_official_reader_cannot_read_is_still_faithful(tmp_path):
    """A recorded limitation of the official script, not of this writer.

    Their reader seeds its result with `de` and `en` only and then indexes into
    it by directory name, so a `fr` record raises a `KeyError` inside their
    code. The test set does contain such records, so the writer emits the true
    cell and the limitation is documented here rather than papered over by
    quietly relabelling French records as English.
    """
    module = scorer.official()
    root = tmp_path / "run"

    write_submission([Prediction("r", codes())], {"r": ("Book", "fr")}, root)

    assert (root / "Book" / "fr" / "r.json").exists()
    with pytest.raises(KeyError, match="fr"):
        module.read_gnd_files(str(root), False)


# --- The gold side of the same layout (ticket 17) -----------------------------


def gold_record(id="r", subjects=("gnd:1", "gnd:2"), type="Book", lang="de"):
    return Record(
        id=id, type=type, lang=lang, title="t", abstract="a",
        subjects=tuple(subjects),
    )


def test_the_official_reader_recovers_the_gold_lists_that_went_in(tmp_path):
    """The same round-trip assertion as the prediction side, from the other end."""
    module = scorer.official()
    records = [
        gold_record("r1", ("gnd:1", "gnd:2")),
        gold_record("r2", ("gnd:3",), type="Thesis", lang="en"),
    ]

    write_gold_tree(records, tmp_path / "gold")

    read = module.read_gnd_files(str(tmp_path / "gold"), True)
    assert read["Book"]["de"]["r1.jsonld"] == ["gnd:1", "gnd:2"]
    assert read["Thesis"]["en"]["r2.jsonld"] == ["gnd:3"]


def test_the_gold_tree_pairs_with_a_submission_tree_by_basename(tmp_path):
    """Their validator pairs `<id>.jsonld` with `<id>.json`; that is the contract."""
    module = scorer.official()
    records = [gold_record("r1"), gold_record("r2", type="Thesis", lang="en")]

    write_gold_tree(records, tmp_path / "gold")
    write_submission(
        [Prediction(r.id, codes()) for r in records],
        {r.id: (r.type, r.lang) for r in records},
        tmp_path / "run",
    )

    assert module.validate_directory_structure(
        module.read_gnd_files(str(tmp_path / "gold"), True),
        module.read_gnd_files(str(tmp_path / "run"), False),
    )


def test_a_record_with_no_gold_is_written_and_dropped_by_their_reader(tmp_path):
    """`evaluate` excludes it for the same reason: recall has no denominator."""
    module = scorer.official()

    write_gold_tree([gold_record("r1", ())], tmp_path / "gold")

    assert (tmp_path / "gold" / "Book" / "de" / "r1.jsonld").exists()
    assert module.read_gnd_files(str(tmp_path / "gold"), True) == {}


def test_a_blank_cell_is_refused_rather_than_written_somewhere(tmp_path):
    with pytest.raises(ValueError, match="blank cell"):
        write_gold_tree([gold_record(lang="")], tmp_path / "gold")


def test_the_same_gold_record_twice_is_refused_rather_than_overwritten(tmp_path):
    with pytest.raises(ValueError, match="duplicate record"):
        write_gold_tree([gold_record("r"), gold_record("r", ("gnd:9",))], tmp_path)
