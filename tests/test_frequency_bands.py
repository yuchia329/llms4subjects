"""The frozen bands: read from a committed artifact, never recomputed.

Every later ticket reports a band breakdown, so a band that quietly changes
meaning between rungs would make those tables incomparable while still looking
plausible. These tests are what stops that.
"""

from collections import Counter

import pytest

from llms4subjects.corpus import (
    MissingDataset,
    UNSEEN_BAND,
    band_for_count,
    frequency_band_reference,
    frequency_bands,
    label_counts,
    load_split,
)
from llms4subjects.paths import FREQUENCY_BANDS_FILE
from llms4subjects.stages import evaluator
from llms4subjects.stages.evaluator import BANDS, evaluate

# The boundaries as docs/spec.md states them, compared against the committed
# reference rather than against the script that wrote it: the artifact is what
# every later result reads, so the artifact is what has to say this.
DOCUMENTED = (
    {"band": "head", "min": 101, "max": None},
    {"band": "torso", "min": 10, "max": 100},
    {"band": "tail", "min": 1, "max": 9},
    {"band": "zero", "min": 0, "max": 0},
)

# docs/spec.md: 65 head labels against tib-core train.
DOCUMENTED_HEAD_LABELS = 65


@pytest.fixture(scope="module")
def reference():
    if not FREQUENCY_BANDS_FILE.exists():
        pytest.fail(
            f"{FREQUENCY_BANDS_FILE} is a committed artifact and must be in the tree; "
            "regenerate it with `python scripts/freeze_bands.py --force`"
        )
    return frequency_band_reference()


def dev_records():
    try:
        return load_split("core_dev")
    except MissingDataset as error:
        pytest.skip(str(error))


def all_subjects_records():
    """The rung-3 index corpus, which only exists after the second build."""
    try:
        return load_split("all_train")
    except MissingDataset as error:
        pytest.skip(str(error))


def test_the_two_declarations_of_the_unseen_band_agree():
    """`corpus` reads the reference; a stage may not import `corpus`.

    So the constant is stated in both places (see tests/test_stage_layout.py),
    and this is what keeps the two statements the same one.
    """
    assert evaluator.UNSEEN_BAND == UNSEEN_BAND
    assert UNSEEN_BAND in BANDS


def test_the_boundaries_are_the_documented_ones(reference):
    assert tuple(reference["boundaries"]) == DOCUMENTED


def test_the_reference_records_where_it_came_from(reference):
    assert reference["schema"] == 1
    assert reference["frozen_from"]["split"] == "core_train"
    assert reference["frozen_from"]["records"] > 0
    assert reference["frozen_from"]["distinct_labels"] == len(label_counts())


def test_the_head_band_holds_the_documented_number_of_labels(reference):
    assert len(reference["labels"]["head"]) == DOCUMENTED_HEAD_LABELS


def test_the_zero_band_is_the_absence_of_an_entry_not_a_listed_one(reference):
    assert UNSEEN_BAND not in reference["labels"]
    assert frequency_bands().get("gnd:no-such-label", UNSEEN_BAND) == UNSEEN_BAND


def test_every_recorded_count_sits_inside_its_bands_range(reference):
    for boundary in reference["boundaries"]:
        band = boundary["band"]
        if band == UNSEEN_BAND:
            continue
        for code, count in reference["labels"][band].items():
            assert count >= boundary["min"], code
            if boundary["max"] is not None:
                assert count <= boundary["max"], code


def test_each_label_is_named_by_exactly_one_band(reference):
    named = [code for labels in reference["labels"].values() for code in labels]

    assert len(named) == len(set(named))
    assert len(named) == len(frequency_bands())


def test_growing_the_index_would_move_labels_but_the_frozen_bands_do_not():
    """The property the freeze exists for, stated as a comparison.

    Counting over a larger corpus genuinely reclassifies labels — that is not
    hypothetical, and the assertion below proves it on this data. The frozen
    mapping is unaffected because it is read rather than derived.
    """
    frozen = frequency_bands()
    boundaries = frequency_band_reference()["boundaries"]

    grown = Counter()
    for record in dev_records():
        grown.update(record.subjects)

    def grown_band(code, frozen_count):
        return band_for_count(frozen_count + grown[code], boundaries)

    moved = {
        code: (frozen[code], grown_band(code, frozen_count))
        for code, frozen_count in label_counts().items()
        if code in grown and frozen[code] != grown_band(code, frozen_count)
    }
    assert moved, "a larger corpus that moves no label would make this test vacuous"

    assert frequency_bands() == frozen
    for code, (frozen_band, would_be) in moved.items():
        assert frequency_bands()[code] == frozen_band != would_be


def test_the_rung_3_corpus_does_not_touch_the_frozen_bands():
    """Ticket 15's own version of the property, over the corpus it indexes.

    The all-subjects split is 38,545 documents on top of tib-core train, and it
    is the largest thing this project ever counts labels over. Reporting its
    effect on the bands is `scripts/rung3_report.py`'s job and is a table of its
    own; what must not happen is the bands themselves moving, because every
    band figure at rungs 1 and 2 was measured before the corpus existed.
    """
    frozen = frequency_bands()
    boundaries = frequency_band_reference()["boundaries"]

    grown = Counter()
    for record in all_subjects_records():
        grown.update(dict.fromkeys(record.subjects).keys())

    reached = [
        code
        for code in grown
        if frozen.get(code, UNSEEN_BAND) == UNSEEN_BAND
        and band_for_count(grown[code], boundaries) != UNSEEN_BAND
    ]
    assert reached, "a corpus that reaches no unseen label would make this vacuous"

    # Read, not derived: every one of those labels is still zero-shot here.
    assert frequency_bands() == frozen
    assert all(frequency_bands().get(code, UNSEEN_BAND) == UNSEEN_BAND for code in reached)
    assert frequency_band_reference()["frozen_from"]["split"] == "core_train"


def test_the_evaluator_bands_a_label_by_the_frozen_reference_not_by_the_data():
    bands = frequency_bands()
    head = next(code for code, band in bands.items() if band == "head")
    tail = next(code for code, band in bands.items() if band == "tail")

    report = evaluate(
        {"r": [head, tail]},
        {"r": [head, tail]},
        bands,
        {"r": ("Book", "en")},
        ks=(5,),
    )

    assert report.by_band["head"].assignments == 1
    assert report.by_band["tail"].assignments == 1
    assert all(report.by_band[band].assignments == 0 for band in ("torso", "zero"))
    assert tuple(report.by_band) == BANDS
