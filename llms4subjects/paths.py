"""Where the project's committed inputs live.

Dataset files are not tracked in git; rebuild them with `python build_tibkat_csv.py`.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TIBKAT_DIR = REPO_ROOT / "TIBKAT_dataset"
GND_DIR = REPO_ROOT / "GND_dataset"
CONFIG_DIR = REPO_ROOT / "configs"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

# Committed reference artifacts: small, tracked, and frozen on purpose. Unlike
# `artifacts/`, nothing here is regenerated as a side effect of a run.
REFERENCE_DIR = REPO_ROOT / "reference"

# Label frequency bands, frozen against tib-core train counts so that growing
# the index later cannot reclassify which labels count as tail. Regenerated
# only by `python scripts/freeze_bands.py --force`; see docs/artifacts.md.
FREQUENCY_BANDS_FILE = REFERENCE_DIR / "frequency_bands.json"

# English translations of the vocabulary's German preferred names, their
# qualifiers and the 66 classification names, translated once and committed so
# that no run loads a translation model. Regenerated only by
# `python scripts/translate_labels.py`; see docs/artifacts.md.
LABEL_TRANSLATIONS_FILE = REFERENCE_DIR / "label_translations.json"

# Every model any part of this project loads, with its release date and the
# revision it is pinned to, so that "released on or before 2025-01-31" is a
# checked fact rather than a comment. Regenerated only by
# `python scripts/verify_model_releases.py --force`; see docs/artifacts.md.
MODEL_RELEASES_FILE = REFERENCE_DIR / "model_releases.json"

# Which corpora are clear to index, checked against the held-out splits and
# committed so that no run has to open the gold test set to prove it is not in
# the index. Regenerated only by `python scripts/verify_split_alignment.py`; see
# docs/artifacts.md.
SPLIT_ALIGNMENT_FILE = REFERENCE_DIR / "split_alignment.json"

# The configurations ticket 17's single test run will score, digested and
# committed *before* the gold test split is read, and the receipt written after
# it has been. Both are `scripts/final_test.py`'s; see llms4subjects/testset.py
# for why there are two of them and what each refuses.
TEST_PLAN_FILE = REFERENCE_DIR / "test_plan.json"
TEST_RUN_FILE = REFERENCE_DIR / "test_run.json"

SPLIT_FILES = {
    "core_train": TIBKAT_DIR / "core_train.csv",
    "core_dev": TIBKAT_DIR / "core_dev.csv",
    "core_test": TIBKAT_DIR / "core_test.csv",
    # The all-subjects train split, used as index and training data only; it is
    # written by `build_tibkat_csv.py --subset all-subjects` and contains every
    # core_train record plus 38,545 more.
    "all_train": TIBKAT_DIR / "all_train.csv",
}

VOCABULARY_FILES = {
    "tib-core": GND_DIR / "GND-Subjects-tib-core.json",
    "all": GND_DIR / "GND-Subjects-all.json",
}

# Preferred-name qualifier boundaries, recovered from the release's SKOS files.
NAME_QUALIFIER_FILES = {
    "tib-core": GND_DIR / "GND-Name-Qualifiers-tib-core.json",
    "all": GND_DIR / "GND-Name-Qualifiers-all.json",
}
