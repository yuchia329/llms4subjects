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
