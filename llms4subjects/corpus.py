"""The only module that reads the dataset from disk.

Stages receive `Record` and `VocabularyEntry` objects as arguments, so nothing
downstream re-reads a CSV, and no stage can accidentally depend on a file that
another stage happened to write.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import Code, Record, VocabularyEntry
from .paths import (
    FREQUENCY_BANDS_FILE,
    LABEL_TRANSLATIONS_FILE,
    NAME_QUALIFIER_FILES,
    SPLIT_FILES,
    VOCABULARY_FILES,
)

# Some abstracts are long enough to trip the default csv field limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


class MissingDataset(FileNotFoundError):
    """The dataset is gitignored; this says how to rebuild it."""


REBUILD_HINT = (
    "The dataset is not tracked in git. Rebuild it, clone included, with:\n"
    "  python build_tibkat_csv.py"
)


def load_split(split: str, path: str | Path | None = None) -> list[Record]:
    """Load one official split as records, in file order."""
    source = _resolve(path, SPLIT_FILES, split, "split")
    with _open(source) as handle:
        return [_record(row) for row in csv.DictReader(handle)]


def load_vocabulary(
    name: str = "tib-core", path: str | Path | None = None
) -> dict[str, VocabularyEntry]:
    """Load a GND vocabulary file, keyed by code."""
    source = _resolve(path, VOCABULARY_FILES, name, "vocabulary")
    with _open(source) as handle:
        raw = json.load(handle)
    entries = raw.values() if isinstance(raw, dict) else raw
    return {entry["Code"]: entry_from_release(entry) for entry in entries}


def load_name_qualifiers(
    name: str = "tib-core", path: str | Path | None = None
) -> dict[str, str]:
    """Load the preferred-name qualifier boundaries recovered at rebuild time.

    The release's JSON flattens "Muster,Struktur" to "Muster Struktur"; the boundary
    survives only in its SKOS files, so `build_tibkat_csv.py` writes it out separately.
    An absent file is not an error — it means the rendering falls back to whatever
    boundary the strings themselves carry.
    """
    source = _resolve(path, NAME_QUALIFIER_FILES, name, "vocabulary")
    if not source.exists():
        return {}
    with _open(source) as handle:
        return json.load(handle)


class MissingTranslations(FileNotFoundError):
    """`bilingual` is on and the translation cache is not there."""


def load_label_translations(path: str | Path | None = None) -> dict[str, str]:
    """The frozen German-string to English-string map, without its provenance.

    Callers get the mapping alone: which model produced it is recorded in the
    file and read by nobody, which is what "independent of the translating
    model" means in practice. An absent file raises rather than returning an
    empty map — an empty map renders German-only, which is the other half of the
    ablation, and a bilingual run that quietly produced the German-only numbers
    would be reported as "translation does not help".
    """
    source = Path(path) if path is not None else LABEL_TRANSLATIONS_FILE
    if not source.exists():
        raise MissingTranslations(
            f"{source} is missing, and `label_text.bilingual` needs it. Write it "
            "with:\n  python scripts/translate_labels.py"
        )
    with _open(source) as handle:
        return json.load(handle)["translations"]


# The band a label falls in when the frozen reference does not name it: it was
# never seen in tib-core train, which is what "zero-shot" means here.
UNSEEN_BAND = "zero"


def frequency_band_reference(path: str | Path | None = None) -> dict:
    """The frozen band reference, verbatim, including its provenance block."""
    source = Path(path) if path is not None else FREQUENCY_BANDS_FILE
    with _open(source) as handle:
        return json.load(handle)


def frequency_bands(path: str | Path | None = None) -> dict[Code, str]:
    """Every label the frozen reference names, mapped to its band.

    Labels absent from the mapping are `zero` by construction, so callers read
    it with `bands.get(code, UNSEEN_BAND)` rather than expecting a total map:
    enumerating the zero band would mean listing most of a 79,427-code
    vocabulary to say nothing about it.

    The assignment is read, never derived, so no amount of extra index data can
    move a label between bands. That is the whole point of freezing it.
    """
    reference = frequency_band_reference(path)
    return {
        code: band
        for band, labels in reference["labels"].items()
        for code in labels
    }


def label_counts(path: str | Path | None = None) -> dict[Code, int]:
    """Frozen tib-core train occurrence count per label, for band migration."""
    reference = frequency_band_reference(path)
    return {
        code: count
        for labels in reference["labels"].values()
        for code, count in labels.items()
    }


def band_for_count(count: int, boundaries: Iterable[Mapping[str, Any]]) -> str:
    """The band an occurrence count falls in, per the reference's own boundaries.

    Driven by the boundaries as recorded in the frozen artifact rather than by a
    constant, so the only statement of where the bands begin is the one that
    shipped with the assignment. Reporting band migration under a larger index
    means asking this what a grown count *would* have been; it is not how a
    label's band is looked up, which is `frequency_bands`.
    """
    for boundary in boundaries:
        low, high = boundary["min"], boundary["max"]
        if count >= low and (high is None or count <= high):
            return str(boundary["band"])
    return UNSEEN_BAND


def data_revision(paths: Iterable[str | Path] | None = None) -> str:
    """A short digest of the dataset files, for keying cached artifacts.

    Two checkouts that rebuilt the dataset from the same release produce the
    same revision, so caches survive a rebuild but not a data change. Files
    that do not exist are skipped rather than raising: the all-subjects split
    is built only for rung 3, and the qualifier sidecar is optional, but the
    revision has to be obtainable either way. A file's absence still changes
    the revision, because its name is only mixed in when it is there.
    """
    if paths is None:
        paths = [
            *SPLIT_FILES.values(),
            *VOCABULARY_FILES.values(),
            *NAME_QUALIFIER_FILES.values(),
        ]
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        if not path.exists():
            continue
        digest.update(path.name.encode())
        with _open_binary(path) as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()[:12]


def _resolve(
    path: str | Path | None, known: dict[str, Path], name: str, kind: str
) -> Path:
    """An explicit path, or the registered one for `name`."""
    if path is not None:
        return Path(path)
    source = known.get(name)
    if source is None:
        raise KeyError(f"unknown {kind} {name!r} (known: {', '.join(known)})")
    return source


def _open(path: Path):
    try:
        return path.open(newline="", encoding="utf-8")
    except FileNotFoundError as error:
        raise MissingDataset(f"{path} is missing.\n{REBUILD_HINT}") from error


def _open_binary(path: Path):
    try:
        return path.open("rb")
    except FileNotFoundError as error:
        raise MissingDataset(f"{path} is missing.\n{REBUILD_HINT}") from error


def _record(row: dict[str, str]) -> Record:
    subjects = tuple((row.get("subjects") or "").split())
    return Record(
        id=row["id"],
        type=row.get("type", ""),
        lang=_language(row.get("lang", "")),
        title=row.get("title") or "",
        abstract=row.get("abstract") or "",
        subjects=subjects,
    )


def _language(value: str) -> str:
    """The CSV carries an ISO-639 URI; the evaluator's cells use the code."""
    return value.rstrip("/").rsplit("/", 1)[-1] if value else ""


def entry_from_release(raw: dict) -> VocabularyEntry:
    """One entry of a release vocabulary file, as the pipeline's contract type."""
    return VocabularyEntry(
        code=raw["Code"],
        name=raw.get("Name", ""),
        classification_name=raw.get("Classification Name", ""),
        classification_number=str(raw.get("Classification Number", "")),
        alternate_names=tuple(raw.get("Alternate Name", []) or ()),
        related_subjects=tuple(raw.get("Related Subjects", []) or ()),
        definition=raw.get("Definition", "") or "",
    )
