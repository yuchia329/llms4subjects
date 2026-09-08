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
from typing import Iterable

from .contracts import Record, VocabularyEntry
from .paths import NAME_QUALIFIER_FILES, SPLIT_FILES, VOCABULARY_FILES

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
