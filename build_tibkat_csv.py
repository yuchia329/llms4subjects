"""Rebuild the local dataset from the official LLMs4Subjects release.

Produces both halves of the data, none of which is tracked in git:

    TIBKAT_dataset/{core,all}_{train,dev,test}.csv   flat CSVs from the JSON-LD records
    GND_dataset/GND-Subjects-{all,tib-core}.json    vocabulary, re-keyed by GND code
    GND_dataset/GND-Name-Qualifiers-{all,tib-core}.json   preferred-name qualifier boundaries
    GND_dataset/qualifier_stats.json                how much of the vocabulary carries a qualifier

One command does the whole rebuild, sparse clone included:

    python build_tibkat_csv.py

Source: https://github.com/jd-coderepos/llms4subjects, sparse-cloned into .cache/ so it
stays out of both the repository and the import path.

The official vocabulary files are JSON lists; the code in this project indexes subjects
by their GND code, so they are converted to dicts keyed by "Code" on the way in. Entry
contents are copied verbatim — including GND's comma-qualifiers, which an earlier,
unreproducible step had stripped from 18,043 of the 79,427 tib-core entries.

Preferred names are the one place the release itself drops the qualifier boundary: the JSON
holds "Muster Struktur" where the accompanying *_dnb-skos.ttl holds "Muster (Struktur)". The
boundaries recoverable from those SKOS files are written to a sidecar map that the label-text
builder consumes; see llms4subjects/stages/label_text.py for how each mode renders them.
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import unicodedata

from llms4subjects.corpus import entry_from_release
from llms4subjects.stages.label_text import count_qualifiers

SPLITS = {
    "train": "train",
    "dev": "dev",
    "test": "test/gold-standard-testset",
}

# The output prefix follows the subset: the two tracks share record ids, so writing both
# under one name would silently swap the dataset the pipeline reads.
SUBSET_PREFIXES = {"tib-core-subjects": "core", "all-subjects": "all"}

FIELDS = ["id", "type", "lang", "title", "abstract", "subjects"]


def flatten(value):
    """JSON-LD fields are sometimes a scalar, sometimes a list of scalars."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value)


def record_node(graph):
    """The bibliographic node is the only one carrying a title."""
    for node in graph:
        if "title" in node:
            return node
    return None


def subjects_of(node):
    raw = node.get("dcterms:subject") or []
    if isinstance(raw, dict):
        raw = [raw]
    codes = []
    for entry in raw:
        code = entry.get("@id") if isinstance(entry, dict) else entry
        if isinstance(code, str) and code.startswith("gnd:"):
            codes.append(code)
    return " ".join(dict.fromkeys(codes))


def parse(path):
    with open(path, encoding="utf-8") as handle:
        node = record_node(json.load(handle)["@graph"])
    if node is None:
        return None
    record_type = flatten(node.get("@type")).split(":")[-1]
    return {
        "id": os.path.basename(path).removesuffix(".jsonld"),
        "type": record_type,
        "lang": flatten(node.get("language")),
        "title": " ".join(flatten(node.get("title")).split()),
        "abstract": " ".join(flatten(node.get("abstract")).split()),
        "subjects": subjects_of(node),
    }


GND_SUBSETS = ["all", "tib-core"]

CONCEPT = re.compile(r'^gnd:(\S+) a skos:Concept', re.MULTILINE)
PREF_LABEL = re.compile(r'skos:prefLabel "((?:[^"\\]|\\.)*)"@de')
PARENTHETICAL = re.compile(r"^(.+?) \((.+)\)$")


def parse_pref_labels(ttl):
    """Map GND code to skos:prefLabel from a Turtle file, one label per concept block."""
    labels = {}
    matches = list(CONCEPT.finditer(ttl))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(ttl)
        label = PREF_LABEL.search(ttl, match.end(), end)
        if label:
            labels[f"gnd:{match.group(1)}"] = label.group(1).replace('\\"', '"').replace("\\\\", "\\")
    return labels


def name_qualifiers(entries, pref_labels):
    """Recover the qualifier boundary the JSON flattened out of preferred names.

    Only accepted when the SKOS label reconstructs the JSON name exactly — "Muster (Struktur)"
    against "Muster Struktur" — so a label that diverges for any other reason is left alone.
    """
    qualifiers = {}
    for entry in entries:
        label = pref_labels.get(entry["Code"])
        if not label:
            continue
        match = PARENTHETICAL.match(label)
        if not match:
            continue
        term, qualifier = match.groups()
        suffix = name_suffix(entry["Name"], term, qualifier)
        if suffix is not None:
            qualifiers[entry["Code"]] = suffix
    return qualifiers


def name_suffix(name, term, qualifier):
    """The qualifier as the JSON name spells it, or None if the label does not fit.

    Comparison is on NFC, because the two files disagree about composed and decomposed
    accents, but what is stored is the raw substring: the renderer matches the name it is
    given, byte for byte, so a normalisation difference cannot silently drop a boundary.
    """
    normalise = lambda text: unicodedata.normalize("NFC", text)
    if normalise(name) != normalise(f"{term} {qualifier}"):
        return None
    for index, char in enumerate(name):
        if char == " " and normalise(name[index + 1:]) == normalise(qualifier):
            return name[index + 1:]
    return None


def write_json(payload, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


RELEASE_URL = "https://github.com/jd-coderepos/llms4subjects"

DEFAULT_REPO = os.path.join(".cache", "llms4subjects-release")

SPARSE_PATH = "shared-task-datasets"


def clone_commands(repo):
    """The sparse clone, as argv lists: blobs on demand, one directory checked out."""
    return [
        ["git", "clone", "--filter=blob:none", "--sparse", "--depth", "1", RELEASE_URL, repo],
        ["git", "-C", repo, "sparse-checkout", "set", SPARSE_PATH],
    ]


def ensure_release(repo):
    """Clone the official release if it is not already on disk.

    Each step is skipped when its result is already there, so an interrupted clone is
    resumed by re-running the build rather than by deleting the directory by hand.
    """
    if os.path.isdir(os.path.join(repo, SPARSE_PATH)):
        return repo
    parent = os.path.dirname(repo)
    if parent:
        os.makedirs(parent, exist_ok=True)
    clone, sparse = clone_commands(repo)
    for command in ([sparse] if os.path.isdir(os.path.join(repo, ".git")) else [clone, sparse]):
        print("$ " + " ".join(command))
        subprocess.run(command, check=True)
    return repo


def build_gnd(repo, out_dir):
    """Re-key the official vocabulary lists by GND code and record their qualifiers."""
    base = os.path.join(repo, "shared-task-datasets", "GND", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for subset in GND_SUBSETS:
        with open(os.path.join(base, f"GND-Subjects-{subset}.json"), encoding="utf-8") as handle:
            entries = json.load(handle)
        with open(os.path.join(base, f"GND-Subjects-{subset}_dnb-skos.ttl"), encoding="utf-8") as handle:
            qualifiers = name_qualifiers(entries, parse_pref_labels(handle.read()))

        out_path = os.path.join(out_dir, f"GND-Subjects-{subset}.json")
        write_json({entry["Code"]: entry for entry in entries}, out_path)
        write_json(qualifiers, os.path.join(out_dir, f"GND-Name-Qualifiers-{subset}.json"))

        stats[subset] = count_qualifiers(
            (entry_from_release(entry) for entry in entries), qualifiers
        )
        counts = stats[subset]
        print(f"{len(entries):6d} subjects -> {out_path}")
        print(f"       {counts['entries_with_qualifier']:6d} entries and "
              f"{counts['terms_with_qualifier']} terms carry a qualifier "
              f"{counts['by_field']}")
    write_json(stats, os.path.join(out_dir, "qualifier_stats.json"))


def build(repo, subset, out_dir):
    base = os.path.join(repo, "shared-task-datasets", "TIBKAT", subset, "data")
    prefix = SUBSET_PREFIXES[subset]
    os.makedirs(out_dir, exist_ok=True)
    for split, rel in SPLITS.items():
        name = f"{prefix}_{split}"
        paths = sorted(glob.glob(os.path.join(base, rel, "**", "*.jsonld"), recursive=True))
        rows = [row for row in (parse(p) for p in paths) if row]
        out_path = os.path.join(out_dir, f"{name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        unlabelled = sum(1 for row in rows if not row["subjects"])
        print(f"{name:5s} {len(rows):6d} records -> {out_path} ({unlabelled} without subjects)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help="clone of jd-coderepos/llms4subjects; cloned here if absent")
    parser.add_argument("--subset", default="tib-core-subjects",
                        choices=["tib-core-subjects", "all-subjects"])
    parser.add_argument("--out", default="TIBKAT_dataset")
    parser.add_argument("--gnd-out", default="GND_dataset")
    parser.add_argument("--skip-gnd", action="store_true")
    args = parser.parse_args()
    ensure_release(args.repo)
    build(args.repo, args.subset, args.out)
    if not args.skip_gnd:
        build_gnd(args.repo, args.gnd_out)
