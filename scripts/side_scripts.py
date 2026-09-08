"""Archived one-off analysis scripts. Run from the repository root.

These predate the pipeline and are kept for reference, not developed. Note that
`clean_GND_labels` rewrites a vocabulary file in place — that class of edit is
what produced the lossy vocabulary the rebuild replaced, so treat
`build_tibkat_csv.py` as the only thing allowed to write `GND_dataset/`.
"""

from collections import Counter
import matplotlib.pyplot as plt
import polars as pl
import json
import os, glob
import csv

def generate_train_label_dist():
    df = pl.read_csv("TIBKAT_dataset/core_train.csv")
    data = df["subjects"].str.split(by=" ").explode()
    print(data)
    subjects = []
    frequency = []
    # Count occurrences
    counter = Counter(data)
    for key, count in counter.items():
        subjects.append(key)
        frequency.append(count)
    new_df = pl.DataFrame({"subject":subjects,"frequency":frequency})
    new_df.write_csv("train_label_dist.csv")


def clean_GND_labels():
    file_path = '../llms4subjects/shared-task-datasets/GND/dataset/GND-Subjects-tib-core.json'
    df = pl.read_json(file_path)
    pl.Config.set_fmt_str_lengths = 200
    pl.Config.set_tbl_width_chars = 200
    df = df.select(["Code", "Name", "Alternate Name", "Related Subjects", "Classification Name", "Definition"])
    df.write_csv('GND_core.csv')
    with pl.Config() as cfg:
        cfg.set_tbl_cols(20)
        print(df)


def generate_duplicate_subject_name_json():
    df = pl.read_csv("TIBKAT_dataset/core_dev.csv")
    data = df["subjects"].str.split(by=" ").explode()
    
    with open("GND_dataset/GND-Subjects-all.json", mode="r") as file:
        label_mapping = json.load(file)
        name_id_mapping = {}
        for id in data:
            value = name_id_mapping.get(label_mapping[id]["Name"])
            if value is None:
                name_id_mapping[label_mapping[id]["Name"]] = set([id])#[label_mapping[id]]
            else:
                name_id_mapping[label_mapping[id]["Name"]].add(id)
        
        filtered_data = {key: [label_mapping[gnd] for gnd in value] for key, value in name_id_mapping.items() if len(value) > 1}
        with open('duplicate_labels.json', 'w') as f:
            json.dump(filtered_data, f)


def merge_TIBKAT_files():
    """Removed. Use build_tibkat_csv.py instead.

    This walked the JSON-LD tree with a fixed train/dev directory depth, appended to
    the output file on every run, kept no record id, and wrote list-valued abstracts
    through str(), which is how the previous CSVs picked up duplicate and malformed
    rows. See legacy/README.md for the damage it did.
    """
    raise NotImplementedError("use build_tibkat_csv.py")


def generate_subject_mapping_file():
    with open("GND/GND-Subjects-all.json", mode="r") as file:
        content = json.load(file)
        data = {}
        for item in content:
            code = item.get("Code")
            data[code] = item
        with open("GND_dataset/GND-Subjects-all.json", mode="w", encoding='utf-8') as newfile:
            json.dump(data, newfile, ensure_ascii=False, indent=4)


def check_labels():
    y_label_sets = set()
    with open("GND_dataset/GND-Subjects-all.json", mode="r") as jsonfile:
        tags = json.load(jsonfile)
        for key in tags.keys():
            y_label_sets.add(key)
    print(len(y_label_sets))
    x_label_sets = set()
    with open("TIBKAT_dataset/core_dev.csv", mode="r") as file:
        reader = csv.reader(file, delimiter=',')
        # content = content[1:10]
        contents = [line for line in reader]
        contents = contents[1:]
        for content in contents:
            labels = content[2].split(" ")
            for label in labels:
                x_label_sets.add(label)
                y_label_sets.add(label)
    print(len(y_label_sets))
    print(len(x_label_sets))