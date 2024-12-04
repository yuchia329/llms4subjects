import json
import polars as pl
from collections import Counter

def load_training_data():
    # with open("GND_dataset/GND-Subjects-all.json", mode="r") as file:
    #     label_mapping = json.load(file)

    df = pl.read_csv("TIBKAT_dataset/core_train.csv")
    df = df.with_columns(
        pl.concat_str(["title", "abstract"], separator=" ").alias("input")
    )
    df = df["input", "subjects"]
    unique_labels = set()
    def filter_unique_labels(label_str):
        if label_str is None:
            return None
        labels = label_str.split()
        for label in labels:
            unique_labels.add(label)
    df.with_columns(
        pl.col("subjects").map_elements(lambda x: filter_unique_labels(x), return_dtype=pl.Utf8).alias("subjects"))
    # label_set = set()
    # label_name_set = set()
    # label_maps = {}
    # def replace_labels(label_str, mapping):
        
    #     if label_str is None:
    #         return None
    #     labels = label_str.split()
    #     for label in labels:
    #         label_set.add(label)
    #         classification_name = mapping.get(label).get("Classification Name")
    #         name = mapping.get(label).get("Name")
    #         label_name = f'{classification_name}_{name}'
    #         if label_name not in label_maps:
    #             label_maps[label_name] = set()
    #         label_maps[label_name].add(label)
        
    #     return " ".join(mapping.get(label, label).get("Name") for label in label_str.split())

    # df2 = df.with_columns(
    #     pl.col("subjects").map_elements(lambda x: replace_labels(x, label_mapping), return_dtype=pl.Utf8).alias("subjects")
    # )
    # print(len(label_set))
    df = df.with_columns(
        pl.col('subjects').str.split(' '), 
        pl.lit(1).alias('__one__')
        ).explode('subjects').pivot(on='subjects', values='__one__', aggregate_function='first').fill_null(0)

    return df, unique_labels

def load_training_label():
    df = pl.read_csv("TIBKAT_dataset/core_train.csv")
    # df = df["subjects"]
    unique_labels = set()
    def filter_unique_labels(label_str):
        if label_str is None:
            return None
        labels = label_str.split()
        for label in labels:
            unique_labels.add(label)
    
    df.with_columns(
        pl.col("subjects").map_elements(lambda x: filter_unique_labels(x), return_dtype=pl.Utf8).alias("subjects"))
    return unique_labels

def load_dev_label(records_size=None):
    df = pl.read_csv("TIBKAT_dataset/core_dev.csv")
    # df = df["subjects"]
    if records_size is not None:
        df = df[:records_size]
    
    unique_labels = set()
    def filter_unique_labels(label_str):
        if label_str is None:
            return None
        labels = label_str.split()
        for label in labels:
            unique_labels.add(label)
    
    df.with_columns(
        pl.col("subjects").map_elements(lambda x: filter_unique_labels(x), return_dtype=pl.Utf8).alias("subjects"))
    return unique_labels

def load_dev_data(label_size = None):
    df = pl.read_csv("TIBKAT_dataset/core_dev.csv", row_index_name="id")
    df = df.with_columns(
        pl.concat_str(["title", "abstract"], separator=" ").alias("input")
    )
    if label_size:
        # df = df.filter(
        #     df["subjects"].str.split(" ").arr.lengths() == label_size
        # )
        df = df.filter(
            pl.col("subjects").str.split(by=" ").list.len()==label_size
            )
    print(df)

    df = df["id", "input", "subjects"]
    all_labels = load_training_label()
    
    # Explode labels into multiple rows for efficient matching
    df_explode = df.with_columns(pl.col("subjects").str.split(" ")).explode("subjects")
    
    # Pivot into binary indicator columns
    df_pivot = df_explode.with_columns(
        pl.col('subjects').str.split(' '), 
        pl.lit(1).alias('__one__')
        ).explode('subjects').pivot(on='subjects', values='__one__', aggregate_function='first').fill_null(0)
    
    # Aggregate back to the original rows by grouping
    df_result = df_pivot.group_by("id", maintain_order=True).agg(
        [pl.all().sum()]  # Sum the binary indicators to regroup them
    )
    df_result = df_result.drop("input")
    df = df.drop("subjects")
    
    df_result = df.join(df_result, on="id", how="inner")

    # Add missing columns for labels not in the dataset
    missing_labels = all_labels - set(df_result.columns)
    for label in missing_labels:
        df_result = df_result.with_columns(pl.lit(0).alias(label))

    # Reorder columns to match the order of all_labels
    df_result = df_result.select(["input"] + list(all_labels))
    print(df_result)
    return df_result

def load_dev_input_label(label_size=None, records_size=None):
    df = pl.read_csv("TIBKAT_dataset/core_dev.csv", row_index_name="id")
    df = df.with_columns(
        pl.concat_str(["title", "abstract"], separator=" ").alias("input")
    )
    df = df["input", "subjects"]
    if records_size is not None:
        df = df[:records_size]

    if label_size:
        # df = df.filter(
        #     df["subjects"].str.split(" ").arr.lengths() == label_size
        # )
        df = df.filter(
            pl.col("subjects").str.split(by=" ").list.len()==label_size
            )
    # print(df)
    return df


def list_duplicate_subjects():
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


# list_duplicate_subjects()