"""Data loading for the archived classifier.

Reads the clean official splits through `llms4subjects.paths`, so the paths are
correct from any working directory and there is exactly one place in the repo
that knows where the dataset lives. Nothing in `llms4subjects/` imports this
module; the dependency runs one way only.
"""

import polars as pl

from llms4subjects.paths import SPLIT_FILES

TRAIN_CSV = SPLIT_FILES["core_train"]
DEV_CSV = SPLIT_FILES["core_dev"]


def load_training_data_one_hot_labelsets(records_size=None):
    """One-hot label matrix over the training split.

    `records_size` takes a prefix of the split. The full matrix is 32,043
    records by 14,607 labels, which is a GPU-host workload; the prefix exists
    so a smoke run reaches the first training step on a laptop.
    """
    df = pl.read_csv(TRAIN_CSV)
    if records_size is not None:
        df = df[:records_size]
    # combine title + abstract
    df = df.with_columns(
        pl.concat_str(["title", "abstract"], separator=" ").alias("input")
    )
    df = df["input", "subjects"]

    unique_labels = get_TIBKAT_unique_labels(df)

    # separate laels can create one hot pivot table
    df = df.with_columns(
        pl.col('subjects').str.split(' '),
        pl.lit(1).alias('__one__')
    ).explode('subjects').pivot(on='subjects', values='__one__', aggregate_function='first').fill_null(0)

    return df, unique_labels


def get_TIBKAT_unique_labels(dataframe):
    unique_labels = set()

    def filter_unique_labels(label_str):
        if label_str is None:
            return None
        labels = label_str.split()
        for label in labels:
            unique_labels.add(label)
    dataframe.with_columns(
        pl.col("subjects").map_elements(lambda x: filter_unique_labels(x), return_dtype=pl.Utf8).alias("subjects"))
    return unique_labels


def load_unique_training_label():
    df = pl.read_csv(TRAIN_CSV)
    unique_labels = get_TIBKAT_unique_labels(df)
    return unique_labels


def load_unique_dev_label(records_size=None):
    df = pl.read_csv(DEV_CSV)
    if records_size is not None:
        df = df[:records_size]

    unique_labels = get_TIBKAT_unique_labels(df)
    return unique_labels


def load_dev_data_one_hot(label_size=None):
    # the CSV carries the official TIBKAT record id, no synthetic row index needed
    df = pl.read_csv(DEV_CSV)
    df = df.with_columns(
        pl.concat_str(["title", "abstract"], separator=" ").alias("input")
    )
    if label_size:
        df = df.filter(
            pl.col("subjects").str.split(by=" ").list.len() == label_size
        )
    # print(df)

    df = df["id", "input", "subjects"]
    all_labels = load_unique_training_label()

    # Explode labels into multiple rows for efficient matching
    df_explode = df.with_columns(
        pl.col("subjects").str.split(" ")).explode("subjects")

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
    # print(df_result)
    return df_result


def load_dev_data(label_size=None, records_size=None):
    # the CSV carries the official TIBKAT record id, no synthetic row index needed
    df = pl.read_csv(DEV_CSV)
    df = df.with_columns(
        pl.concat_str(["title", "abstract"], separator=" ").alias("input")
    )
    df = df["input", "subjects"]
    if records_size is not None:
        df = df[:records_size]

    if label_size:
        df = df.filter(
            pl.col("subjects").str.split(by=" ").list.len() == label_size
        )
    return df
