import polars as pl
from collections import Counter

def load_training_data_one_hot_labelsets():
    file_path = "TIBKAT_dataset/core_train.csv"
    df = pl.read_csv(file_path)
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

# generate unique_labels set
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
    file_path = "TIBKAT_dataset/core_train.csv"
    df = pl.read_csv(file_path)
    unique_labels = get_TIBKAT_unique_labels(df)
    return unique_labels


def load_unique_dev_label(records_size=None):
    file_path = "TIBKAT_dataset/core_dev.csv"
    df = pl.read_csv(file_path)
    if records_size is not None:
        df = df[:records_size]
    
    unique_labels = get_TIBKAT_unique_labels(df)
    return unique_labels

def load_dev_data_one_hot(label_size = None):
    df = pl.read_csv("TIBKAT_dataset/core_dev.csv", row_index_name="id")
    df = df.with_columns(
        pl.concat_str(["title", "abstract"], separator=" ").alias("input")
    )
    if label_size:
        df = df.filter(
            pl.col("subjects").str.split(by=" ").list.len()==label_size
            )
    # print(df)

    df = df["id", "input", "subjects"]
    all_labels = load_unique_training_label()
    
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


def load_dev_data(label_size=None, records_size=None):
    df = pl.read_csv("TIBKAT_dataset/core_dev.csv", row_index_name="id")
    df = df.with_columns(
        pl.concat_str(["title", "abstract"], separator=" ").alias("input")
    )
    df = df["input", "subjects"]
    if records_size is not None:
        df = df[:records_size]

    if label_size:
        df = df.filter(
            pl.col("subjects").str.split(by=" ").list.len()==label_size
            )
    return df


