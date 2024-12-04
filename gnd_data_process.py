import polars as pl
import json

file_path = '../llms4subjects/shared-task-datasets/GND/dataset/GND-Subjects-tib-core.json'
df = pl.read_json(file_path)
pl.Config.set_fmt_str_lengths = 200
pl.Config.set_tbl_width_chars = 200
df = df.select(["Code", "Name", "Alternate Name", "Related Subjects", "Classification Name", "Definition"])
df.write_csv('GND_core.csv')
with pl.Config() as cfg:
    cfg.set_tbl_cols(20)
    print(df)