import polars as pl
import json
import os, glob
import csv

dir = 'TIBKAT/tib-core-subjects/data'
for dataset in os.listdir(dir):
    output_file = f"TIBKAT_dataset/core_{dataset}.csv"
    next_dir_1 = os.path.join(dir, dataset)
    for data_type in os.listdir(next_dir_1):
        next_dir_2 = os.path.join(next_dir_1, data_type)
        for lang in os.listdir(next_dir_2):
            next_dir_3 = os.path.join(next_dir_2, lang)
            for item in os.listdir(next_dir_3):
                filepath = os.path.join(next_dir_3, item)
                # TIBKAT_sources_path.append(dir)
                if os.path.isfile(filepath):
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = json.load(file)
                        graph = content.get("@graph")
                        if graph:
                            filtered_data = [item for item in graph if "title" in item]
                            filtered_data = filtered_data[0]
                            title = filtered_data.get("title")
                            abstract = filtered_data.get("abstract")
                            subject = filtered_data.get("dcterms:subject")
                            if not isinstance(subject, list):
                                subject = [subject]
                            subjects = " ".join(item["@id"] for item in subject)
                            data = [{"title": title, "abstract": abstract, "subjects": subjects}]
                            file_exists = os.path.isfile(output_file)
                            with open(output_file, mode="a" if file_exists else "w", encoding="utf-8", newline="") as csvfile:
                                headers = data[0].keys()
                                writer = csv.DictWriter(csvfile, fieldnames=headers)
                                if not file_exists:
                                    writer.writeheader()
                                writer.writerows(data)                            
                
            
# df = pl.read_json(file_path)
# pl.Config.set_fmt_str_lengths = 200
# pl.Config.set_tbl_width_chars = 200
# df = df.select(["Code", "Name", "Alternate Name", "Related Subjects", "Classification Name", "Definition"])
# df.write_csv('GND_core.csv')
# with pl.Config() as cfg:
#     cfg.set_tbl_cols(20)
#     print(df)