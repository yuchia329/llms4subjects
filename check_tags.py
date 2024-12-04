import json
import csv

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
            # print("title: ", content[0])
            # print("abstract: ", content[1])
            # print("subjects: ", content[2])
            # subjects = content[2].split(" ")
            # subjectNames = [tags[subject].get("Name") for subject in subjects]
            # print("subjectNames: ", subjectNames)
        # print(contents)
        # for lines in content:
        #     print(lines)
    print(len(y_label_sets))
    print(len(x_label_sets))
check_labels()
# generate_subject_mapping_file()