from sentence_transformers import SentenceTransformer
import json

def concat_subject_metadata(subject_metadata_array):
    merge_array = []
    for item in subject_metadata_array:
        metadata_text = (
            item.get("Name")
            + " " + item.get("Classification Name")
            + " " + " ".join(item.get("Alternate Name", []))
            # + " " + " ".join(item.get("Related Subjects", []))
            # + " " + item.get("Definition","")
        )
        merge_array.append(metadata_text)
    return merge_array

def get_subject_metadata(unique_label_set):
    with open("GND_dataset/GND-Subjects-all.json", mode="r") as file:
        label_mapping = json.load(file)
        subject_metadata_mapping = {}
        category_subject_mapping = {}
        for label in unique_label_set:
            label_metadata = label_mapping.get(label)
            if category_subject_mapping.get(label) is None:
                category_subject_mapping[label_metadata["Classification Name"]] = [label]
            else:
                category_subject_mapping[label_metadata["Classification Name"]].append(label)

            core_label_metadata = {
                "Code": label_metadata.get("Code"),
                "Classification Name": label_metadata.get("Classification Name"),
                "Name": label_metadata.get("Name"),
                "Alternate Name": label_metadata.get("Alternate Name",[]),
                "Related Subjects": label_metadata.get("Related Subjects",[]),
                "Definition": label_metadata.get("Definition",""),
            }
            subject_metadata_mapping[label] = core_label_metadata
        return subject_metadata_mapping, category_subject_mapping

def generateLabelMetadata(unique_label_set):
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    with open("GND_dataset/GND-Subjects-all.json", mode="r") as file:
        label_mapping = json.load(file)
        subjects_metadata = []
        for label in unique_label_set:
            label_metadata = label_mapping.get(label)
            core_label_metadata = {
                "Code": label_metadata.get("Code"),
                "Classification Name": label_metadata.get("Classification Name"),
                "Name": label_metadata.get("Name"),
                "Alternate Name": label_metadata.get("Alternate Name",[]),
                "Related Subjects": label_metadata.get("Related Subjects",[]),
                "Definition": label_metadata.get("Definition",""),
            }
            subjects_metadata.append(core_label_metadata)

    subject_embeddings = {}
    for subject in subjects_metadata:
        metadata_text = (
            subject["Name"] + " " +
            subject["Classification Name"] + " " +
            " ".join(subject["Alternate Name"]) +
            " ".join(subject["Related Subjects"]) +
            " " + subject["Definition"]
        )
        subject_embeddings[subject["Code"]] = embedder.encode(metadata_text)

    return subject_embeddings
