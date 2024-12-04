from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_handler import load_training_label, load_dev_label, load_dev_input_label
from label_metadata import get_subject_metadata, merge_subject_metadata

LABEL_SIZE = 5
DEV_RECORD_SIZE = 15

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']
unique_label_set = load_dev_label(records_size=DEV_RECORD_SIZE)
subject_description_mapping, category_subject_mapping = get_subject_metadata(unique_label_set)
category_position_mapping = {}
position_category_mapping = {}
position = 0
for category in list(category_subject_mapping.keys()):
    category_position_mapping[category] = position
    position_category_mapping[position] = category
    position+=1

subject_position_mapping = {}
position_subject_mapping = {}
position = 0
for subject in list(subject_description_mapping.keys()):
    subject_position_mapping[subject] = position
    position_subject_mapping[position] = subject
    position+=1


MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Function to compute sentence embeddings
def get_embeddings(sentences):
    # Tokenize the input sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling to get sentence embeddings
    embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    # Perform mean pooling
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask

    return sentence_embeddings

df = load_dev_input_label(label_size=LABEL_SIZE, records_size=DEV_RECORD_SIZE)
x_true = df["input"].to_list()
y_true = [label_str.split(" ") for label_str in df["subjects"].to_list()]

y_true_flat = [subject for item in y_true for subject in item]

subject_descriptions = list(subject_description_mapping.values())

for index, x in enumerate(x_true):
    # Compute embeddings
    print("index: ", index)
    # print(f"Query: {x}\n")
    print("true_label: ", y_true[index])
    
    query_embedding = get_embeddings([x])    
    
    # find top k category
    category_embedding = get_embeddings(list(category_subject_mapping.keys()))
    similarities = cosine_similarity(query_embedding.numpy(), category_embedding.numpy())[0]
    top_k = LABEL_SIZE
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    possible_subjects = []
    for top_k_index in top_k_indices:
        category = position_category_mapping[top_k_index]
        print('category: ', category)
        for subject in category_subject_mapping[category]:
            possible_subjects.append(subject)
    possible_subject_metadata = []
    # print('subject_description_mapping: ', subject_description_mapping)
    for subject in possible_subjects:
        # print('\nsubject: ', subject)
        # print('subject_description_mapping[subject]: ', subject_description_mapping[subject])
        possible_subject_metadata.append(subject_description_mapping[subject])
    print(possible_subjects)
    # print("____________")
    # print(possible_subject_metadata)
    # print("____________")
    # print(subject_description_mapping)
    possible_subject_metadata_merge = merge_subject_metadata(possible_subject_metadata)
    
    subject_metadata_embedding = get_embeddings(possible_subject_metadata_merge)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding.numpy(), subject_metadata_embedding.numpy())[0]
    
    # Get the top-k similar sentences
    top_k = LABEL_SIZE
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]  # Sort in descending order
    
    predicted_subjects = set()
    true_subjects = set(y_true[index])
    for idx in top_k_indices:
        # print(f"Similarity: {similarities[idx]:.4f} | Predicted subjects: {position_subject_mapping[idx]}")
        predicted_subjects.add(position_subject_mapping[idx])
    correct = len(set(predicted_subjects) & set(true_subjects))
    print("predicted_label: ", list(predicted_subjects))
    print(f"accuracy: {correct/LABEL_SIZE}")
