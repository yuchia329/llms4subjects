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
print(unique_label_set)
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
subject_metadata = merge_subject_metadata(subject_descriptions)

for index, x in enumerate(x_true):
    # Compute embeddings
    print("index: ", index)
    # print(f"Query: {x}\n")
    print("true_label: ", y_true[index])
    query_embedding = get_embeddings([x])    
    # print(subject_metadata)
    subject_descriptions_embedding = get_embeddings(subject_metadata)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding.numpy(), subject_descriptions_embedding.numpy())[0]
    
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
