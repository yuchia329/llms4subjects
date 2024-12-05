import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from data_handler import load_dev_input_label
from try_4_metadata import get_subject_metadata, merge_subject_metadata
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
LABEL_SIZE=5
df = load_dev_input_label(label_size=LABEL_SIZE, records_size=500)
GND_ids = df["subjects"].str.split(by=" ").explode()
unique_GNDs = set(GND_ids)
print(unique_GNDs)
subject_metadata_mapping, _ = get_subject_metadata(unique_GNDs)
merge_labels = merge_subject_metadata(list(subject_metadata_mapping.values()))
# Load E5 model and tokenizer
model_name = "intfloat/e5-mistral-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)#, device_map=DEVICE)

def embed_texts(texts):
    """
    Embed a list of texts using the E5 model.
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)#.to(DEVICE)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.cpu().numpy()

# Step 1: Embed labels
labels = merge_labels
label_embeddings = embed_texts(labels)

# Step 2: Embed input articles
articles = df["input"].to_list()
article_embeddings = embed_texts(articles)

# Step 3: Compute similarity
similarities = cosine_similarity(article_embeddings, label_embeddings)  # Shape: (num_articles, num_labels)

# Step 4: Predict top-k labels or apply threshold
top_k = 5  # Predict top 5 labels
predictions = torch.topk(torch.tensor(similarities), k=top_k, dim=1).indices.numpy()
print(predictions)
