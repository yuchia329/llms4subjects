import torch
from transformers import AutoTokenizer, AutoModel, MarianMTModel, MarianTokenizer
import faiss
import numpy as np

# Load E5 model and tokenizer
model_name = "intfloat/e5-mistral-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Translation function (optional)
def translate_texts(texts, source_lang="de", target_lang="en"):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Embedding function
def generate_embeddings(texts, tokenizer, model, max_length=256, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.cpu().numpy()

# Step 1: Translate German labels to English (if necessary)
german_labels = ["Maschinelles Lernen", "Gesundheitswesen", "Natürliche Sprachverarbeitung"]
translated_labels = translate_texts(german_labels, source_lang="de", target_lang="en")
print("Translated Labels:", translated_labels)

# Step 2: Generate label embeddings
label_texts = translated_labels  # Use translated labels if needed
print("Generating label embeddings...")
label_embeddings = generate_embeddings(label_texts, tokenizer, model)

# Step 3: Build FAISS index
embedding_dim = label_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
faiss.normalize_L2(label_embeddings)
index.add(label_embeddings)

# Step 4: Embed inputs
input_texts = [
    "Machine Learning Techniques in Healthcare",  # English
    "Natürliche Sprachverarbeitung für Chatbots"  # German
]
translated_inputs = translate_texts(input_texts, source_lang="de", target_lang="en")  # Optional for German inputs
print("Generating input embeddings...")
input_embeddings = generate_embeddings(translated_inputs, tokenizer, model)
faiss.normalize_L2(input_embeddings)

# Step 5: Perform search
k = 5
distances, indices = index.search(input_embeddings, k)

# Display results
for i, input_text in enumerate(input_texts):
    print(f"\nInput: {input_text}")
    print("Top-k Labels:")
    for j in range(k):
        label_id = indices[i, j]
        print(f"  Label {label_id} (distance: {distances[i, j]:.4f})")
