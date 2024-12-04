import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
import os
from data_handler import load_training_data
from label_metadata import generateLabelMetadata
import torch.nn as nn
from torch.optim import AdamW
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
df, unique_label_set = load_training_data()
# Define constants
MODEL_NAME = "FacebookAI/xlm-roberta-base"  # "xlm-roberta-base" "bert-base-multilingual-cased"
NUM_LABELS = len(unique_label_set)  # Number of unique subjects
BATCH_SIZE = 32
MAX_LEN = 512
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

# Load pretrained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(DEVICE)

# Custom Dataset
class MultiLabelDataset(Dataset):
    def __init__(self, articles, labels, tokenizer, max_len=MAX_LEN):
        """
        :param articles: List of concatenated article titles and abstracts.
        :param labels: List of binary label vectors (size NUM_LABELS).
        :param tokenizer: Pretrained tokenizer.
        :param max_len: Maximum token length for truncation.
        """
        self.articles = articles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        labels = self.labels[idx]

        # Tokenize article text
        encoding = self.tokenizer(
            article,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels.to_numpy(), dtype=torch.float32).squeeze()
        }


# Example data
texts = df["input"]
labels = df.drop("input")


# Convert subject_embeddings to a tensor
subject_embeddings = generateLabelMetadata(unique_label_set)
# subject_codes = list(subject_embeddings.keys())
subject_tensor = torch.tensor(np.stack(list(subject_embeddings.values()))).to(DEVICE)  # Shape: (NUM_LABELS, EMBEDDING_DIM)
# Create dataset and dataloader
dataset = MultiLabelDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Linear layers for dimension alignment
class ProjectionMapper(nn.Module):
    def __init__(self, bert_dim=768, minilm_dim=384, shared_dim=512):
        super(ProjectionMapper, self).__init__()
        self.bert_projection = nn.Linear(bert_dim, shared_dim)
        self.minilm_projection = nn.Linear(minilm_dim, shared_dim)

    def forward(self, bert_embeddings, minilm_embeddings):
        bert_shared = self.bert_projection(bert_embeddings)
        minilm_shared = self.minilm_projection(minilm_embeddings)
        return bert_shared, minilm_shared

mapper = ProjectionMapper().to(DEVICE)

# Optimizer and loss
optimizer = AdamW(list(model.parameters()) + list(mapper.parameters()), lr=5e-5)
criterion = torch.nn.BCEWithLogitsLoss()
# Training loop
EPOCHS = 5
for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Get BERT embeddings
        outputs = model(input_ids, attention_mask=attention_mask)
        bert_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

        # Map BERT embeddings to shared space
        bert_shared, subject_shared = mapper(bert_embeddings, subject_tensor)

        # Compute similarity scores
        logits = torch.matmul(bert_shared, subject_shared.T)  # Shape: (batch_size, NUM_LABELS)

        # Compute loss
        loss = criterion(logits, labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()
    print('duration: ', time.time() - start)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.6f}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model_xlm_64")
tokenizer.save_pretrained("fine_tuned_model_xlm_64")