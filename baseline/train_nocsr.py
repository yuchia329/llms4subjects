"""The XLM-R variant of the archived classifier, without the sparse label matrix.

Same standing as `baseline/train.py`: archived, not developed, kept runnable so
its numbers can be a row in the results table.

    python -m baseline.train_nocsr --smoke
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv
from transformers import AutoModel, AutoTokenizer

from baseline.data_handler import load_training_data_one_hot_labelsets
from baseline.label_metadata import concat_subject_metadata, get_subject_metadata
from llms4subjects.hardware import describe_device, select_device

max_len = 256
model_name = "xlm-roberta-base"

# --------------------------
# 1. Custom Dataset
# --------------------------


class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        """
        A dataset for multi-label classification without using csr_matrix.

        Args:
            texts: List of input texts.
            labels: Dense PyTorch tensor or NumPy array with shape (num_samples, num_labels).
            tokenizer: Pretrained tokenizer.
            max_len: Maximum sequence length.
        """
        self.texts = texts
        # Convert labels to tensor
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]  # Directly access dense labels
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label
        }


# --------------------------
# 2. Label Graph Refiner
# --------------------------
class LabelGraphRefiner(nn.Module):
    def __init__(self, num_labels, label_features, edge_index):
        super(LabelGraphRefiner, self).__init__()
        self.label_embeddings = nn.Parameter(label_features)
        self.conv1 = GCNConv(label_features.shape[1], 128)
        self.conv2 = GCNConv(128, 64)
        self.edge_index = edge_index  # Correct edge_index

    def forward(self):
        x = F.relu(self.conv1(self.label_embeddings, self.edge_index))
        x = self.conv2(x, self.edge_index)
        return x


# --------------------------
# 3. Label-Aware Multi-Label Classifier
# --------------------------
class LabelAwareMultiLabelClassifier(nn.Module):
    def __init__(self, text_model, num_labels, label_features=None, edges=None):
        super(LabelAwareMultiLabelClassifier, self).__init__()
        self.text_model = text_model  # Pretrained language model
        self.label_graph_refiner = LabelGraphRefiner(
            num_labels, label_features, edges)
        self.classifier = nn.Linear(
            text_model.config.hidden_size + 64,  # Include refined label embedding size
            num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        # Refine label embeddings
        refined_label_embeddings = self.label_graph_refiner()

        # Get text embeddings from the pretrained model
        text_outputs = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask)
        # CLS token (batch_size, embedding_dim)
        text_embeddings = text_outputs.last_hidden_state[:, 0]

        # Expand refined label embeddings to match batch size
        batch_size = text_embeddings.size(0)
        expanded_label_embeddings = refined_label_embeddings.mean(
            dim=0).unsqueeze(0).repeat(batch_size, 1)

        # Combine text and label embeddings
        combined_embeddings = torch.cat(
            [text_embeddings, expanded_label_embeddings], dim=-1
        )

        # Classifier
        logits = self.classifier(combined_embeddings)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# --------------------------
# 4. Metrics
# --------------------------
def compute_metrics(logits, labels, k=5):
    """
    Compute Precision@K, Recall@K, and F1@K for multi-label classification.

    Args:
        logits: numpy.ndarray of shape (num_samples, num_labels)
            Predicted logits or probabilities for each label.
        labels: numpy.ndarray of shape (num_samples, num_labels)
            Ground truth binary label matrix.
        k: int
            Number of top predictions to consider for metrics.

    Returns:
        metrics: dict
            Dictionary containing P@K, R@K, and F1@K scores.
    """
    # Sigmoid to convert logits to probabilities
    probabilities = 1 / (1 + np.exp(-logits))

    # Get the indices of the top K predictions for each record
    top_k_indices = np.argsort(-probabilities, axis=1)[:, :k]

    # Initialize counters
    precision_scores = []
    recall_scores = []

    # Loop through each record
    for i in range(labels.shape[0]):
        true_labels = np.where(labels[i] == 1)[0]  # Indices of true labels
        # Indices of top K predictions
        predicted_top_k = top_k_indices[i]

        # Calculate intersection
        true_positives = len(set([int(label) for label in true_labels]) & set(
            [label.item() for label in predicted_top_k]))

        # Precision@K for this record
        precision = true_positives / k
        precision_scores.append(precision)

        # Recall@K for this record
        recall = true_positives / \
            len(true_labels) if len(true_labels) > 0 else 0
        recall_scores.append(recall)

    # Average Precision@K and Recall@K across all records
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)

    # Compute F1@K
    avg_f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)
              if avg_precision + avg_recall > 0 else 0)

    res = {
        "Precision@K": avg_precision,
        "Recall@K": avg_recall,
        "F1@K": avg_f1
    }
    print(res)
    # Return metrics
    return res

# --------------------------
# 5. Training and Evaluation
# --------------------------


def train(model, train_loader, val_loader, optimizer, device, num_epoch):
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        start = time.time()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = total_loss / len(train_loader)
        print(f"Epoch {
              epoch + 1}: Avg Train Loss = {train_loss:.6f} Duration: {time.time()-start:.2f}")
        model.eval()
        total_loss = 0
        start = time.time()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                total_loss += loss.item()
                val_loss = total_loss / len(val_loader)
            print(f"Epoch {
                  epoch + 1}: Avg Vali Loss = {val_loss:.6f} Duration: {time.time()-start:.2f}")


def evaluate(model, data_loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics_1 = compute_metrics(all_logits, all_labels, k=1)
    metrics_2 = compute_metrics(all_logits, all_labels, k=3)
    metrics_3 = compute_metrics(all_logits, all_labels, k=5)
    return metrics_1, metrics_2, metrics_3


# --------------------------
# 6. Entrypoint
# --------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="train on a small prefix of the split, to reach "
                             "the first training step on a laptop")
    parser.add_argument("--records", type=int, default=None,
                        help="use only the first N training records")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    args = parser.parse_args(argv)
    if args.smoke:
        args.records = args.records or 256
        args.epochs = 1
        args.batch_size = min(args.batch_size, 8)
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(select_device(args.device))
    print(describe_device(str(device)))

    df, unique_labels = load_training_data_one_hot_labelsets(records_size=args.records)
    texts_train = df["input"]
    labels = df.drop("input")
    x_train, x_test, y_train, y_test = train_test_split(
        texts_train, labels, test_size=0.3, random_state=42)

    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=42)

    # Dense label matrix; this is the variant without csr_matrix.
    labels_train = np.array(y_train.to_numpy().tolist())
    labels_val = np.array(y_val.to_numpy().tolist())
    labels_test = np.array(y_test.to_numpy().tolist())

    # Tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = MultiLabelDataset(x_train, labels_train, tokenizer, max_len)
    val_dataset = MultiLabelDataset(x_val, labels_val, tokenizer, max_len)
    test_dataset = MultiLabelDataset(x_test, labels_test, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Pretrained language model
    text_model = AutoModel.from_pretrained(model_name)

    # Generate label embeddings using SentenceTransformer
    subject_metadata_mapping, label_hierarchy = get_subject_metadata(unique_labels)
    label_descriptions = concat_subject_metadata(
        list(subject_metadata_mapping.values()))
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    label_features = torch.tensor(embedding_model.encode(
        label_descriptions), dtype=torch.float32).to(device)

    # Should be (num_labels, embedding_dim)
    print(f"Label embeddings shape: {label_features.shape}")

    # Flatten labels into a list
    flattened_list = []
    for parent, children in label_hierarchy.items():
        flattened_list.append(parent)
        flattened_list.extend(children)

    # Map labels to indices
    label_to_index = {label: idx for idx, label in enumerate(flattened_list)}

    # Define edges based on parent-child relationships
    edges = []
    for parent, children in label_hierarchy.items():
        parent_idx = label_to_index[parent]
        for child in children:
            child_idx = label_to_index[child]
            edges.append((parent_idx, child_idx))  # Parent -> Child
            # Child -> Parent (bidirectional)
            edges.append((child_idx, parent_idx))

    # Convert to PyTorch tensor and transpose
    edges = torch.tensor(edges, dtype=torch.long).t()

    print(f"edge_index shape: {edges.shape}, dtype: {edges.dtype}")

    self_loops = torch.arange(len(label_to_index), dtype=torch.long).repeat(2, 1)
    edges = torch.cat([edges, self_loops], dim=1).to(device)

    # Initialize model with embeddings
    model = LabelAwareMultiLabelClassifier(
        text_model, len(label_descriptions), label_features, edges).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train(model=model, train_loader=train_loader, val_loader=val_loader,
          optimizer=optimizer, device=device, num_epoch=args.epochs)
    return evaluate(model=model, data_loader=test_loader, device=device)


if __name__ == "__main__":
    main()
