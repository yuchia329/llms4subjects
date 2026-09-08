"""Evaluation for the archived classifier, against the clean dev split.

Scores a checkpoint directory. Note that `baseline/train.py` does not currently
write one — its save calls are commented out, so the checkpoints this was run
against were produced by hand. See the defect list in `baseline/__init__.py`.

Its metrics are the original ones and are NOT the project's measurement seam:
comparable numbers come from the shared evaluator with frozen bands (ticket 03),
which is what ticket 14 scores this model through.

    python -m baseline.eval_bert_rebota --model-path fine_tuned_model_xlm_64
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from baseline.data_handler import load_dev_data_one_hot, load_unique_training_label
from baseline.label_metadata import generateLabelMetadata
from llms4subjects.hardware import describe_device, select_device

LABEL_SIZE = 1
BATCH_SIZE = 64
MAX_LEN = 512
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

def evaluation(model, dataloader, DEVICE, k=LABEL_SIZE):
    # Example usage
    model.eval()
    all_predictions = []
    all_labels = []
    start = time.time()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.last_hidden_state[:, 0, :]  # CLS embeddings or classification logits
            predictions = torch.sigmoid(logits)  # Convert logits to probabilities

            # Collect predictions and labels
            all_predictions.append(predictions)
            all_labels.append(labels)
    print('predicition time: ', time.time()-start)
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute metrics
    precision = precision_at_k(all_predictions, all_labels, k)
    recall = recall_at_k(all_predictions, all_labels, k)
    f1 = f1_at_k(all_predictions, all_labels, k)
    map_score = mean_average_precision(all_predictions, all_labels)

    print(f"Precision@{k}: {precision:.10f}")
    print(f"Recall@{k}: {recall:.10f}")
    print(f"F1@{k}: {f1:.10f}")
    print(f"Mean Average Precision (mAP): {map_score:.10f}")

def precision_at_k(predictions, labels, k):
    """
    Compute Precision@K for multi-label classification.
    :param predictions: Tensor of predicted probabilities (batch_size, num_labels).
    :param labels: Tensor of true binary labels (batch_size, num_labels).
    :param k: Number of top predictions to consider.
    :return: Precision@K score.
    """
    # Get the indices of the top-k predictions
    top_k_indices = torch.topk(predictions, k=k, dim=1).indices  # Shape: (batch_size, k)
    precision_scores = []
    for i in range(len(labels)):
        # Get true labels and top-k predicted labels
        true_labels = torch.where(labels[i] == 1)[0].tolist()
        predicted_labels = top_k_indices[i].tolist()

        # Calculate precision
        # print('predict: ', set(predicted_labels))
        # print('true: ', set(true_labels))
        correct = len(set(predicted_labels) & set(true_labels))
        precision_scores.append(correct / k)

    return sum(precision_scores) / len(precision_scores)  # Average over the batch

def recall_at_k(predictions, labels, k):
    """
    Compute Recall@K for multi-label classification.
    :param predictions: Tensor of predicted probabilities (batch_size, num_labels).
    :param labels: Tensor of true binary labels (batch_size, num_labels).
    :param k: Number of top predictions to consider.
    :return: Recall@K score.
    """
    top_k_indices = torch.topk(predictions, k=k, dim=1).indices
    recall_scores = []

    for i in range(len(labels)):
        # Get true labels and top-k predicted labels
        true_labels = torch.where(labels[i] == 1)[0].tolist()
        predicted_labels = top_k_indices[i].tolist()

        # Calculate recall
        if len(true_labels) == 0:
            recall_scores.append(0.0)  # No true labels, recall is undefined
        else:
            correct = len(set(predicted_labels) & set(true_labels))
            recall_scores.append(correct / len(true_labels))

    return sum(recall_scores) / len(recall_scores)  # Average over the batch

def f1_at_k(predictions, labels, k):
    """
    Compute F1@K for multi-label classification.
    :param predictions: Tensor of predicted probabilities (batch_size, num_labels).
    :param labels: Tensor of true binary labels (batch_size, num_labels).
    :param k: Number of top predictions to consider.
    :return: F1@K score.
    """
    precision = precision_at_k(predictions, labels, k)
    recall = recall_at_k(predictions, labels, k)
    
    if precision + recall == 0:
        return 0.0  # Avoid division by zero
    return 2 * (precision * recall) / (precision + recall)

def mean_average_precision(predictions, labels):
    """
    Compute Mean Average Precision (mAP) for multi-label classification.
    :param predictions: Tensor of predicted probabilities (batch_size, num_labels).
    :param labels: Tensor of true binary labels (batch_size, num_labels).
    :return: Mean Average Precision (mAP).
    """
    average_precisions = []

    for i in range(len(labels)):
        true_labels = torch.where(labels[i] == 1)[0].tolist()
        if len(true_labels) == 0:
            continue  # Skip instances with no true labels

        # Get sorted indices based on predictions
        sorted_indices = torch.argsort(predictions[i], descending=True).tolist()

        # Calculate precision at each relevant label
        ap = 0.0
        correct = 0
        for rank, idx in enumerate(sorted_indices, start=1):
            if idx in true_labels:
                correct += 1
                ap += correct / rank  # Precision at this rank

        average_precisions.append(ap / len(true_labels))  # Average over all true labels

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="fine_tuned_model_xlm_64",
                        help="checkpoint directory written by baseline/train.py")
    parser.add_argument("--label-size", type=int, default=LABEL_SIZE,
                        help="score only dev records carrying this many labels")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    args = parser.parse_args(argv)

    device = select_device(args.device)
    print(describe_device(device))

    df = load_dev_data_one_hot(label_size=args.label_size)
    unique_label_set = load_unique_training_label()
    num_labels = len(unique_label_set)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    bert_model = AutoModel.from_pretrained(
        args.model_path, num_labels=num_labels).to(device)
    print(f"LABEL_SIZE: {args.label_size} BATCH_SIZE: {args.batch_size}, "
          f"model: {args.model_path}")

    texts = df["input"]
    labels = df.drop("input")

    # Convert subject_embeddings to a tensor
    subject_embeddings = generateLabelMetadata(unique_label_set)
    # Shape: (NUM_LABELS, EMBEDDING_DIM)
    torch.tensor(np.stack(list(subject_embeddings.values()))).to(device)

    dataset = MultiLabelDataset(texts, labels, tokenizer, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    evaluation(bert_model, dataloader, device, k=args.label_size)


if __name__ == "__main__":
    main()