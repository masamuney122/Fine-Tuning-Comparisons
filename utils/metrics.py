import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(predictions, labels):
    """Compute accuracy, precision, recall, and F1 score."""
    preds = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary", zero_division=0)
    recall = recall_score(labels, preds, average="binary", zero_division=0)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_metrics(metrics, model_name="Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"{'=' * 50}\n")
