"""
Stratified evaluation metrics for cross-generator deepfake detection.

Computes per-generator accuracy, AUC, EER, and confusion matrices
to assess how well a model trained on one manipulation type transfers to others.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
    }

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1

    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
            metrics["eer"] = compute_eer(y_true, y_prob)
        except ValueError:
            metrics["auc"] = 0.0
            metrics["eer"] = 1.0

    return metrics


def compute_eer(y_true: List[int], y_prob: List[float]) -> float:
    """Compute Equal Error Rate (where FAR == FRR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float(fpr[idx])


def cross_generator_evaluation(
    model,
    dataloaders: Dict[str, "DataLoader"],
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a trained model across multiple generator-specific test sets.

    Args:
        model: Trained PyTorch model.
        dataloaders: Dict mapping generator name -> DataLoader.
        device: Device string.

    Returns:
        Dict mapping generator name -> metrics dict.
    """
    import torch

    model.eval()
    results = {}

    for generator_name, loader in dataloaders.items():
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        results[generator_name] = compute_metrics(all_labels, all_preds, all_probs)

    return results


def print_cross_generator_table(results: Dict[str, Dict[str, float]]) -> str:
    """Format cross-generator results as a readable table."""
    header = f"{'Generator':<25} {'Accuracy':>10} {'AUC':>8} {'EER':>8} {'F1':>8}"
    separator = "-" * len(header)
    lines = [header, separator]

    for gen_name, metrics in results.items():
        line = (
            f"{gen_name:<25} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics.get('auc', 0):>8.4f} "
            f"{metrics.get('eer', 0):>8.4f} "
            f"{metrics['f1']:>8.4f}"
        )
        lines.append(line)

    return "\n".join(lines)
