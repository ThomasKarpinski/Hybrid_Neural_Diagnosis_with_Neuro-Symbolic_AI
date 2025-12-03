import os
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)
from typing import Dict, Any


def evaluate_model(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    save_path: str = "experiments/hpo_results/baseline_metrics.json",
) -> Dict[str, Any]:
    """
    Evaluate the trained model and compute:
    - Accuracy
    - F1-score
    - ROC-AUC
    - Brier Score
    - Confusion Matrix
    Returns a metrics dict and also saves JSON to disk.
    """
    model.eval()
    device = next(model.parameters()).device

    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(X_t).cpu().numpy().reshape(-1)

    preds = (probs >= threshold).astype(int)
    y_true = y_test.reshape(-1).astype(int)

    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, preds))
    metrics["f1_score"] = float(f1_score(y_true, preds))
    # roc_auc may fail if only one class present — handle gracefully
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        metrics["roc_auc"] = None
    metrics["brier_score"] = float(brier_score_loss(y_true, probs))
    metrics["confusion_matrix"] = confusion_matrix(y_true, preds).tolist()

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # print summary
    print("=== Baseline MLP Evaluation ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"Confusion matrix:\n{metrics['confusion_matrix']}")
    print(f"Saved metrics to: {save_path}")

    return metrics

