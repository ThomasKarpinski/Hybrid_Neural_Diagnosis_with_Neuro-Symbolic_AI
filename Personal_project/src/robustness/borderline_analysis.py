"""
Analyze model behavior on borderline cases.

We adapt the 'mean_radius' example to CDC Diabetes dataset:
- analyze BMI borderline ranges (e.g., [24,26] or [29,31] for overweight/obesity boundary)
- analyze Age borderline ranges (e.g., [44,46] or [59,61] depending on interest)

The function will:
 - select samples whose given feature falls into a specified interval
 - report metrics and probability distribution for that subset
 - optionally compare to full test set
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss


def _compute_metrics_from_probs(probs: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (probs >= threshold).astype(int)
    y_true = y_true.reshape(-1).astype(int)
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred))
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        metrics["roc_auc"] = float('nan')
    metrics["brier"] = float(brier_score_loss(y_true, probs))
    return metrics


def analyze_borderline_feature(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_index: int,
    feature_name: str,
    low: float,
    high: float,
    feature_names: list = None,
    device: str = None,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Select samples where feature_value ∈ [low, high] and analyze model predictions.

    - feature_index: column index in X (if X is numpy array)
    - feature_name: used for plotting/title
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.DataFrame(X, columns=feature_names) if feature_names is not None else pd.DataFrame(X)
    mask = (df.iloc[:, feature_index] >= low) & (df.iloc[:, feature_index] <= high)
    idx = mask.values.nonzero()[0]

    if len(idx) == 0:
        raise ValueError(f"No samples found in {feature_name} range [{low}, {high}]")

    X_subset = X[idx]
    y_subset = y[idx]

    # get probabilities
    model.eval()
    X_t = torch.tensor(X_subset, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(X_t).cpu().numpy().reshape(-1)

    subset_metrics = _compute_metrics_from_probs(probs, y_subset)

    # full-test metrics for comparison
    X_t_all = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs_all = model(X_t_all).cpu().numpy().reshape(-1)
    full_metrics = _compute_metrics_from_probs(probs_all, y)

    if plot:
        plt.figure(figsize=(8, 3))
        plt.hist(probs_all, bins=30, alpha=0.6, label='All test probs')
        plt.hist(probs, bins=30, alpha=0.6, label=f'{feature_name} in [{low},{high}]')
        plt.xlabel('Predicted probability')
        plt.legend()
        plt.title(f'Probability distribution — {feature_name} borderline [{low},{high}]')
        plt.tight_layout()
        plt.show()

    return {
        "feature_name": feature_name,
        "range": (low, high),
        "n_samples": int(len(idx)),
        "subset_metrics": subset_metrics,
        "full_metrics": full_metrics,
        "indices": idx.tolist()
    }

