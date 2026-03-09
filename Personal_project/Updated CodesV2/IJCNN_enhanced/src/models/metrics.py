"""Metrics and threshold selection.

Use this to keep evaluation consistent across the project.

Publication-quality rule:
- Never tune the decision threshold on the TEST set.
- If threshold tuning is needed, tune on VALIDATION and then freeze.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalResult:
    metrics: Dict[str, Any]
    threshold: float


def select_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    metric: str = "f1",
    grid: Optional[np.ndarray] = None,
) -> float:
    """Select a threshold on probs using y_true.

    metric: "f1" (default) or "youden" (maximizes TPR-FPR).
    """
    y_true = y_true.reshape(-1).astype(int)
    probs = probs.reshape(-1)

    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)

    best_t = 0.5
    best_score = -np.inf

    if metric.lower() == "f1":
        for t in grid:
            preds = (probs >= t).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_t = float(t)
        return best_t

    if metric.lower() == "youden":
        # Youden's J = TPR - FPR
        for t in grid:
            preds = (probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            score = tpr - fpr
            if score > best_score:
                best_score = score
                best_t = float(t)
        return best_t

    raise ValueError(f"Unknown threshold metric: {metric}")


def compute_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute classification metrics for a fixed threshold."""
    y_true = y_true.reshape(-1).astype(int)
    probs = probs.reshape(-1)
    preds = (probs >= threshold).astype(int)

    out: Dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probs)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }

    try:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        out["roc_auc"] = None

    return out
