import os
import json
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.utils.inference import predict_proba
from src.models.metrics import compute_metrics, select_threshold


def evaluate_model(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    tune_threshold: bool = False,
    tune_metric: str = "f1",
    save_path: str = "experiments/hpo_results/baseline_metrics.json",
    batch_size: int = 4096,
) -> Dict[str, Any]:
    """Evaluate a trained model.

    Parameters
    ----------
    tune_threshold:
        If True, select the threshold on the provided validation set (X_val,y_val)
        using `tune_metric` (default: F1). The selected threshold is then frozen
        and used to compute TEST metrics.

    Notes
    -----
    - This function never tunes threshold on the test set (IJCNN standard).
    - Model outputs are logits; probabilities are computed via sigmoid.
    """

    if tune_threshold and (X_val is None or y_val is None):
        raise ValueError("tune_threshold=True requires X_val and y_val")

    # optional threshold tuning on validation set
    if tune_threshold:
        probs_val = predict_proba(model, X_val, batch_size=batch_size)
        threshold = select_threshold(probs_val, y_val, metric=tune_metric)
        print(f"Threshold selected on VAL: {threshold:.3f} (metric={tune_metric})")

    probs_test = predict_proba(model, X_test, batch_size=batch_size)
    metrics = compute_metrics(probs_test, y_test, threshold=threshold)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # console summary
    print("=== Model Evaluation ===")
    print(f"Threshold: {metrics['threshold']:.3f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}")
    print(f"Brier:     {metrics['brier_score']:.4f}")
    if save_path:
        print(f"Saved metrics to: {save_path}")

    return metrics
