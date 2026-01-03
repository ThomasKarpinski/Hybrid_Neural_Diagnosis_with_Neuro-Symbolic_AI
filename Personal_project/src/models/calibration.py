import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from typing import Sequence, Tuple, Dict, Any
import torch


def compute_brier(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, device: str = None) -> float:
    """
    Compute Brier score on X,y. X is numpy array, y is 0/1 labels.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(X_t).cpu().numpy().reshape(-1)
    return float(brier_score_loss(y.reshape(-1).astype(int), probs))


def reliability_diagram(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    device: str = None,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Compute calibration curve and optionally plot reliability diagram.
    Returns dict with fraction_of_positives, mean_predicted_value, brier.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    import torch
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(X_t).cpu().numpy().reshape(-1)

    prob_true, prob_pred = calibration_curve(y.reshape(-1).astype(int), probs, n_bins=n_bins, strategy=strategy)
    brier = float(brier_score_loss(y.reshape(-1).astype(int), probs))

    if plot:
        plt.figure(figsize=(6, 6))
        # reliability curve
        plt.plot(prob_pred, prob_true, marker='o', label='Reliability')
        # diagonal
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Reliability diagram (n_bins={n_bins})\nBrier score = {brier:.4f}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # histogram of predicted probabilities
        plt.figure(figsize=(6, 2.5))
        plt.hist(probs, bins=20)
        plt.xlabel('Predicted probability')
        plt.title('Predicted probability histogram')
        plt.tight_layout()
        plt.show()

    return {"fraction_of_positives": prob_true, "mean_predicted_value": prob_pred, "brier": brier, "probs": probs}

