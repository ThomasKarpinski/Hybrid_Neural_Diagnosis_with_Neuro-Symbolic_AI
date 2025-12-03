"""
Noise sensitivity experiments.

- Adds Gaussian noise to features (in normalized feature space).
- Runs multiple repetitions at each noise level to reduce variance.
- Reports metrics across noise levels and plots metric vs noise_std curves.

Metrics collected: Accuracy, F1, ROC-AUC, Brier score.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from typing import List, Dict, Any


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


def evaluate_noise_sensitivity(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    noise_levels: List[float] = None,
    n_runs: int = 5,
    device: str = None,
    seed: int = 42,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    For each noise std in noise_levels (applied to each feature independently),
    add gaussian noise N(0, sigma^2) to X and evaluate model metrics averaged over n_runs.

    NOTE: X should be in the same feature space the model expects (i.e., normalized if model trained on normalized data).
    """
    import time, random
    random.seed(seed)
    np.random.seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]  # std deviations relative to normalized features

    results = {nl: {"accuracy": [], "f1": [], "roc_auc": [], "brier": []} for nl in noise_levels}

    X = X.astype(float)
    for nl in noise_levels:
        for run in range(n_runs):
            if nl == 0.0:
                X_noisy = X.copy()
            else:
                noise = np.random.normal(loc=0.0, scale=nl, size=X.shape)
                X_noisy = X + noise
            # model inference
            model.eval()
            X_t = torch.tensor(X_noisy, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = model(X_t).cpu().numpy().reshape(-1)
            metrics = _compute_metrics_from_probs(probs, y)
            for k, v in metrics.items():
                results[nl][k].append(v)

    # aggregate means and stds
    summary = {}
    for nl, vals in results.items():
        summary[nl] = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in vals.items()}

    if plot:
        # plot each metric vs noise level
        metrics_to_plot = ["accuracy", "f1", "roc_auc", "brier"]
        plt.figure(figsize=(10, 6))
        for m in metrics_to_plot:
            means = [summary[nl][m][0] for nl in noise_levels]
            stds = [summary[nl][m][1] for nl in noise_levels]
            plt.errorbar(noise_levels, means, yerr=stds, marker='o', label=m)
        plt.xlabel('Gaussian noise std (applied to normalized features)')
        plt.title('Noise sensitivity analysis')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {"noise_levels": noise_levels, "summary": summary}

