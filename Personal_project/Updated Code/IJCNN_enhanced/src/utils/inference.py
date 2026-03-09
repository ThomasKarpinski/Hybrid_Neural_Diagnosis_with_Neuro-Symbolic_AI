"""Centralized inference helpers.

The MLP is trained with BCEWithLogitsLoss, therefore the model outputs *logits*.
All evaluation (ROC-AUC, thresholding, confusion matrices) must convert logits to
probabilities with a sigmoid.

Keeping this logic in one place prevents subtle inconsistencies between:
- baseline evaluation
- HPO evaluators (Optuna/GA/DE/PSO)
- plotting utilities
- paper table generation
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def get_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 4096,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Return raw logits as a 1D numpy array."""
    model.eval()
    device = device or get_device(model)

    X_t = torch.tensor(X, dtype=torch.float32)
    n = X_t.shape[0]
    outs = []
    for start in range(0, n, batch_size):
        xb = X_t[start : start + batch_size].to(device)
        logits = model(xb)
        outs.append(logits.detach().cpu().numpy().reshape(-1))
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def predict_proba(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 4096,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Return probabilities in [0,1] as a 1D numpy array."""
    logits = predict_logits(model, X, batch_size=batch_size, device=device)
    # stable sigmoid
    return 1.0 / (1.0 + np.exp(-logits))
