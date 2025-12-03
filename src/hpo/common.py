import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from typing import Dict, Any

from src.models.train import train_baseline

def train_and_eval_on_val(hparams: Dict[str, Any], X_train, y_train, X_val, y_val, input_dim:int, seed: int = 42):
    """
    Train with hparams and return validation ROC-AUC and training time.
    hparams keys:
      - lr, batch_size, epochs, dropout, weight_decay, beta1, beta2
    """
    start = time.time()
    model, history = train_baseline(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        lr=hparams.get("lr", 1e-3),
        batch_size=int(hparams.get("batch_size", 64)),
        epochs=int(hparams.get("epochs", 20)),
        dropout=float(hparams.get("dropout", 0.0)),
        weight_decay=float(hparams.get("weight_decay", 0.0)),
        betas=(hparams.get("beta1", 0.9), hparams.get("beta2", 0.999)),
        save_dir=hparams.get("save_dir", "experiments/best_models"),
        verbose=False,
        seed=seed,
    )
    train_time = history.get("train_time", time.time() - start)

    # compute val probs and roc_auc
    device = next(model.parameters()).device
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        probs = model(X_v).cpu().numpy().reshape(-1)
    try:
        roc = float(roc_auc_score(y_val.reshape(-1).astype(int), probs))
    except Exception:
        roc = float("nan")

    return {"roc_auc": roc, "train_time": train_time, "hparams": hparams}

