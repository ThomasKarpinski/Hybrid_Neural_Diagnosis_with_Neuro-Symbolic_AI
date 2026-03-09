import time
from typing import Any, Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from src.models.train import train_baseline
from src.utils.inference import predict_proba

def train_and_eval_on_val(hparams: Dict[str, Any], X_train, y_train, X_val, y_val, input_dim:int, seed: int = 42):
    """
    Train with hparams and return validation ROC-AUC and training time.
    hparams keys:
      - lr, batch_size, epochs, dropout, weight_decay, beta1, beta2
      - optimizer_name (optional, default="adam")
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
        optimizer_name=hparams.get("optimizer_name", "adam"),
        save_dir=hparams.get("save_dir", "experiments/best_models"),
        verbose=False,
        seed=seed,
        use_fuzzy=hparams.get("use_fuzzy", False)
    )
    train_time = history.get("train_time", time.time() - start)

    # compute validation probabilities and ROC-AUC
    probs = predict_proba(model, X_val)
    try:
        roc = float(roc_auc_score(y_val.reshape(-1).astype(int), probs))
    except Exception:
        roc = float("nan")

    return {"roc_auc": roc, "train_time": train_time, "hparams": hparams}