import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from typing import Dict, Any

from src.models.train import train_baseline

def train_and_eval_on_val(hparams: Dict[str, Any], X_train, y_train, X_val, y_val, input_dim:int, seed: int = 42):
    """
    Train with hparams and return validation ROC-AUC and training time.
    Supports both numpy arrays and torch tensors.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure data is on device if it's already a tensor, otherwise train_baseline handles it
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
        use_fuzzy=hparams.get("use_fuzzy", False),
        device=device
    )
    train_time = history.get("train_time", time.time() - start)

    # compute val probs and roc_auc
    model.eval()
    
    # Fast evaluation on GPU
    if not isinstance(X_val, torch.Tensor):
        X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
    else:
        X_v = X_val.to(device)
        
    with torch.no_grad():
        logits = model(X_v)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        
    y_v_true = y_val.cpu().numpy().reshape(-1) if isinstance(y_val, torch.Tensor) else y_val.reshape(-1)
    
    try:
        roc = float(roc_auc_score(y_v_true.astype(int), probs))
    except Exception:
        roc = float("nan")

    return {"roc_auc": roc, "train_time": train_time, "hparams": hparams}