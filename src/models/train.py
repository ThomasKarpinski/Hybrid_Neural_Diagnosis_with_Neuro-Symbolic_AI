import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any

from src.models.mlp_baseline import BaselineMLP

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int = 21,
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 50,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999),
    device: str = None,
    save_dir: str = "experiments/best_models",
    verbose: bool = False,
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train the baseline MLP and return (model, history).
    Uses BCE loss and returns the trained best model on validation loss.
    """
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_v, y_v)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BaselineMLP(input_dim=input_dim, hidden_dims=(32,16), dropout=dropout).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "baseline_mlp_hpo.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
        history["val_loss"].append(avg_val_loss)

        # save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_path)

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch {epoch}/{epochs} — train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")

    total_time = time.time() - start_time
    history["train_time"] = total_time

    # load best model
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, history

