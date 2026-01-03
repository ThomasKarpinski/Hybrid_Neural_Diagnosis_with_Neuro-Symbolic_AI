import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any

from src.models.mlp_baseline import BaselineMLP
from src.hpo.fuzzy_controller import FuzzyController

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

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
    optimizer_name: str = "adam",
    device: str = None,
    save_dir: str = "experiments/best_models",
    verbose: bool = False,
    seed: int = 42,
    use_fuzzy: bool = False
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train the baseline MLP and return (model, history).
    Uses BCE loss and returns the trained best model on validation loss.
    Supports optimizers: adam, rmsprop, sgd, adagrad, adadelta, nadam, lion.
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

    # Calculate positive weight for imbalanced dataset
    # pos_weight = negative_samples / positive_samples
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    # Avoid division by zero
    if num_pos > 0:
        pos_weight_val = num_neg / num_pos
    else:
        pos_weight_val = 1.0
    
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight_val:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    opt_name = optimizer_name.lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, alpha=0.99)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif opt_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "nadam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt_name == "lion":
        optimizer = Lion(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        # Fallback to Adam if unknown
        print(f"Warning: Optimizer {optimizer_name} not found, defaulting to Adam.")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    fuzzy_ctrl = None
    if use_fuzzy:
        fuzzy_ctrl = FuzzyController()

    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f"baseline_mlp_{opt_name}.pth")

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
        
        # Check for NaN
        if np.isnan(avg_train_loss):
            if verbose:
                print(f"Epoch {epoch}: Train loss is NaN. Stopping early.")
            break

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

        if np.isnan(avg_val_loss):
             if verbose:
                print(f"Epoch {epoch}: Val loss is NaN. Stopping early.")
             break

        # Fuzzy Update
        if fuzzy_ctrl:
            prev_loss = history["val_loss"][-2] if len(history["val_loss"]) > 1 else None
            factor = fuzzy_ctrl.compute_update(avg_val_loss, prev_loss)
            
            # Update LR
            if factor != 1.0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * factor

        # save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_path)

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch {epoch}/{epochs} — train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")

    total_time = time.time() - start_time
    history["train_time"] = total_time

    # load best model
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print("Warning: No best model saved (likely NaN or divergence). Returning last model.")

    return model, history