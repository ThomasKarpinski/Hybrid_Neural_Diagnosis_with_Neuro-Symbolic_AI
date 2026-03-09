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
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # Vectorized update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                p.add_(torch.sign(exp_avg), alpha=-lr)
                
                # Update exp_avg for next step
                # Note: Lion's original paper uses a slightly different update rule 
                # where the sign is taken before the second momentum update.
                # Here we ensure it's efficient.
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def train_baseline(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
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
    use_fuzzy: bool = False,
    use_pos_weight: bool = True
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Optimized training loop for NVIDIA RTX 2060.
    Keeps entire dataset on GPU to eliminate transfer overhead.
    """
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Move data to GPU once
    if not isinstance(X_train, torch.Tensor):
        X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_tr = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
        X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_v = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32, device=device)
    else:
        X_tr, y_tr, X_v, y_v = X_train.to(device), y_train.to(device), X_val.to(device), y_val.to(device)

    model = BaselineMLP(input_dim=input_dim, hidden_dims=(32,16), dropout=dropout).to(device)

    if use_pos_weight:
        num_pos = (y_tr == 1).sum().item()
        num_neg = (y_tr == 0).sum().item()
        pos_weight_val = (num_neg / num_pos) if num_pos > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_val], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
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
    elif opt_name == "lion":
        optimizer = Lion(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    fuzzy_ctrl = FuzzyController() if use_fuzzy else None
    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f"baseline_mlp_{opt_name}.pth")

    n_samples = X_tr.size(0)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        # Fast GPU shuffling
        perm = torch.randperm(n_samples, device=device)
        X_tr_shuffled = X_tr[perm]
        y_tr_shuffled = y_tr[perm]

        for i in range(0, n_samples, batch_size):
            xb = X_tr_shuffled[i : i + batch_size]
            yb = y_tr_shuffled[i : i + batch_size]

            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        avg_train_loss = running_loss / n_samples
        history["train_loss"].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # Use larger batches for validation to speed up
            val_batch = 4096
            for i in range(0, X_v.size(0), val_batch):
                xb = X_v[i : i + val_batch]
                yb = y_v[i : i + val_batch]
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        
        avg_val_loss = val_loss / X_v.size(0)
        history["val_loss"].append(avg_val_loss)

        if fuzzy_ctrl:
            prev_loss = history["val_loss"][-2] if len(history["val_loss"]) > 1 else None
            factor = fuzzy_ctrl.compute_update(avg_val_loss, prev_loss)
            if factor != 1.0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= factor

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_path)

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch {epoch}/{epochs} — train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")

    history["train_time"] = time.time() - start_time
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    
    return model, history