import time
import json
import optuna
import os
from typing import Dict, Any

from src.hpo.common import train_and_eval_on_val

def run_optuna(
    X_train, y_train, X_val, y_val,
    input_dim: int,
    n_trials: int = 30,
    seed: int = 42,
    save_path: str = "experiments/hpo_results/optuna.json",
    fixed_hparams: Dict[str, Any] = None
):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    start = time.time()

    def objective(trial):
        # hyperparameter space
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        epochs = trial.suggest_int("epochs", 10, 80)
        beta1 = trial.suggest_float("beta1", 0.8, 0.99)
        beta2 = trial.suggest_float("beta2", 0.9, 0.9999)

        hparams = {
            "lr": lr, "batch_size": int(batch_size), "weight_decay": weight_decay,
            "dropout": dropout, "epochs": int(epochs), "beta1": float(beta1), "beta2": float(beta2),
            "save_dir": "experiments/best_models/optuna"
        }
        
        if fixed_hparams:
            hparams.update(fixed_hparams)

        res = train_and_eval_on_val(hparams, X_train, y_train, X_val, y_val, input_dim, seed=seed + trial.number)
        # report result and let optuna drive next trials
        return float(res["roc_auc"]) if (res["roc_auc"] is not None and not (res["roc_auc"]!=res["roc_auc"])) else 0.0

    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - start

    # Extract full history
    all_trials = []
    for t in study.trials:
        if t.value is not None:
            all_trials.append({
                "trial_number": t.number,
                "roc_auc": t.value,
                "hparams": t.params,
                "state": t.state.name
            })

    best_trial = study.best_trial
    best = {
        "roc_auc": best_trial.value,
        "hparams": best_trial.params
    }
    
    if fixed_hparams:
        best["hparams"].update(fixed_hparams)

    summary = {
        "method": "optuna", 
        "n_evals": n_trials, 
        "time_s": elapsed, 
        "best": best,
        "all_results": all_trials,
        "fixed_hparams": fixed_hparams
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary