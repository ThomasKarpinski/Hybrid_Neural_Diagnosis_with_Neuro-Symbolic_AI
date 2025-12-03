import time
import json
import numpy as np
import random
import os
from typing import Dict, Any

from src.hpo.common import train_and_eval_on_val

def _init_population(pop_size, param_bounds):
    # param_bounds: dict name -> (low, high, type) type in {"float", "int", "cat"}
    pop = []
    for _ in range(pop_size):
        ind = {}
        for k, (low, high, typ, choices) in param_bounds.items():
            if typ == "float":
                ind[k] = random.uniform(low, high)
            elif typ == "int":
                ind[k] = random.randint(low, high)
            elif typ == "cat":
                ind[k] = random.choice(choices)
        pop.append(ind)
    return pop

def _crossover(a, b):
    # uniform crossover
    child = {}
    for k in a.keys():
        child[k] = a[k] if random.random() < 0.5 else b[k]
    return child

def _mutate(ind, param_bounds, mut_rate=0.1):
    for k, (low, high, typ, choices) in param_bounds.items():
        if random.random() < mut_rate:
            if typ == "float":
                ind[k] = np.clip(ind[k] + random.gauss(0, (high-low)*0.1), low, high)
            elif typ == "int":
                ind[k] = int(np.clip(ind[k] + random.randint(-2, 2), low, high))
            elif typ == "cat":
                ind[k] = random.choice(choices)
    return ind

def run_genetic(
    X_train, y_train, X_val, y_val,
    input_dim: int,
    param_bounds: Dict[str, Any] = None,
    pop_size: int = 12,
    generations: int = 6,
    mut_rate: float = 0.2,
    seed: int = 42,
    save_path: str = "experiments/hpo_results/genetic.json"
):
    random.seed(seed)
    np.random.seed(seed)
    if param_bounds is None:
        # default bounds
        param_bounds = {
            "lr": (1e-5, 1e-1, "float", None),
            "batch_size": (16, 128, "int", None),
            "weight_decay": (1e-7, 1e-2, "float", None),
            "dropout": (0.0, 0.5, "float", None),
            "epochs": (10, 60, "int", None),
            "beta1": (0.8, 0.99, "float", None),
            "beta2": (0.9, 0.9999, "float", None),
        }

    start_time = time.time()
    population = _init_population(pop_size, param_bounds)
    history = []
    best = {"roc_auc": -1.0}

    for gen in range(1, generations+1):
        print(f"GA Generation {gen}/{generations} — evaluating {len(population)} individuals")
        scored = []
        for i, ind in enumerate(population):
            # ensure proper types
            hparams = {
                "lr": float(ind["lr"]),
                "batch_size": int(ind["batch_size"]),
                "weight_decay": float(ind["weight_decay"]),
                "dropout": float(ind["dropout"]),
                "epochs": int(ind["epochs"]),
                "beta1": float(ind["beta1"]),
                "beta2": float(ind["beta2"]),
                "save_dir": "experiments/best_models/genetic"
            }
            res = train_and_eval_on_val(hparams, X_train, y_train, X_val, y_val, input_dim, seed=seed + gen * pop_size + i)
            scored.append((ind, res["roc_auc"], res["train_time"], hparams))
            if res["roc_auc"] is not None and not np.isnan(res["roc_auc"]) and res["roc_auc"] > best["roc_auc"]:
                best = {"roc_auc": res["roc_auc"], "hparams": hparams}
            print(f"  ind roc_auc={res['roc_auc']:.4f} time={res['train_time']:.1f}s")

        # sort by fitness desc
        scored.sort(key=lambda x: x[1] if (x[1] is not None and not (x[1]!=x[1])) else -1.0, reverse=True)
        history.append(scored)
        # selection: top 50%
        survivors = [s[0] for s in scored[: max(2, len(scored)//2)]]

        # create new population
        new_pop = survivors.copy()
        while len(new_pop) < pop_size:
            a, b = random.sample(survivors, 2)
            child = _crossover(a, b)
            child = _mutate(child, param_bounds, mut_rate=mut_rate)
            new_pop.append(child)
        population = new_pop

    elapsed = time.time() - start_time
    
    # Flatten history for easier plotting
    # history is list of lists of tuples: (ind, score, time, hparams)
    all_results = []
    for gen_idx, generation in enumerate(history):
        for ind_idx, (ind_dict, score, t_time, hparams) in enumerate(generation):
            all_results.append({
                "generation": gen_idx + 1,
                "individual": ind_idx + 1,
                "roc_auc": score,
                "train_time": t_time,
                "hparams": hparams
            })

    summary = {
        "method": "genetic",
        "n_evals": pop_size * generations,
        "time_s": elapsed,
        "best": best,
        "history_len": len(history),
        "all_results": all_results
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary

