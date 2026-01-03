import json
import time
import itertools
import random
import numpy as np
from typing import Dict, Any

from src.hpo.common import train_and_eval_on_val
import os

def _sample_from_space(space, mode="random"):
    """
    If mode=="random", space values are:
      - for continuous: ("loguniform", low, high) or ("uniform", low, high)
      - for categorical: list of choices
    For simplicity, caller passes a dict with values to sample from.
    """
    sample = {}
    for k, v in space.items():
        if isinstance(v, tuple) and v[0] in ("loguniform", "uniform"):
            typ = v[0]
            low, high = v[1], v[2]
            if typ == "loguniform":
                val = 10 ** random.uniform(np.log10(low), np.log10(high))
            else:
                val = random.uniform(low, high)
            # if batch_size or epochs, round
            if k in ("batch_size", "epochs"):
                val = int(round(val))
            sample[k] = val
        else:
            # categorical
            sample[k] = random.choice(v)
    return sample

def run_random_search(
    X_train, y_train, X_val, y_val,
    input_dim: int,
    search_space: Dict[str, Any],
    n_iter: int = 20,
    seed: int = 42,
    save_path: str = "experiments/hpo_results/random_search.json",
    fixed_hparams: Dict[str, Any] = None
):
    random.seed(seed)
    results = []
    start = time.time()
    best = {"roc_auc": -1.0}
    for i in range(n_iter):
        hparams = _sample_from_space(search_space, mode="random")
        if fixed_hparams:
            hparams.update(fixed_hparams)
            
        res = train_and_eval_on_val(hparams, X_train, y_train, X_val, y_val, input_dim, seed=seed+i)
        res["eval_index"] = i+1
        results.append(res)
        if res["roc_auc"] is not None and not np.isnan(res["roc_auc"]) and res["roc_auc"] > best["roc_auc"]:
            best = res
        print(f"[Random {i+1}/{n_iter}] roc_auc={res['roc_auc']:.4f} time={res['train_time']:.1f}s hparams={hparams}")

    elapsed = time.time() - start
    summary = {
        "method": "random_search",
        "n_evals": n_iter,
        "time_s": elapsed,
        "best": best,
        "all_results": results,
        "fixed_hparams": fixed_hparams
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, default=lambda x: x if isinstance(x, (int, float, str, dict, list, type(None))) else str(x), indent=2)
    return summary

def run_grid_search(
    X_train, y_train, X_val, y_val,
    input_dim: int,
    grid_space: Dict[str, list],
    save_path: str = "experiments/hpo_results/grid_search.json",
    fixed_hparams: Dict[str, Any] = None
):
    # cartesian product
    keys = list(grid_space.keys())
    combos = list(itertools.product(*(grid_space[k] for k in keys)))
    results = []
    start = time.time()
    best = {"roc_auc": -1.0}
    for i, combo in enumerate(combos):
        hparams = {}
        for k, v in zip(keys, combo):
            hparams[k] = v
        
        if fixed_hparams:
            hparams.update(fixed_hparams)
            
        res = train_and_eval_on_val(hparams, X_train, y_train, X_val, y_val, input_dim, seed=i)
        res["eval_index"] = i+1
        results.append(res)
        if res["roc_auc"] is not None and not np.isnan(res["roc_auc"]) and res["roc_auc"] > best["roc_auc"]:
            best = res
        print(f"[Grid {i+1}/{len(combos)}] roc_auc={res['roc_auc']:.4f} time={res['train_time']:.1f}s hparams={hparams}")

    elapsed = time.time() - start
    summary = {
        "method": "grid_search",
        "n_evals": len(combos),
        "time_s": elapsed,
        "best": best,
        "all_results": results,
        "fixed_hparams": fixed_hparams
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, default=lambda x: x if isinstance(x, (int, float, str, dict, list, type(None))) else str(x), indent=2)
    return summary