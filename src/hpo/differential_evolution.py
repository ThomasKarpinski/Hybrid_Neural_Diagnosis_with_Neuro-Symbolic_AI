import time
import json
import numpy as np
import random
import os
from typing import Dict, Any

from src.hpo.common import train_and_eval_on_val

def ensure_bounds(val, low, high, typ):
    if typ == "int":
        return int(np.clip(val, low, high))
    elif typ == "cat":
        # For categorical, val is an index float, clip to [0, len-1]
        idx = int(np.clip(round(val), 0, len(high)-1))
        return high[idx] # high holds the list of choices
    else:
        return np.clip(val, low, high)

def run_differential_evolution(
    X_train, y_train, X_val, y_val,
    input_dim: int,
    pop_size: int = 10,
    generations: int = 5,
    F: float = 0.8, # Differential weight
    CR: float = 0.7, # Crossover probability
    seed: int = 42,
    save_path: str = "experiments/hpo_results/differential_evolution.json"
):
    """
    Differential Evolution (DE/rand/1/bin) for hyperparameter optimization.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Define search space bounds (low, high, type)
    # For categorical (batch_size), we map index 0..N
    bounds = {
        "lr": (1e-5, 1e-1, "float"),
        "batch_size": (0, 3, "cat", [16, 32, 64, 128]), # 4 options
        "weight_decay": (1e-7, 1e-2, "float"),
        "dropout": (0.0, 0.5, "float"),
        "epochs": (10, 60, "int"),
        "beta1": (0.8, 0.99, "float"),
        "beta2": (0.9, 0.9999, "float"),
    }

    # Initialize Population
    population = [] # List of vectors (dicts of values)
    for _ in range(pop_size):
        ind = {}
        for k, v in bounds.items():
            if v[2] == "cat":
                # Random choice from list
                ind[k] = random.choice(v[3])
            elif v[2] == "int":
                ind[k] = random.randint(v[0], v[1])
            else:
                ind[k] = random.uniform(v[0], v[1])
        population.append(ind)

    best = {"roc_auc": -1.0}
    history = [] # Store all evaluations
    
    start_time = time.time()

    for gen in range(1, generations + 1):
        print(f"[DE] Generation {gen}/{generations}")
        new_population = []
        
        for i in range(pop_size):
            target = population[i]
            
            # 1. Mutation: Pick 3 distinct others
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = random.sample(idxs, 3)
            a, b, c = population[a_idx], population[b_idx], population[c_idx]
            
            mutant = {}
            for k, v in bounds.items():
                # Treat categorical as index for math? Simplified:
                # For DE, we strictly need numerical vectors. 
                # Let's handle categorical by randomly picking one of the parents or random if CR hits.
                # Or better: convert categorical to index 0..3 for the math, then round back.
                
                val_a = a[k]
                val_b = b[k]
                val_c = c[k]
                
                if v[2] == "cat":
                    choices = v[3]
                    # Map to index
                    idx_a = choices.index(val_a)
                    idx_b = choices.index(val_b)
                    idx_c = choices.index(val_c)
                    
                    diff = F * (idx_b - idx_c)
                    mut_val = idx_a + diff
                    # Will resolve to value later
                else:
                    mut_val = val_a + F * (val_b - val_c)
                
                mutant[k] = mut_val

            # 2. Crossover (Binomial)
            trial = {}
            for k, v in bounds.items():
                if random.random() < CR:
                    # Use mutant value (clipped/resolved)
                    if v[2] == "cat":
                        # Fix: Pass the list of choices (v[3]) as 'high'
                        trial[k] = ensure_bounds(mutant[k], 0, v[3], "cat")
                    else:
                        trial[k] = ensure_bounds(mutant[k], v[0], v[1], v[2])
                else:
                    # Use target (parent) value
                    trial[k] = target[k]
            
            # 3. Selection (Evaluate Trial vs Target)
            # To save time, if we haven't evaluated 'target' yet (gen 1), we must.
            # But standard DE keeps population scores. Let's eval trial.
            
            # Note: In standard DE, parent is already evaluated from prev gen.
            # We need to store scores.
            if gen == 1 and "score" not in target:
                # Eval initial population first? Or just eval pairwise?
                # Let's eval target first if needed.
                hparams_t = {**target, "save_dir": "experiments/best_models/de"}
                res_t = train_and_eval_on_val(hparams_t, X_train, y_train, X_val, y_val, input_dim, seed=seed+i)
                target["score"] = res_t["roc_auc"]
                # Log it
                history.append({"gen": 0, "ind": i, "roc_auc": res_t["roc_auc"], "hparams": target})
                if res_t["roc_auc"] > best["roc_auc"]:
                    best = {"roc_auc": res_t["roc_auc"], "hparams": target}

            # Eval Trial
            hparams_trial = {**trial, "save_dir": "experiments/best_models/de"}
            res_trial = train_and_eval_on_val(hparams_trial, X_train, y_train, X_val, y_val, input_dim, seed=seed+gen*pop_size+i)
            
            # Log Trial
            history.append({"gen": gen, "ind": i, "roc_auc": res_trial["roc_auc"], "hparams": trial})
            if res_trial["roc_auc"] > best["roc_auc"]:
                best = {"roc_auc": res_trial["roc_auc"], "hparams": trial}

            # Selection
            if res_trial["roc_auc"] >= target["score"]:
                trial["score"] = res_trial["roc_auc"]
                new_population.append(trial)
                print(f"  Ind {i}: Trial ({res_trial['roc_auc']:.4f}) >= Target ({target['score']:.4f}) -> Swap")
            else:
                new_population.append(target)
                print(f"  Ind {i}: Trial ({res_trial['roc_auc']:.4f}) < Target ({target['score']:.4f}) -> Keep")

        population = new_population

    elapsed = time.time() - start_time
    summary = {
        "method": "differential_evolution",
        "n_evals": len(history),
        "time_s": elapsed,
        "best": best,
        "all_results": history
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary
