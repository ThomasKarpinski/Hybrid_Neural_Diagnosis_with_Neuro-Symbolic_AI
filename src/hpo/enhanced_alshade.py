import time
import json
import numpy as np
import random
import os
import math
from typing import Dict, Any, List, Tuple

from src.hpo.common import train_and_eval_on_val

# --- Helper Functions ---

def cauchy_rnd(loc, scale, size=None):
    """Generate Cauchy random numbers."""
    return loc + scale * np.tan(np.pi * (np.random.random(size) - 0.5))

def normalize_value(val, low, high, typ, choices=None):
    """Normalize value to [0, 1]."""
    if typ == "cat":
        # Categorical: map index to [0, 1]
        # If 4 choices, indices 0, 1, 2, 3.
        # 0 -> 0.0, 3 -> 1.0
        try:
            idx = choices.index(val)
        except ValueError:
            idx = 0 # Fallback
        if len(choices) <= 1:
            return 0.0
        return idx / (len(choices) - 1)
    else:
        # float or int
        if high == low:
            return 0.0
        return (val - low) / (high - low)

def denormalize_value(val, low, high, typ, choices=None):
    """Denormalize value from [0, 1] to original scale."""
    val = np.clip(val, 0.0, 1.0)
    if typ == "cat":
        idx = int(round(val * (len(choices) - 1)))
        return choices[idx]
    elif typ == "int":
        return int(round(low + val * (high - low)))
    else:
        # float
        return low + val * (high - low)

def denormalize_vector(vec, bounds_list):
    """Denormalize a vector [0,1]^D to parameter dictionary."""
    params = {}
    for i, (name, (low, high, typ, choices)) in enumerate(bounds_list):
        params[name] = denormalize_value(vec[i], low, high, typ, choices)
    return params

# --- Enhanced ALSHADE Class ---

class EnhancedALSHADE:
    def __init__(self, 
                 param_bounds: Dict[str, Any], 
                 pop_size: int, 
                 max_evals: int, 
                 seed: int = 42,
                 fixed_hparams: Dict[str, Any] = None):
        
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.fixed_hparams = fixed_hparams if fixed_hparams else {}
        
        self.bounds_list = [] # List of (name, bounds_tuple)
        for k, v in param_bounds.items():
            # Commonize format: (low, high, type, choices)
            # Input format might be (low, high, type) or (low, high, type, choices)
            if len(v) == 3:
                # (low, high, type)
                self.bounds_list.append((k, (v[0], v[1], v[2], None)))
            else:
                self.bounds_list.append((k, v))
        
        self.dim = len(self.bounds_list)
        self.pop_size = pop_size
        self.max_evals = max_evals
        
        # ALSHADE Parameters
        self.p = 0.11
        self.H = 6
        self.rarc = 2.6
        self.NP_min = 4
        self.NP_init = pop_size
        
        # Memory
        self.M_F = 0.5 * np.ones(self.H)
        self.M_CR = 0.5 * np.ones(self.H)
        self.M_F[self.H - 1] = 0.9
        self.M_CR[self.H - 1] = 0.9
        self.k_mem = 0
        
        # Archive
        self.archive = [] # List of arrays
        
        # Population (in normalized [0,1] space)
        self.X = np.random.rand(self.pop_size, self.dim)
        self.fitness = -np.inf * np.ones(self.pop_size) # AUC, so -inf is bad
        self.best_params = None
        self.best_score = -np.inf
        
        self.eval_count = 0
        
        self.history = [] # Log of all evaluations

    def _sort_population(self):
        # Sort descending (AUC)
        indices = np.argsort(self.fitness)[::-1]
        self.X = self.X[indices]
        self.fitness = self.fitness[indices]

    def run(self, X_train, y_train, X_val, y_val, input_dim):
        # 1. Evaluate Initial Population
        print(f"Evaluating initial population ({self.pop_size})...")
        for i in range(self.pop_size):
            self._evaluate_ind(i, X_train, y_train, X_val, y_val, input_dim)
            
        # Sort
        self._sort_population()
        
        gen = 1
        while self.eval_count < self.max_evals:
            print(f"[ALSHADE] Generation {gen}, Evals {self.eval_count}/{self.max_evals}, Best AUC: {self.best_score:.4f}")
            
            pop_size_curr = self.X.shape[0]
            
            # Generate CR and F
            r_idx = np.random.randint(0, self.H, pop_size_curr)
            
            # CR: Normal(M_CR, 0.1)
            CR = np.random.normal(self.M_CR[r_idx], 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            
            # F: Cauchy(M_F, 0.1)
            F = np.zeros(pop_size_curr)
            for i in range(pop_size_curr):
                while True:
                    f_val = cauchy_rnd(self.M_F[r_idx[i]], 0.1)
                    if f_val > 0:
                        F[i] = min(f_val, 1.0)
                        break
            
            # Mutation and Crossover
            U = np.zeros_like(self.X)
            
            # Archive Union
            if len(self.archive) > 0:
                A_arr = np.array(self.archive)
                XA = np.vstack((self.X, A_arr))
            else:
                XA = self.X
            
            # p-best count
            p_num = max(2, int(round(self.p * pop_size_curr)))
            
            successful_F = []
            successful_CR = []
            successful_diff = []
            
            fitness_trial = np.zeros(pop_size_curr)
            
            # Loop for Trial Vector Generation
            for i in range(pop_size_curr):
                pbest_idx = np.random.randint(0, p_num)
                x_pbest = self.X[pbest_idx]
                
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(0, pop_size_curr)
                x_r1 = self.X[r1]
                
                r2 = np.random.randint(0, len(XA))
                x_r2 = XA[r2]
                
                # Mutation: current-to-pbest/1
                v = self.X[i] + F[i] * (x_pbest - self.X[i]) + F[i] * (x_r1 - x_r2)
                v = np.clip(v, 0.0, 1.0)
                
                # Crossover
                j_rand = np.random.randint(0, self.dim)
                u = np.zeros(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR[i] or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = self.X[i, j]
                U[i] = u
                
            # Evaluate Trial Vectors
            for i in range(pop_size_curr):
                if self.eval_count >= self.max_evals:
                    break
                
                # Evaluate U[i]
                hparams = denormalize_vector(U[i], self.bounds_list)
                hparams["save_dir"] = "experiments/best_models/enhanced_alshade"
                if self.fixed_hparams:
                    hparams.update(self.fixed_hparams)
                
                # Seed: diverse seed
                seed_eval = self.seed + self.eval_count
                res = train_and_eval_on_val(hparams, X_train, y_train, X_val, y_val, input_dim, seed=seed_eval)
                
                score = res["roc_auc"]
                if score is None or np.isnan(score):
                    score = 0.0
                
                self.eval_count += 1
                self.history.append({
                    "eval_idx": self.eval_count,
                    "generation": gen,
                    "individual": i,
                    "roc_auc": score,
                    "train_time": res["train_time"],
                    "hparams": hparams
                })
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = hparams
                
                fitness_trial[i] = score
                
                # Selection
                if score >= self.fitness[i]:
                    d = np.linalg.norm(U[i] - self.X[i])
                    
                    successful_F.append(F[i])
                    successful_CR.append(CR[i])
                    successful_diff.append(d)
                    
                    self.archive.append(self.X[i].copy())
                    self.X[i] = U[i]
                    self.fitness[i] = score
            
            # Manage Archive Size
            max_archive_size = int(round(self.rarc * pop_size_curr))
            if len(self.archive) > max_archive_size:
                random.shuffle(self.archive)
                self.archive = self.archive[:max_archive_size]
            
            # Update Memory
            if len(successful_F) > 0:
                w = np.array(successful_diff)
                w_sum = np.sum(w)
                if w_sum == 0:
                    w = np.ones_like(w) / len(w)
                else:
                    w = w / w_sum
                
                scr_arr = np.array(successful_CR)
                sf_arr = np.array(successful_F)
                
                if np.max(scr_arr) == 0:
                     self.M_CR[self.k_mem] = -1
                else:
                    self.M_CR[self.k_mem] = np.sum(w * (scr_arr ** 2)) / (np.sum(w * scr_arr) + 1e-9)
                
                self.M_F[self.k_mem] = np.sum(w * (sf_arr ** 2)) / (np.sum(w * sf_arr) + 1e-9)
                
                self.k_mem = (self.k_mem + 1) % self.H

            # Sort again for next gen
            self._sort_population()
            
            # LPSR
            plan_pop_size = int(round(self.NP_init - (self.NP_init - self.NP_min) * self.eval_count / self.max_evals))
            plan_pop_size = max(self.NP_min, plan_pop_size)
            
            if plan_pop_size < pop_size_curr:
                self.X = self.X[:plan_pop_size]
                self.fitness = self.fitness[:plan_pop_size]
                
                max_archive_size_new = int(round(self.rarc * plan_pop_size))
                if len(self.archive) > max_archive_size_new:
                    random.shuffle(self.archive)
                    self.archive = self.archive[:max_archive_size_new]
            
            gen += 1
        
        # Ensure final best params include fixed ones
        if self.best_params and self.fixed_hparams:
            self.best_params.update(self.fixed_hparams)

        return {
            "method": "enhanced_alshade",
            "n_evals": self.eval_count,
            "best": {"roc_auc": self.best_score, "hparams": self.best_params},
            "all_results": self.history,
            "fixed_hparams": self.fixed_hparams
        }

    def _evaluate_ind(self, i, X_train, y_train, X_val, y_val, input_dim):
        if self.eval_count >= self.max_evals:
            return

        hparams = denormalize_vector(self.X[i], self.bounds_list)
        hparams["save_dir"] = "experiments/best_models/enhanced_alshade"
        if self.fixed_hparams:
            hparams.update(self.fixed_hparams)
        
        res = train_and_eval_on_val(hparams, X_train, y_train, X_val, y_val, input_dim, seed=self.seed + self.eval_count)
        score = res["roc_auc"]
        if score is None or np.isnan(score):
            score = 0.0
        
        self.fitness[i] = score
        self.eval_count += 1
        
        self.history.append({
            "eval_idx": self.eval_count,
            "generation": 0,
            "individual": i,
            "roc_auc": score,
            "train_time": res["train_time"],
            "hparams": hparams
        })
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = hparams

# --- Wrapper Function ---

def run_enhanced_alshade(
    X_train, y_train, X_val, y_val,
    input_dim: int,
    pop_size: int = 20,
    generations: int = 10, 
    seed: int = 42,
    save_path: str = "experiments/hpo_results/enhanced_alshade.json",
    fixed_hparams: Dict[str, Any] = None
):
    # Default bounds
    bounds = {
        "lr": (1e-5, 1e-1, "float"),
        "batch_size": (0, 3, "cat", [16, 32, 64, 128]),
        "weight_decay": (1e-7, 1e-2, "float"),
        "dropout": (0.0, 0.5, "float"),
        "epochs": (10, 60, "int"),
        "beta1": (0.8, 0.99, "float"),
        "beta2": (0.9, 0.9999, "float"),
    }
    
    # Remove fixed params from bounds
    if fixed_hparams:
        for k in fixed_hparams.keys():
            if k in bounds:
                del bounds[k]
    
    # Calculate max_evals roughly
    max_evals = pop_size * generations
    
    start_time = time.time()
    optimizer = EnhancedALSHADE(bounds, pop_size, max_evals, seed, fixed_hparams=fixed_hparams)
    results = optimizer.run(X_train, y_train, X_val, y_val, input_dim)
    end_time = time.time()
    
    results["time_s"] = end_time - start_time
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
        
    return results