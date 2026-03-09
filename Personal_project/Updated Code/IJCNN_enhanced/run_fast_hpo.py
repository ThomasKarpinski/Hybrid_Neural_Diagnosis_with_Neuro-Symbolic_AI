import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.hpo.grid_random_search import run_random_search
from src.hpo.bayesian_opt import run_optuna
from src.hpo.evolutionary_hpo import run_genetic

def run_fast_hpo():
    print("=== FAST HPO: Loading Data ===")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    
    # Subsample for speed
    if len(X_train) > 10000:
        indices = np.random.choice(len(X_train), 10000, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"Subsampled Train Size: {len(X_train)}")

    print("=== Outlier Detection & Oversampling ===")
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train, method="isolation_forest")
    
    X_train_pos = X_train_clean[y_train_clean == 1]
    X_train_neg = X_train_clean[y_train_clean == 0]
    y_train_pos = y_train_clean[y_train_clean == 1]
    y_train_neg = y_train_clean[y_train_clean == 0]
    
    X_pos_resampled, y_pos_resampled = resample(
        X_train_pos, y_train_pos,
        replace=True,
        n_samples=len(y_train_neg),
        random_state=42
    )
    X_train_final = np.vstack((X_train_neg, X_pos_resampled))
    y_train_final = np.hstack((y_train_neg, y_pos_resampled))
    
    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_final, y_train_final,
        test_size=0.1,
        stratify=y_train_final,
        random_state=42
    )
    
    input_dim = X_tr.shape[1]
    
    # Modified Search Space for Speed
    search_space = {
        "lr": ("loguniform", 1e-4, 1e-2),
        "batch_size": [64, 128],
        "weight_decay": ("loguniform", 1e-6, 1e-3),
        "dropout": ("uniform", 0.0, 0.3),
        "epochs": ("uniform", 5, 8), # Very few epochs
        "beta1": ("uniform", 0.9, 0.99),
        "beta2": ("uniform", 0.99, 0.999),
    }
    
    print("\n>>> 1. Random Search (Fast) <<< ")
    rs_results = run_random_search(
        X_tr, y_tr, X_val, y_val,
        input_dim,
        search_space,
        n_iter=3, # Only 3 iterations
        seed=42
    )
    
    print("\n>>> 2. Optuna (Fast) <<< ")
    # Optuna needs its own internal Epochs definition often, 
    # but our run_optuna likely uses the same training loop logic.
    # We'll trust it handles the parameters or defaults.
    # Actually run_optuna defines its own search space internally usually.
    # Let's run it and see.
    opt_results = run_optuna(
        X_tr, y_tr, X_val, y_val,
        input_dim,
        n_trials=3, # Only 3 trials
        seed=42
    )
    
    print("\n>>> 3. Genetic Algorithm (Fast) <<< ")
    gen_results = run_genetic(
        X_tr, y_tr, X_val, y_val,
        input_dim,
        pop_size=4,   # Tiny population
        generations=2, # Only 2 gens
        seed=42
    )
    
    print("\n=== FAST HPO COMPLETE ===")

if __name__ == "__main__":
    run_fast_hpo()
