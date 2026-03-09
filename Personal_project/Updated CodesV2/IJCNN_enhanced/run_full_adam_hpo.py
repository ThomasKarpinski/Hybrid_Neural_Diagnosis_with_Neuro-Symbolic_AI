
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Import local modules
from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.hpo.grid_random_search import run_random_search
from src.hpo.bayesian_opt import run_optuna
from src.hpo.evolutionary_hpo import run_genetic

def setup_data():
    print("=== Data Setup (IsoForest + Random Oversampling) ===")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    
    # 1. Isolation Forest
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train, method="isolation_forest")
    
    # 2. Random Oversampling
    print(f"Before Oversampling: {np.bincount(y_train_clean.astype(int))}")
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
    print(f"After Oversampling: {np.bincount(y_train_final.astype(int))}")
    
    # Train/Val Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_final, y_train_final,
        test_size=0.1,
        stratify=y_train_final,
        random_state=42
    )
    
    return X_tr, X_val, y_tr, y_val

def run_adam_hpo():
    X_tr, X_val, y_tr, y_val = setup_data()
    input_dim = X_tr.shape[1]
    
    # Common fixed params
    fixed_params = {"optimizer_name": "adam"}
    
    # 1. Random Search
    print("\n>>> Running Random Search (Adam) <<<")
    search_space = {
        "lr": ("loguniform", 1e-4, 1e-2),
        "batch_size": [32, 64, 128],
        "weight_decay": ("loguniform", 1e-6, 1e-3),
        "dropout": ("uniform", 0.0, 0.4),
        "epochs": ("uniform", 20, 50),
        "beta1": ("uniform", 0.85, 0.95),
        "beta2": ("uniform", 0.9, 0.999),
    }
    run_random_search(
        X_tr, y_tr, X_val, y_val, input_dim,
        search_space,
        n_iter=10,
        save_path="experiments/hpo_results/raw_adam_random.json",
        fixed_hparams=fixed_params
    )
    
    # 2. Optuna
    print("\n>>> Running Optuna (Adam) <<<")
    run_optuna(
        X_tr, y_tr, X_val, y_val, input_dim,
        n_trials=15,
        save_path="experiments/hpo_results/raw_adam_optuna.json",
        fixed_hparams=fixed_params
    )
    
    # 3. Genetic Algorithm
    print("\n>>> Running Genetic Algorithm (Adam) <<<")
    ga_bounds = {
        "lr": (1e-4, 1e-2, "float", None),
        "batch_size": (32, 128, "int", None),
        "weight_decay": (1e-6, 1e-3, "float", None),
        "dropout": (0.0, 0.4, "float", None),
        "epochs": (20, 50, "int", None),
        "beta1": (0.85, 0.95, "float", None),
        "beta2": (0.9, 0.999, "float", None),
    }
    run_genetic(
        X_tr, y_tr, X_val, y_val, input_dim,
        param_bounds=ga_bounds,
        pop_size=6,
        generations=3,
        save_path="experiments/hpo_results/raw_adam_genetic.json",
        fixed_hparams=fixed_params
    )
    
    print("\n=== Adam HPO Suite Completed ===")

if __name__ == "__main__":
    run_adam_hpo()
