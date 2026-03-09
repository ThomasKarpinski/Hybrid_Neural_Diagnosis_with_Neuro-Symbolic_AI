import sys
import os
import numpy as np
import pandas as pd
import json
import time
from sklearn.model_selection import train_test_split

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.models.autoencoder import train_autoencoder, get_embeddings

# HPO Runners
from src.hpo.grid_random_search import run_random_search
from src.hpo.bayesian_opt import run_optuna
from src.hpo.evolutionary_hpo import run_genetic
from src.hpo.enhanced_alshade import run_enhanced_alshade

def get_ae_features(X_train, X_val, X_test, input_dim, encoding_dim=10):
    print(f"Training Autoencoder (dim={encoding_dim})...")
    model, _, _ = train_autoencoder(X_train, input_dim, encoding_dim=encoding_dim, epochs=20, verbose=True)
    
    X_train_ae = get_embeddings(model, X_train)
    X_val_ae = get_embeddings(model, X_val)
    X_test_ae = get_embeddings(model, X_test)
    return X_train_ae, X_val_ae, X_test_ae, encoding_dim

def run_missing_experiments(target_optimizer=None):
    # 1. Load and Preprocess Data
    print("Loading Data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train)
    
    # Train/Val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_clean, y_train_clean,
        test_size=0.1,
        stratify=y_train_clean,
        random_state=42
    )
    
    input_dim_raw = X_tr.shape[1]
    
    # 2. Prepare AE Representation
    X_tr_ae, X_val_ae, _, dim_ae = get_ae_features(X_tr, X_val, X_test, input_dim_raw, encoding_dim=10)
    
    # 3. Define Missing Experiment Grid
    rep_name = "ae"
    
    all_missing_optimizers = ["adagrad", "adadelta", "lion"]
    
    if target_optimizer:
        if target_optimizer not in all_missing_optimizers:
            print(f"Error: Target optimizer '{target_optimizer}' is not one of the missing optimizers ({', '.join(all_missing_optimizers)}).")
            return
        optimizers_to_run = [target_optimizer]
    else:
        optimizers_to_run = all_missing_optimizers
    
    # Reduced budget (same as original script)
    N_TRIALS_RANDOM = 5
    N_TRIALS_OPTUNA = 5
    POP_SIZE_EVO = 5
    GEN_EVO = 3
    
    X_t, X_v, y_t, y_v, in_dim = (X_tr_ae, X_val_ae, y_tr, y_val, dim_ae)

    print(f"\n\n>>> STARTING REPRESENTATION: {rep_name.upper()} (dim={in_dim})")
    
    for opt_name in optimizers_to_run:
        print(f"\n   >>> OPTIMIZER: {opt_name.upper()}")
        fixed_hparams = {"optimizer_name": opt_name}
        base_path = f"experiments/hpo_results/{rep_name}_{opt_name}"
        
        # 1. Random Search
        print("      Running Random Search...")
        run_random_search(
            X_t, y_t, X_v, y_v, in_dim,
            search_space={
                "lr": ("loguniform", 1e-4, 1e-1),
                "batch_size": [32, 64],
                "epochs": ("uniform", 10, 30)
            },
            n_iter=N_TRIALS_RANDOM,
            save_path=f"{base_path}_random.json",
            fixed_hparams=fixed_hparams
        )
        
        # 2. Optuna
        print("      Running Optuna...")
        run_optuna(
            X_t, y_t, X_v, y_v, in_dim,
            n_trials=N_TRIALS_OPTUNA,
            save_path=f"{base_path}_optuna.json",
            fixed_hparams=fixed_hparams
        )
        
        # 3. Genetic
        print("      Running Genetic Algorithm...")
        run_genetic(
            X_t, y_t, X_v, y_v, in_dim,
            pop_size=POP_SIZE_EVO, generations=GEN_EVO,
            save_path=f"{base_path}_genetic.json",
            fixed_hparams=fixed_hparams
        )
        
        # 4. Enhanced ALSHADE
        print("      Running Enhanced ALSHADE...")
        alshade_hparams = fixed_hparams.copy()
        alshade_hparams["use_fuzzy"] = True
        run_enhanced_alshade(
            X_t, y_t, X_v, y_v, in_dim,
            pop_size=POP_SIZE_EVO, generations=GEN_EVO,
            save_path=f"{base_path}_alshade.json",
            fixed_hparams=alshade_hparams
        )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        optimizer_arg = sys.argv[1]
        run_missing_experiments(target_optimizer=optimizer_arg)
    else:
        run_missing_experiments()

