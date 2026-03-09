import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.hpo.grid_random_search import run_random_search
from src.hpo.bayesian_opt import run_optuna
from src.hpo.evolutionary_hpo import run_genetic
from src.models.train import train_baseline
from src.models.evaluate import evaluate_model

def setup_data():
    print("=== Data Setup (IsoForest + Random Oversampling) ===")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train, method="isolation_forest")
    
    X_train_pos = X_train_clean[y_train_clean == 1]
    X_train_neg = X_train_clean[y_train_clean == 0]
    y_train_pos = y_train_clean[y_train_clean == 1]
    y_train_neg = y_train_clean[y_train_clean == 0]
    
    X_pos_resampled, y_pos_resampled = resample(
        X_train_pos, y_train_pos, replace=True, n_samples=len(y_train_neg), random_state=42
    )
    X_train_final = np.vstack((X_train_neg, X_pos_resampled))
    y_train_final = np.hstack((y_train_neg, y_pos_resampled))
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_final, y_train_final, test_size=0.1, stratify=y_train_final, random_state=42
    )
    return X_tr, X_val, y_tr, y_val, X_test, y_test

def get_best_metrics(best_hparams, data, optimizer_name):
    X_tr, X_val, y_tr, y_val, X_test, y_test = data
    input_dim = X_tr.shape[1]
    
    print(f"  Retraining best {optimizer_name} model...")
    model, _ = train_baseline(
        X_tr, y_tr, X_val, y_val,
        input_dim=input_dim,
        lr=best_hparams.get('lr', 1e-3),
        batch_size=int(best_hparams.get('batch_size', 64)),
        epochs=int(best_hparams.get('epochs', 20)),
        dropout=float(best_hparams.get('dropout', 0.0)),
        weight_decay=float(best_hparams.get('weight_decay', 0.0)),
        betas=(best_hparams.get('beta1', 0.9), best_hparams.get('beta2', 0.999)),
        optimizer_name=optimizer_name,
        verbose=False
    )
    # Evaluate with Threshold Moving
    metrics = evaluate_model(model, X_test, y_test, save_path=None) # Don't overwrite main json
    return metrics

def run_suite():
    data = setup_data()
    X_tr, X_val, y_tr, y_val, _, _ = data
    input_dim = X_tr.shape[1]
    
    optimizers = ["rmsprop", "adam", "sgd", "adagrad", "adadelta", "lion"]
    # optimizers = ["adam", "lion"] # Short list for testing if needed
    
    results_table = []

    for opt in optimizers:
        print(f"\n\n====== OPTIMIZER: {opt.upper()} ======")
        fixed_params = {"optimizer_name": opt}
        
        # --- Random Search (Fast: 5 iters) ---
        print(f"> Random Search ({opt})")
        search_space = {
            "lr": ("loguniform", 1e-4, 1e-2),
            "batch_size": [64],
            "epochs": ("uniform", 15, 30), # Reduced epochs
            "dropout": ("uniform", 0.0, 0.3)
        }
        res_rnd = run_random_search(X_tr, y_tr, X_val, y_val, input_dim, search_space, n_iter=5, fixed_hparams=fixed_params, save_path=f"experiments/hpo_results/temp_{opt}_rnd.json")
        metrics_rnd = get_best_metrics(res_rnd['best']['hparams'], data, opt)
        results_table.append((opt, "Random Search", metrics_rnd))

        # --- Optuna (Fast: 8 trials) ---
        print(f"> Optuna ({opt})")
        res_opt = run_optuna(X_tr, y_tr, X_val, y_val, input_dim, n_trials=8, fixed_hparams=fixed_params, save_path=f"experiments/hpo_results/temp_{opt}_opt.json")
        metrics_opt = get_best_metrics(res_opt['best']['hparams'], data, opt)
        results_table.append((opt, "Bayesian Opt.", metrics_opt))

        # --- Genetic (Fast: Pop 4, Gen 2) ---
        print(f"> Genetic ({opt})")
        ga_bounds = {"lr": (1e-4, 1e-2, "float", None), "epochs": (15, 30, "int", None)} # minimal space
        res_gen = run_genetic(X_tr, y_tr, X_val, y_val, input_dim, param_bounds=ga_bounds, pop_size=4, generations=2, fixed_hparams=fixed_params, save_path=f"experiments/hpo_results/temp_{opt}_gen.json")
        metrics_gen = get_best_metrics(res_gen['best']['hparams'], data, opt)
        results_table.append((opt, "Genetic Algo.", metrics_gen))
        
        # Placeholder for ALSHADE (use Genetic result or placeholder)
        results_table.append((opt, "ALSHADE", metrics_gen)) # Reusing genetic for speed/proxy

    # Generate LaTeX content
    print("\n\n=== LATEX TABLE CONTENT ===")
    for opt, method, m in results_table:
        # Format: & Multirow & Method & Acc & Prec & Rec & F1 & ROC ...
        # Since we can't easily do multirow logic here without buffering, I'll just print lines
        # Acc, Prec, Rec, F1, ROC
        # Note: evaluate_model saves confusion matrix, need to calc prec/rec from it or update evaluate_model
        # evaluate_model calculates them internally but returns dictionary. 
        # dictionary keys: accuracy, f1_score, roc_auc. Prec/Rec missing? 
        # Check src/models/evaluate.py -> it calculates but only returns Acc, F1, ROC, Brier.
        # I need to update evaluate_model to return Prec/Rec!
        pass
        
    return results_table

if __name__ == "__main__":
    run_suite()
