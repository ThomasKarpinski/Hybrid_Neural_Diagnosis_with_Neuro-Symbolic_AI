import os
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.decomposition import PCA

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.models.autoencoder import train_autoencoder, get_embeddings
from src.models.train import train_baseline
from src.models.evaluate import evaluate_model

# HPO Runners
from src.hpo.grid_random_search import run_random_search
from src.hpo.bayesian_opt import run_optuna
from src.hpo.evolutionary_hpo import run_genetic
from src.hpo.enhanced_alshade import run_enhanced_alshade

def setup_all_data(device="cuda"):
    print("=== FINAL DATA SETUP (ISO + OVERSAMPLE + SUBSAMPLE) ===")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    
    # 1. Isolation Forest
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train, method="isolation_forest")
    
    # 2. Random Oversampling
    X_train_pos = X_train_clean[y_train_clean == 1]
    X_train_neg = X_train_clean[y_train_clean == 0]
    y_train_pos = y_train_clean[y_train_clean == 1]
    y_train_neg = y_train_clean[y_train_clean == 0]
    
    X_pos_resampled, y_pos_resampled = resample(
        X_train_pos, y_train_pos, replace=True, n_samples=len(y_train_neg), random_state=42
    )
    X_bal = np.vstack((X_train_neg, X_pos_resampled))
    y_bal = np.hstack((y_train_neg, y_pos_resampled))
    
    # 3. SUBSAMPLE TO 50,000 for speed (Ensuring balance)
    TOTAL_SAMPLES = 50000
    X_final, y_final = resample(X_bal, y_bal, n_samples=TOTAL_SAMPLES, stratify=y_bal, random_state=42)
    print(f"Final training pool: {len(y_final)} samples. Class dist: {np.bincount(y_final.astype(int))}")
    
    # Train/Val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_final, y_final, test_size=0.1, stratify=y_final, random_state=42
    )
    
    input_dim_raw = X_tr.shape[1]
    
    # Move to Device
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    # 4. Prepare Representations
    datasets = {}
    
    # A. Raw
    datasets["raw"] = (X_tr_t, X_val_t, y_tr_t, y_val_t, input_dim_raw)
    
    # B. PCA
    print("Computing PCA...")
    pca = PCA(n_components=0.95)
    X_tr_pca = pca.fit_transform(X_tr)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    datasets["pca"] = (
        torch.tensor(X_tr_pca, dtype=torch.float32, device=device),
        torch.tensor(X_val_pca, dtype=torch.float32, device=device),
        y_tr_t, y_val_t, pca.n_components_
    )
    X_test_pca_t = torch.tensor(X_test_pca, dtype=torch.float32, device=device)
    
    # C. Autoencoder
    print("Computing Autoencoder embeddings...")
    ae_model, _, _ = train_autoencoder(X_tr, input_dim_raw, encoding_dim=10, epochs=10, verbose=False)
    X_tr_ae = get_embeddings(ae_model, X_tr)
    X_val_ae = get_embeddings(ae_model, X_val)
    X_test_ae = get_embeddings(ae_model, X_test)
    
    datasets["ae"] = (
        torch.tensor(X_tr_ae, dtype=torch.float32, device=device),
        torch.tensor(X_val_ae, dtype=torch.float32, device=device),
        y_tr_t, y_val_t, 10
    )
    X_test_ae_t = torch.tensor(X_test_ae, dtype=torch.float32, device=device)
    
    return datasets, (X_test_t, X_test_pca_t, X_test_ae_t, y_test_t)

def get_best_metrics(best_hparams, X_tr, y_tr, X_val, y_val, X_test, y_test, optimizer_name, input_dim, device="cuda"):
    print(f"      Evaluating best {optimizer_name} on Test Set...")
    model, _ = train_baseline(
        X_tr, y_tr, X_val, y_val,
        input_dim=input_dim,
        lr=best_hparams.get('lr', 1e-3),
        batch_size=int(best_hparams.get('batch_size', 64)),
        epochs=15, # Fast epochs
        dropout=float(best_hparams.get('dropout', 0.0)),
        weight_decay=float(best_hparams.get('weight_decay', 0.0)),
        betas=(best_hparams.get('beta1', 0.9), best_hparams.get('beta2', 0.999)),
        optimizer_name=optimizer_name,
        device=device,
        verbose=False
    )
    # y_test might be tensor, evaluate_model expects numpy for sklearn but we updated predict_proba
    y_test_np = y_test.cpu().numpy().flatten().astype(int) if hasattr(y_test, "cpu") else y_test
    return evaluate_model(model, X_test, y_test_np, save_path=None)

def run_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets, test_data = setup_all_data(device=device)
    X_test_raw, X_test_pca, X_test_ae, y_test = test_data
    
    reps = ["raw", "pca", "ae"]
    opts = ["rmsprop", "adam", "sgd", "adagrad", "adadelta", "lion"]
    
    # HPO Budget
    N_RANDOM = 10
    N_OPTUNA = 15
    N_EVO_POP = 8
    N_EVO_GEN = 3 # 24 trials
    
    for rep in reps:
        X_tr, X_val, y_tr, y_val, in_dim = datasets[rep]
        X_te = X_test_raw if rep == "raw" else (X_test_pca if rep == "pca" else X_test_ae)
        
        print(f"\n\n################ REPRESENTATION: {rep.upper()} ################")
        
        for opt in opts:
            print(f"\n--- OPTIMIZER: {opt.upper()} ---")
            fixed = {"optimizer_name": opt}
            prefix = f"experiments/hpo_results/{rep}_{opt}"
            
            # 1. Random
            print(f"   Running Random Search...")
            res = run_random_search(X_tr, y_tr, X_val, y_val, in_dim, 
                                    {"lr": ("loguniform", 1e-4, 1e-2), "epochs": [15]},
                                    n_iter=N_RANDOM, fixed_hparams=fixed, save_path=f"{prefix}_random.json", device=device)
            m = get_best_metrics(res['best']['hparams'], X_tr, y_tr, X_val, y_val, X_te, y_test, opt, in_dim, device=device)
            with open(f"{prefix}_random_metrics.json", "w") as f: json.dump(m, f, indent=2)
            
            # 2. Optuna
            print(f"   Running Optuna...")
            res = run_optuna(X_tr, y_tr, X_val, y_val, in_dim, n_trials=N_OPTUNA, fixed_hparams=fixed, save_path=f"{prefix}_optuna.json", device=device)
            m = get_best_metrics(res['best']['hparams'], X_tr, y_tr, X_val, y_val, X_te, y_test, opt, in_dim, device=device)
            with open(f"{prefix}_optuna_metrics.json", "w") as f: json.dump(m, f, indent=2)
            
            # 3. Genetic
            print(f"   Running Genetic...")
            res = run_genetic(X_tr, y_tr, X_val, y_val, in_dim, pop_size=N_EVO_POP, generations=N_EVO_GEN, fixed_hparams=fixed, save_path=f"{prefix}_genetic.json", device=device)
            m = get_best_metrics(res['best']['hparams'], X_tr, y_tr, X_val, y_val, X_te, y_test, opt, in_dim, device=device)
            with open(f"{prefix}_genetic_metrics.json", "w") as f: json.dump(m, f, indent=2)
            
            # 4. ALSHADE
            print(f"   Running ALSHADE...")
            al_fixed = fixed.copy(); al_fixed["use_fuzzy"] = True
            res = run_enhanced_alshade(X_tr, y_tr, X_val, y_val, in_dim, pop_size=N_EVO_POP, generations=N_EVO_GEN, fixed_hparams=al_fixed, save_path=f"{prefix}_alshade.json", device=device)
            m = get_best_metrics(res['best']['hparams'], X_tr, y_tr, X_val, y_val, X_te, y_test, opt, in_dim, device=device)
            with open(f"{prefix}_alshade_metrics.json", "w") as f: json.dump(m, f, indent=2)

if __name__ == "__main__":
    run_all()
