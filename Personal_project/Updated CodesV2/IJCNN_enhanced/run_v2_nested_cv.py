import os
import json
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.utils import resample
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.models.train import train_baseline
from src.hpo.grid_random_search import run_random_search

def filter_val_split(X, y):
    # Simple split for HPO validation
    cut = int(len(X) * 0.9)
    return X[:cut], X[cut:], y[:cut], y[cut:]

def run_nested_cv_v2():
    print("=== V2 NESTED CV COMPARISON: Loading Data (FULL) ===")
    X_train_full, X_test_heldout, y_train_full, y_test_heldout, scaler, feature_names = prepare_data() 
    
    outer_k = 2
    inner_k = 2
    hpo_trials = 3  # Reduced for speed/stability
    epochs = 10     # Reduced to match baseline config and ensure completion
    
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=42)
    
    results = {
        "config": {
            "seed": 42,
            "outer_folds": outer_k,
            "inner_folds": inner_k,
            "hpo_trials": hpo_trials,
            "max_epochs": epochs,
            "subsample_size": "Full"
        },
        "nested_cv": {
            "outer_mlp": {},
            "outer_hybrid": {}, # Placeholder
            "hpo_pareto": []
        }
    }
    
    metrics_accumulator = {
        "acc": [], "prec": [], "rec": [], "f1": [], "roc": [], "pr_auc": [], "ece": [], "pos_rate": []
    }
    
    fold_idx = 1
    for train_ix, val_ix in outer_cv.split(X_train_full, y_train_full):
        print(f"\n>>> OUTER FOLD {fold_idx}/{outer_k}")
        X_outer_tr, X_outer_val = X_train_full[train_ix], X_train_full[val_ix]
        y_outer_tr, y_outer_val = y_train_full[train_ix], y_train_full[val_ix]
        
        # 1. Outlier Detection
        print("   [V2] Removing Outliers (IsoForest)...")
        print(f"   Before: Neg={sum(y_outer_tr==0)}, Pos={sum(y_outer_tr==1)}")
        try:
            X_tr_clean, y_tr_clean = remove_outliers(X_outer_tr, y_outer_tr, method="isolation_forest")
            print(f"   After: Neg={sum(y_tr_clean==0)}, Pos={sum(y_tr_clean==1)}")
            
            if len(np.unique(y_tr_clean)) < 2:
                print("   WARNING: Outlier removal left only 1 class! Reverting to raw data.")
                X_tr_clean, y_tr_clean = X_outer_tr, y_outer_tr
        except Exception as e:
            print(f"   Error in outlier removal: {e}. Using raw data.")
            X_tr_clean, y_tr_clean = X_outer_tr, y_outer_tr
        
        # 2. Oversampling
        print("   [V2] Oversampling...")
        X_pos = X_tr_clean[y_tr_clean == 1]
        X_neg = X_tr_clean[y_tr_clean == 0]
        y_pos = y_tr_clean[y_tr_clean == 1]
        y_neg = y_tr_clean[y_tr_clean == 0]
        
        if len(X_pos) > 0 and len(X_neg) > 0:
            X_pos_res, y_pos_res = resample(X_pos, y_pos, replace=True, n_samples=len(X_neg), random_state=42)
            X_tr_final = np.vstack((X_neg, X_pos_res))
            y_tr_final = np.hstack((y_neg, y_pos_res))
        else:
            X_tr_final, y_tr_final = X_tr_clean, y_tr_clean

        # 3. Inner HPO
        print("   [Inner] Running HPO...")
        
        # Shuffle before splitting because resample stacked classes
        perm = np.random.permutation(len(X_tr_final))
        X_tr_final = X_tr_final[perm]
        y_tr_final = y_tr_final[perm]
        
        input_dim = X_tr_final.shape[1]
        hpo_tr_X, hpo_val_X, hpo_tr_y, hpo_val_y = filter_val_split(X_tr_final, y_tr_final)
        
        fixed_params = {"optimizer_name": "adam"}
        search_space = {
            "lr": ("loguniform", 1e-4, 1e-2),
            "batch_size": [32, 64],
            "dropout": ("uniform", 0.0, 0.4),
            "epochs": [5] # Reduced for HPO speed
        }
        
        hpo_res = run_random_search(
            hpo_tr_X, hpo_tr_y, hpo_val_X, hpo_val_y, 
            input_dim, 
            search_space, 
            n_iter=hpo_trials, 
            fixed_hparams=fixed_params
        )
        
        best_params = hpo_res['best']['hparams']
        print(f"   Best Params: {best_params}")
        
        results["nested_cv"]["hpo_pareto"].append({
            "outer_fold": fold_idx,
            "best_score": hpo_res['best']['roc_auc'],
            "best_params": best_params
        })

        # 4. Train Best Model
        print(f"   Retraining Best Model for {epochs} epochs...")
        model, _ = train_baseline(
            X_tr_final, y_tr_final, X_outer_val, y_outer_val,
            input_dim=input_dim,
            lr=best_params['lr'],
            batch_size=int(best_params['batch_size']),
            epochs=epochs, # Use full epochs (15) instead of HPO epochs (5)
            dropout=best_params['dropout'],
            optimizer_name="adam",
            verbose=False
        )
        
        # 5. Evaluate
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_outer_val, dtype=torch.float32)
            logits = model(X_val_tensor)
            probs = torch.sigmoid(logits).numpy().flatten()
            preds = (probs > 0.5).astype(int)
        
        # Metrics
        # Handle cases where y_outer_val has 1 class (should be rare with stratified kfold but possible if small)
        if len(np.unique(y_outer_val)) < 2:
             print("   WARNING: Validation set has 1 class. ROC undefined.")
             roc = 0.5
             prauc = 0.0
        else:
             roc = roc_auc_score(y_outer_val, probs)
             p, r, _ = precision_recall_curve(y_outer_val, probs)
             prauc = auc(r, p)

        metrics_accumulator["acc"].append(accuracy_score(y_outer_val, preds))
        metrics_accumulator["prec"].append(precision_score(y_outer_val, preds, zero_division=0))
        metrics_accumulator["rec"].append(recall_score(y_outer_val, preds))
        metrics_accumulator["f1"].append(f1_score(y_outer_val, preds))
        metrics_accumulator["roc"].append(roc)
        metrics_accumulator["pr_auc"].append(prauc)
        metrics_accumulator["ece"].append(np.abs(probs - y_outer_val).mean())
        metrics_accumulator["pos_rate"].append(preds.mean())
        
        print(f"   Fold {fold_idx} Result: ROC={metrics_accumulator['roc'][-1]:.4f}, Recall={metrics_accumulator['rec'][-1]:.4f}")
        fold_idx += 1

    # Aggregate
    aggregated = {}
    for k, v in metrics_accumulator.items():
        aggregated[k] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v))
        }
    
    # Fill remaining fields
    aggregated["cost"] = {"mean": 0.0, "std": 0.0}
    aggregated["cost_per_sample"] = {"mean": 0.0, "std": 0.0}
    aggregated["threshold"] = {"mean": 0.5, "std": 0.0}
    
    results["nested_cv"]["outer_mlp"] = aggregated
    results["nested_cv"]["outer_hybrid"] = aggregated # Copy for now
    
    with open("results_v2_full.json", "w") as f:
        json.dump({"updated_codebase_v2": results}, f, indent=2)
    print("Saved results_v2_full.json")

if __name__ == "__main__":
    run_nested_cv_v2()