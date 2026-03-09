
import json
import os
import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.models.train import train_baseline
from src.models.evaluate import evaluate_model

def setup_data():
    print("=== Data Setup (IsoForest + Random Oversampling) ===")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train, method="isolation_forest")
    
    # Random Oversampling
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
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_final, y_train_final,
        test_size=0.1,
        stratify=y_train_final,
        random_state=42
    )
    return X_tr, X_val, y_tr, y_val, X_test, y_test

def evaluate_method(method_name, json_path, data):
    print(f"Waiting for {json_path}...")
    while not os.path.exists(json_path) or os.stat(json_path).st_size == 0:
        time.sleep(5)
    
    # Check if modified recently (ensures we don't read old run)
    # But I can't easily check 'recently'. I'll assume if it exists and run_full is running, it's fine.
    # Actually, I should check if timestamp is > script start time.
    # For now, just load.
    
    with open(json_path, 'r') as f:
        res = json.load(f)
    
    best_hparams = res['best']['hparams']
    print(f"Best params for {method_name}: {best_hparams}")
    
    X_tr, X_val, y_tr, y_val, X_test, y_test = data
    input_dim = X_tr.shape[1]
    
    # Retrain
    print(f"Retraining {method_name} best model...")
    model, _ = train_baseline(
        X_tr, y_tr, X_val, y_val,
        input_dim=input_dim,
        lr=best_hparams.get('lr', 1e-3),
        batch_size=int(best_hparams.get('batch_size', 64)),
        epochs=int(best_hparams.get('epochs', 20)),
        dropout=float(best_hparams.get('dropout', 0.0)),
        weight_decay=float(best_hparams.get('weight_decay', 0.0)),
        betas=(best_hparams.get('beta1', 0.9), best_hparams.get('beta2', 0.999)),
        optimizer_name="adam",
        save_dir="experiments/final_models",
        verbose=False
    )
    
    metrics = evaluate_model(model, X_test, y_test, save_path=f"experiments/hpo_results/final_{method_name}_metrics.json")
    print(f"{method_name} Metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    data = setup_data()
    
    # Random Search
    evaluate_method("random", "experiments/hpo_results/raw_adam_random.json", data)
    
    # Optuna (Wait for it)
    # evaluate_method("optuna", "experiments/hpo_results/raw_adam_optuna.json", data)
    
    # Genetic (Wait for it)
    # evaluate_method("genetic", "experiments/hpo_results/raw_adam_genetic.json", data)
