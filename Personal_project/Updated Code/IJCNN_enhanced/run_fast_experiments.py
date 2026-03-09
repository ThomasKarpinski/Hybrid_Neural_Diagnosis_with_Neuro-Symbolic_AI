
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.utils import resample

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.models.train import train_baseline
from src.models.evaluate import evaluate_model

def run_fast_experiments():
    print("=== FAST EXPERIMENTS: Loading Data ===")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    
    # Subsample for speed (e.g., 20k total samples before split)
    # We do this AFTER load but BEFORE outlier detection/oversampling to keep pipeline logic similar
    # But wait, original pipeline removes outliers first. Let's stick to that but on a subset.
    
    print(f"Original Train Size: {len(X_train)}")
    
    # Subsample to max 20k for speed
    if len(X_train) > 20000:
        indices = np.random.choice(len(X_train), 20000, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"Subsampled Train Size: {len(X_train)}")

    print("=== Outlier Detection (ISO) ===")
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train, method="isolation_forest")
    print(f"Cleaned Train Size: {len(X_train_clean)}")

    print("=== Oversampling ===")
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
    X_train_balanced = np.vstack((X_train_neg, X_pos_resampled))
    y_train_balanced = np.hstack((y_train_neg, y_pos_resampled))
    print(f"Balanced Train Size: {len(y_train_balanced)}")

    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_balanced, y_train_balanced,
        test_size=0.1,
        stratify=y_train_balanced,
        random_state=42
    )

    optimizers = ["adam", "sgd", "rmsprop"]
    results = {}

    for opt in optimizers:
        print(f"\n>>> Running Experiment: MLP with {opt.upper()} <<<")
        model, history = train_baseline(
            X_tr, y_tr, X_val, y_val,
            input_dim=X_tr.shape[1],
            lr=1e-3,
            batch_size=64,
            epochs=10, # Reduced epochs for speed
            optimizer_name=opt,
            save_dir="experiments/fast_models",
            verbose=True
        )
        
        # Evaluate
        print(f"Evaluating {opt}...")
        # We manually calc metrics to ensure we catch Recall/AUC clearly
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            logits = model(X_test_tensor)
            probs = torch.sigmoid(logits).numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        auc_val = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        
        results[opt] = {
            "Accuracy": acc,
            "Recall": rec,
            "AUC": auc_val,
            "F1": f1
        }
        print(f"Result {opt}: AUC={auc_val:.4f}, Recall={rec:.4f}")

    print("\n" + "="*40)
    print("     FAST EXPERIMENT RESULTS SUMMARY     ")
    print("="*40)
    print(f"{ 'Optimizer':<10} { 'Accuracy':<10} { 'Recall':<10} { 'AUC':<10} { 'F1':<10}")
    print("-" * 50)
    for opt, res in results.items():
        print(f"{opt.upper():<10} {res['Accuracy']:.4f}     {res['Recall']:.4f}     {res['AUC']:.4f}     {res['F1']:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    run_fast_experiments()
