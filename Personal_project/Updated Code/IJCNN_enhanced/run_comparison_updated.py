
import os
import sys
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score
from sklearn.utils import resample

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.models.train import train_baseline

def run_updated_experiments():
    print("=== UPDATED CODE COMPARISON: Loading Data ===")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    
    # Subsample for speed (10k)
    if len(X_train) > 10000:
        indices = np.random.choice(len(X_train), 10000, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"Subsampled Train Size: {len(X_train)}")

    print("=== Outlier Detection (ISO) ===")
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train, method="isolation_forest")
    
    # Oversampling logic from updated pipeline
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
    
    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_balanced, y_train_balanced,
        test_size=0.1,
        stratify=y_train_balanced,
        random_state=42
    )

    optimizers = ["adam", "sgd"]
    results = {}

    for opt in optimizers:
        print(f"\n>>> Running Updated Experiment: MLP with {opt.upper()} <<<")
        model, history = train_baseline(
            X_tr, y_tr, X_val, y_val,
            input_dim=X_tr.shape[1],
            lr=1e-3,
            batch_size=64,
            epochs=10, 
            optimizer_name=opt,
            save_dir="experiments/comparison_updated_models",
            verbose=True
        )
        
        # Evaluate
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
            "Representation": "Raw",
            "Optimizer": opt,
            "Hyperparameters": {"lr": 0.001, "epochs": 10, "batch_size": 64},
            "Metrics": {
                "Accuracy": acc,
                "Recall": rec,
                "AUC": auc_val,
                "F1": f1
            }
        }
        print(f"Result {opt}: AUC={auc_val:.4f}, Recall={rec:.4f}")

    with open("updated_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved updated_results.json")

if __name__ == "__main__":
    run_updated_experiments()
