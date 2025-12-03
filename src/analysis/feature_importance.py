import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.load_data import prepare_data
from src.models.mlp_baseline import BaselineMLP

def load_best_model(model_path, input_dim):
    """Load the trained PyTorch model."""
    model = BaselineMLP(input_dim=input_dim)
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None

def calculate_permutation_importance(model, X_val, y_val, feature_names, n_repeats=5):
    """
    Calculate permutation importance:
    1. Measure baseline accuracy/AUC.
    2. Shuffle one feature at a time.
    3. Measure drop in performance.
    """
    print(f"Calculating Permutation Importance over {len(feature_names)} features...")
    
    # Convert to tensor once
    X_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_true = y_val
    
    # Baseline Score (Accuracy for simplicity, or Mean Abs Error since prob output)
    with torch.no_grad():
        preds = model(X_tensor).numpy().flatten()
    
    # Using simple Mean Absolute Error as "Loss" (lower is better)
    # Importance = Increase in Error when shuffled
    baseline_error = np.mean(np.abs(y_true - preds))
    print(f"Baseline MAE: {baseline_error:.5f}")
    
    importances = {}
    
    for i, feat in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            # Shuffle column i
            X_shuffled = X_val.copy()
            np.random.shuffle(X_shuffled[:, i])
            
            X_shuffled_tensor = torch.tensor(X_shuffled, dtype=torch.float32)
            with torch.no_grad():
                p_shuffled = model(X_shuffled_tensor).numpy().flatten()
            
            shuffled_error = np.mean(np.abs(y_true - p_shuffled))
            scores.append(shuffled_error - baseline_error) # Higher diff = more important
            
        importances[feat] = np.mean(scores)
        # print(f"  {feat}: {importances[feat]:.5f}")

    return importances

def plot_feature_importance(importances, save_path="experiments/feature_importance.png"):
    # Sort
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    feats, scores = zip(*sorted_feats)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=list(scores), y=list(feats), palette="viridis")
    plt.title("Feature Importance (Permutation based on MAE)")
    plt.xlabel("Increase in Prediction Error (Importance)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Print top 5 textual
    print("\nTop 5 Important Features:")
    for f, s in sorted_feats[:5]:
        print(f"{f:<20}: {s:.5f}")

def run_analysis():
    # 1. Load Data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    input_dim = X_train.shape[1]
    
    # 2. Load Model (Optuna best or fallback)
    model_path = "experiments/best_models/optuna/best_model.pth"
    if not os.path.exists(model_path):
        model_path = "experiments/best_models/baseline_mlp_hpo.pth"
        
    model = load_best_model(model_path, input_dim)
    if not model:
        return

    # 3. Calculate
    # Use a subset of Test data for speed if needed, or full test
    subset_idx = np.random.choice(len(X_test), min(2000, len(X_test)), replace=False)
    X_sub = X_test[subset_idx]
    y_sub = y_test[subset_idx]
    
    imps = calculate_permutation_importance(model, X_sub, y_sub, feature_names)
    
    # 4. Plot
    plot_feature_importance(imps)

if __name__ == "__main__":
    run_analysis()
