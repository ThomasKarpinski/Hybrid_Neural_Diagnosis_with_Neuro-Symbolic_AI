import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.models.train import train_baseline
from src.run_comprehensive_experiments import get_pca_features, get_ae_features # For dataset prep

# Configuration
RESULTS_DIR = "experiments/hpo_results"
FINAL_MODELS_DIR = "experiments/final_models"
FIGURES_DIR = "paper/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def get_best_hparams_for_rep(representation: str):
    """
    Loads HPO results from JSON files and finds the best hyperparameters for a given representation.
    """
    best_roc_auc = -1
    best_hparams = None
    best_optimizer = None

    optimizers = ["adam", "rmsprop", "sgd"]
    hpo_methods = ["random", "optuna", "genetic", "alshade"] # Assuming these are suffixes

    for opt in optimizers:
        for method in hpo_methods:
            fname = f"{representation}_{opt}_{method}.json"
            path = os.path.join(RESULTS_DIR, fname)
            data = load_json(path)
            
            if data and "best" in data and "roc_auc" in data["best"]:
                current_roc_auc = data["best"]["roc_auc"]
                if current_roc_auc > best_roc_auc:
                    best_roc_auc = current_roc_auc
                    best_hparams = data["best"]["hparams"]
                    best_optimizer = opt
    
    if best_hparams:
        best_hparams["optimizer_name"] = best_optimizer
    
    return best_hparams, best_roc_auc


def generate_training_loss_plot():
    print("Generating Training/Validation Loss Convergence Plot...")
    
    # 1. Prepare Data (Replicate from generate_paper_outputs.py)
    X_train_full, X_test_full, y_train_full, y_test_full, scaler, feature_names = prepare_data()
    X_train_clean, y_train_clean = remove_outliers(X_train_full, y_train_full)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_clean, y_train_clean, 
        test_size=0.1, stratify=y_train_clean, random_state=42
    )
    
    input_dim_raw = X_tr.shape[1]
    
    # 2. Get Best Hyperparameters for 'raw' representation
    # We prioritize 'raw' as it's typically shown for baseline training.
    best_hparams, best_roc = get_best_hparams_for_rep("raw")

    if not best_hparams:
        print("Could not retrieve best hyperparameters for 'raw' representation. Skipping plot.")
        return

    print(f"Retrieved best HPs for raw: ROC={best_roc:.4f}, Opt={best_hparams.get('optimizer_name', 'N/A')}, LR={best_hparams.get('lr', 'N/A')}, Epochs={best_hparams.get('epochs', 'N/A')}")

    # 3. Retrain model with best HPs to get history
    # Use the retrieved best epochs, ensure verbose is True to get console output if needed
    model, history = train_baseline(
        X_tr, y_tr, X_val, y_val,
        input_dim=input_dim_raw,
        lr=best_hparams["lr"],
        batch_size=best_hparams["batch_size"],
        epochs=best_hparams["epochs"],
        dropout=best_hparams.get("dropout", 0),
        weight_decay=best_hparams.get("weight_decay", 0),
        optimizer_name=best_hparams.get("optimizer_name", "adam"),
        save_dir=os.path.join(FINAL_MODELS_DIR, "temp_for_plot"), # Save to a temp location
        verbose=False # Set to True for debug output during retraining
    )
    
    # 4. Plot Training History
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Binary Cross-Entropy)")
    plt.title(f"Training and Validation Loss Convergence (Best Raw MLP - {best_hparams.get('optimizer_name', 'Adam')})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(FIGURES_DIR, "training_loss_convergence.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    generate_training_loss_plot()
