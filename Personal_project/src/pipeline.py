"""
Main pipeline for the Neuro-Symbolic Diabetes Diagnostic Project.
Handles:
 - Data loading & preprocessing
 - Outlier detection
 - Baseline MLP training & evaluation
 - Full HPO (Grid/Random, Bayesian, Evolutionary)
 - Classical AI Interpretability & Unsupervised Analysis
"""
# ===============================================================
# Imports
# ===============================================================

# --- Data ---
from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers

# --- Visualization ---
from src.utils.visualization import plot_class_distribution, pairplot_features, perform_unsupervised_analysis, plot_confusion_matrix

# --- Baseline Model ---
from src.models.train import train_baseline
from src.models.evaluate import evaluate_model

# --- HPO Methods ---
from src.hpo.grid_random_search import run_random_search, run_grid_search
from src.hpo.bayesian_opt import run_optuna
from src.hpo.evolutionary_hpo import run_genetic

# --- Interpretability ---
from src.interpretability.rules import apply_rules_dataframe
from src.interpretability.fuzzy import compute_p_fuzzy
from src.interpretability.bayesian_update import GaussianNaiveBayesLike

# --- System ---
import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split


# ===============================================================
# BASELINE PIPELINE
# ===============================================================

def run_pipeline():
    """Runs the full baseline experiment: data → outliers → MLP → evaluation."""

    print("=== Data loading & preprocessing ===")
    # Prepare data now includes feature_names
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()

    print("=== Outlier detection on training set ===")
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train, method="isolation_forest")
    print(f"Train samples: before={len(y_train)}, after={len(y_train_clean)}")

    print("=== Applying Random Oversampling to balance classes ===")
    from sklearn.utils import resample
    X_train_pos = X_train_clean[y_train_clean == 1]
    X_train_neg = X_train_clean[y_train_clean == 0]
    y_train_pos = y_train_clean[y_train_clean == 1]
    y_train_neg = y_train_clean[y_train_clean == 0]
    
    # Oversample minority class to match majority
    X_pos_resampled, y_pos_resampled = resample(
        X_train_pos, y_train_pos,
        replace=True,
        n_samples=len(y_train_neg),
        random_state=42
    )
    X_train_clean = np.vstack((X_train_neg, X_pos_resampled))
    y_train_clean = np.hstack((y_train_neg, y_pos_resampled))
    print(f"After Oversampling: {len(y_train_clean)} samples. Class dist: {np.bincount(y_train_clean.astype(int))}")

    # Optional visualizations
    plot_class_distribution(y_train_clean, save_path="class_distribution.png")
    pairplot_features(X_train_clean, y_train_clean, save_path="pairplot.png")

    # Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_clean, y_train_clean,
        test_size=0.1,
        stratify=y_train_clean,
        random_state=42
    )

    print("=== Training baseline MLP ===")
    os.makedirs("experiments/best_models", exist_ok=True)
    os.makedirs("experiments/hpo_results", exist_ok=True)

    model, history = train_baseline(
        X_tr, y_tr, X_val, y_val,
        input_dim=X_tr.shape[1],
        lr=1e-3,
        batch_size=64,
        epochs=50,
        save_dir="experiments/best_models",
    )

    print("=== Evaluating on held-out test set ===")
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        save_path="experiments/hpo_results/baseline_metrics.json"
    )

    # Plot Confusion Matrix
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_test_tensor)
        test_preds_probs = torch.sigmoid(logits).numpy().flatten()
    plot_confusion_matrix(y_test, test_preds_probs, save_path="experiments/confusion_matrix.png")

    print("\n=== Classical AI Demonstration ===")
    # Select 10 random samples
    indices = np.random.choice(len(X_test), 10, replace=False)
    X_sample_scaled = X_test[indices]
    y_sample = y_test[indices]
    
    # INVERSE TRANSFORM for Rules/Fuzzy (they expect raw values like Age=45, BMI=30)
    X_sample_raw = scaler.inverse_transform(X_sample_scaled)
    # Convert to DataFrame with feature names
    df_sample = pd.DataFrame(X_sample_raw, columns=feature_names)
    
    # 1. Rules
    rule_decisions = apply_rules_dataframe(df_sample) # returns list of dicts/None
    
    # 2. Fuzzy
    fuzzy_risks = compute_p_fuzzy(df_sample) # returns list of floats
    
    # 3. Bayes
    # Train Bayes on TRAINING data.
    # Using scaled data for Bayes since it learns means/vars from provided data.
    bayes_model = GaussianNaiveBayesLike(feature_names)
    bayes_model.fit(X_train, y_train, feature_names=feature_names)
    
    bayes_probs = bayes_model.predict_proba(X_sample_scaled, feature_names=feature_names)
    
    # MLP Predictions for these samples
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_sample_scaled, dtype=torch.float32)
        mlp_logits = model(X_tensor)
        mlp_preds = torch.sigmoid(mlp_logits).numpy()
    
    # Flatten if necessary (assuming mlp_preds is shape (N,1))
    mlp_probs = mlp_preds.flatten() 
    
    # Print Table
    print(f"{ 'ID':<5} {'True':<5} {'MLP':<6} {'Rule':<25} {'Fuzzy':<6} {'Bayes':<6}")
    print("-" * 65)
    for i in range(10):
        r_res = rule_decisions[i]
        r_str = r_res['decision'] if r_res else "None"
        print(f"{indices[i]:<5} {y_sample[i]:<5} {mlp_probs[i]:.4f} {r_str:<25} {fuzzy_risks[i]:.4f} {bayes_probs[i]:.4f}")

    print("\n=== Unsupervised Learning Analysis ===")
    # Run on subset of training data (scaled)
    subset_idx = np.random.choice(len(X_train), min(len(X_train), 5000), replace=False)
    perform_unsupervised_analysis(
        X_train[subset_idx], 
        y_train[subset_idx], 
        save_prefix="experiments/unsupervised"
    )

    print("Pipeline finished.")
    return model, metrics, (X_train_clean, X_test, y_train_clean, y_test), scaler


# ===============================================================
# FULL HPO PIPELINE
# ===============================================================

def run_all_hpo():
    """Runs Grid/Random Search, Bayesian Optimization, and Genetic Algorithm HPO."""

    print("=== Preparing data for HPO ===")
    # Updated unpacking
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train, method="isolation_forest")

    # Apply Random Oversampling
    from sklearn.utils import resample
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
    X_train_clean = np.vstack((X_train_neg, X_pos_resampled))
    y_train_clean = np.hstack((y_train_neg, y_pos_resampled))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_clean, y_train_clean,
        test_size=0.1,
        stratify=y_train_clean,
        random_state=42
    )

    input_dim = X_tr.shape[1]

    # -------------------------------
    # Define unified search space
    # -------------------------------
    search_space = {
        "lr": ("loguniform", 1e-5, 1e-2),
        "batch_size": [32, 64, 128],
        "weight_decay": ("loguniform", 1e-7, 1e-3),
        "dropout": ("uniform", 0.0, 0.4),
        "epochs": ("uniform", 10, 50),
        "beta1": ("uniform", 0.85, 0.95),
        "beta2": ("uniform", 0.9, 0.999),
    }

    print(">>> Running random search (n_iter=12)")
    rs = run_random_search(
        X_tr, y_tr, X_val, y_val,
        input_dim,
        search_space,
        n_iter=12,
        seed=42
    )

    print(">>> Running Optuna (n_trials=12)")
    opt = run_optuna(
        X_tr, y_tr, X_val, y_val,
        input_dim,
        n_trials=12,
        seed=42
    )

    print(">>> Running Genetic Algorithm (pop=8, gen=4)")
    gen = run_genetic(
        X_tr, y_tr, X_val, y_val,
        input_dim,
        pop_size=8,
        generations=4,
        seed=42
    )

    print("=== HPO finished ===")
    print("Results saved under experiments/hpo_results/")
    return rs, opt, gen


# ===============================================================
# MAIN ENTRY
# ===============================================================

if __name__ == "__main__":

    # 1. Run Baseline + Classical AI Demo + Unsupervised Analysis

    run_pipeline()

    

    # 2. Run Full Hyperparameter Optimization (Random, Optuna, Genetic)

    print("\n" + "="*60)

    print("STARTING HYPERPARAMETER OPTIMIZATION")

    print("="*60 + "\n")

    run_all_hpo()
