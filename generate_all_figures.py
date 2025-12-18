
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from src.data.load_data import load_raw_cdc, prepare_data
from src.data.outlier_detection import remove_outliers, plot_outlier_distributions
from src.utils.visualization import plot_class_distribution, pairplot_features
from src.models.mlp_baseline import BaselineMLP
from src.models.train import train_baseline
from src.models.calibration import reliability_diagram
from src.robustness.noise_sensitivity import evaluate_noise_sensitivity
from src.robustness.borderline_analysis import analyze_borderline_feature

def generate_all_figures():
    """
    This script generates and saves all the figures for the project.
    """
    print("--- Generating all figures ---")

    # Create a directory for the figures if it doesn't exist
    os.makedirs("figures", exist_ok=True)

    # --- 1. Data Loading and Preparation ---
    print("\n--- 1. Loading and preparing data ---")
    X_raw, y_raw = load_raw_cdc()
    # feature_names is also returned by prepare_data now
    X_train_orig, X_test_orig, y_train_orig, y_test_orig, scaler, feature_names_prep = prepare_data()
    
    # Use the feature names from prepare_data if X_raw wasn't available, but we have X_raw
    feature_names = X_raw.columns.tolist()

    # --- 2. Data-related plots ---
    print("\n--- 2. Generating data-related plots ---")

    # Outlier plots
    X_train_clean, y_train_clean = remove_outliers(X_train_orig, y_train_orig)
    plot_outlier_distributions(X_train_orig, feature_names=feature_names)
    plt.savefig("figures/outliers_before.png")
    plt.close()
    plot_outlier_distributions(X_train_clean, feature_names=feature_names)
    plt.savefig("figures/outliers_after.png")
    plt.close()
    print("Saved outlier plots.")

    # Class distribution
    plot_class_distribution(y_train_clean, save_path="figures/class_distribution.png")
    print("Saved class distribution plot.")

    # Pairplot (on a sample to avoid memory issues)
    print("Generating pairplot (this may take a while)...")
    pairplot_features(X_train_clean, y_train_clean, save_path="figures/pairplot.png")
    print("Saved pairplot.")

    # --- 3. Model Training (Skipped - Loading existing) ---
    print("\n--- 3. Loading baseline model ---")
    
    # Define model (match parameters from training)
    input_dim = X_train_clean.shape[1]
    model = BaselineMLP(input_dim)
    
    model_path = "experiments/best_models/baseline_mlp_hpo.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Training a quick dummy model for plots.")
        # Train a quick dummy model if file missing
        X_tr, X_val, y_tr, y_val = torch.utils.data.random_split(
            torch.utils.data.TensorDataset(torch.from_numpy(X_train_clean).float(), torch.from_numpy(y_train_clean).float()),
            [int(len(X_train_clean)*0.9), len(X_train_clean) - int(len(X_train_clean)*0.9)]
        )
        model, history = train_baseline(
            np.array([item[0] for item in X_tr]), np.array([item[1] for item in X_tr]),
            np.array([item[0] for item in X_val]), np.array([item[1] for item in X_val]),
            input_dim=input_dim,
            epochs=5 # Short training
        )
    
    print("Model ready.")

    # --- 4. Model-related plots ---
    print("\n--- 4. Generating model-related plots ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Calibration plot
    calib_results = reliability_diagram(model, X_test_orig, y_test_orig, plot=True, device=device)
    plt.savefig("figures/reliability_diagram.png")
    plt.close()
    print("Saved reliability diagram.")

    # Noise sensitivity plot
    noise_results = evaluate_noise_sensitivity(model, X_test_orig, y_test_orig, plot=True, device=device)
    plt.savefig("figures/noise_sensitivity.png")
    plt.close()
    print("Saved noise sensitivity plot.")

    # Borderline analysis plots
    X_test_unscaled = scaler.inverse_transform(X_test_orig)
    
    # BMI
    bmi_idx = feature_names.index("BMI")
    analyze_borderline_feature(
        model, X_test_unscaled, y_test_orig,
        feature_index=bmi_idx, feature_name="BMI", low=24, high=26,
        feature_names=feature_names, plot=True, device=device
    )
    plt.savefig("figures/borderline_bmi.png")
    plt.close()
    print("Saved BMI borderline analysis plot.")

    # Age
    age_idx = feature_names.index("Age")
    analyze_borderline_feature(
        model, X_test_unscaled, y_test_orig,
        feature_index=age_idx, feature_name="Age", low=8, high=10,
        feature_names=feature_names, plot=True, device=device
    )
    plt.savefig("figures/borderline_age.png")
    plt.close()
    print("Saved Age borderline analysis plot.")

    print("\n--- All figures generated successfully in the 'figures' directory ---")

if __name__ == "__main__":
    generate_all_figures()
