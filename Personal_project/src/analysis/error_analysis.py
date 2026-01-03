import torch
import pandas as pd
import numpy as np
import os
import json
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.load_data import prepare_data
from src.models.mlp_baseline import BaselineMLP
from src.interpretability.rules import apply_rules_dataframe
from src.interpretability.fuzzy import compute_p_fuzzy
from src.interpretability.bayesian_update import GaussianNaiveBayesLike

def load_best_model(model_path, input_dim):
    """Load the trained PyTorch model."""
    model = BaselineMLP(input_dim=input_dim)
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Loaded model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None

def analyze_errors():
    print("=== Detailed Neuro-Symbolic Error Analysis ===")
    
    # 1. Load Data (consistent with pipeline)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    input_dim = X_train.shape[1]

    # 2. Load Best Model
    model_path = "experiments/best_models/optuna/best_model.pth"
    if not os.path.exists(model_path):
        model_path = "experiments/best_models/baseline_mlp.pth"
        if not os.path.exists(model_path):
             model_path = "experiments/best_models/baseline_mlp_hpo.pth"

    model = load_best_model(model_path, input_dim)
    if model is None:
        return

    # 3. Generate ML Predictions
    print("Generating Neural Network predictions on Test Set...")
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_tensor).numpy().flatten()

    # 4. Generate Symbolic Predictions (Full Test Set)
    print("Generating Symbolic Expert predictions on Test Set...")
    X_test_raw = scaler.inverse_transform(X_test)
    df_test_raw = pd.DataFrame(X_test_raw, columns=feature_names)
    
    # Rules
    rule_outputs = apply_rules_dataframe(df_test_raw)
    rule_preds = []
    for res in rule_outputs:
        if res is None:
            rule_preds.append(0.5) # Uncertain
        elif res['decision'] == 'high':
            rule_preds.append(1.0)
        else: # low
            rule_preds.append(0.0)
    rule_preds = np.array(rule_preds)

    # Fuzzy
    fuzzy_preds = np.array(compute_p_fuzzy(df_test_raw))

    # Bayes
    bayes_model = GaussianNaiveBayesLike(feature_names)
    bayes_model.fit(X_train, y_train, feature_names=feature_names)
    bayes_preds = np.array(bayes_model.predict_proba(X_test, feature_names=feature_names))

    # 5. Comparative Analysis on Full Test Set
    print("\n=== Global Neuro-Symbolic Comparison ===")
    
    # Disagreement Metrics
    # Threshold MLP at 0.5 for hard class
    mlp_class = (preds > 0.5).astype(int)
    
    # Threshold Symbolic layers
    rule_class = (rule_preds > 0.5).astype(int) # 0.5 is uncertain, strict rules might need different logic
    fuzzy_class = (fuzzy_preds > 0.5).astype(int)
    bayes_class = (bayes_preds > 0.5).astype(int)
    
    def calc_disagreement(name, other_class):
        dis = np.mean(mlp_class != other_class)
        print(f"MLP vs {name} Disagreement Rate: {dis:.2%}")
        
    calc_disagreement("Rules", rule_class)
    calc_disagreement("Fuzzy", fuzzy_class)
    calc_disagreement("Bayes", bayes_class)
    
    # 6. "Safety Net" Analysis
    # Definition: MLP is WRONG (False Neg/Pos), but Expert is RIGHT (High/Low Risk correct).
    # Focus on False Negatives (MLP says Healthy, Patient is Diabetic) -> Expert says High Risk?
    
    print("\n=== Safety Net Analysis (False Negatives) ===")
    false_negatives_mask = (y_test == 1) & (mlp_class == 0)
    num_fn = np.sum(false_negatives_mask)
    print(f"Total MLP False Negatives: {num_fn}")
    
    if num_fn > 0:
        # Check if Expert caught them
        rule_catch = np.sum((rule_preds[false_negatives_mask] > 0.5))
        fuzzy_catch = np.sum((fuzzy_preds[false_negatives_mask] > 0.5))
        bayes_catch = np.sum((bayes_preds[false_negatives_mask] > 0.5))
        
        print(f"Rules caught: {rule_catch} ({rule_catch/num_fn:.2%})")
        print(f"Fuzzy caught: {fuzzy_catch} ({fuzzy_catch/num_fn:.2%})")
        print(f"Bayes caught: {bayes_catch} ({bayes_catch/num_fn:.2%})")
    
    print("\n=== Safety Net Analysis (False Positives) ===")
    false_positives_mask = (y_test == 0) & (mlp_class == 1)
    num_fp = np.sum(false_positives_mask)
    print(f"Total MLP False Positives: {num_fp}")
    
    if num_fp > 0:
        # Check if Expert corrected them (Predicted Low Risk)
        rule_catch_fp = np.sum((rule_preds[false_positives_mask] < 0.5))
        fuzzy_catch_fp = np.sum((fuzzy_preds[false_positives_mask] < 0.5))
        bayes_catch_fp = np.sum((bayes_preds[false_positives_mask] < 0.5))
        
        print(f"Rules corrected: {rule_catch_fp} ({rule_catch_fp/num_fp:.2%})")
        print(f"Fuzzy corrected: {fuzzy_catch_fp} ({fuzzy_catch_fp/num_fp:.2%})")
        print(f"Bayes corrected: {bayes_catch_fp} ({bayes_catch_fp/num_fp:.2%})")

    # 7. Save Summary to file for report
    with open("experiments/neuro_symbolic_stats.txt", "w") as f:
        f.write("MLP vs Rules Disagreement: {:.2%}\n".format(np.mean(mlp_class != rule_class)))
        f.write("MLP vs Fuzzy Disagreement: {:.2%}\n".format(np.mean(mlp_class != fuzzy_class)))
        f.write("MLP vs Bayes Disagreement: {:.2%}\n".format(np.mean(mlp_class != bayes_class)))
        if num_fn > 0:
            f.write("SafetyNet_FN_Rules: {:.2%}\n".format(rule_catch/num_fn))
            f.write("SafetyNet_FN_Fuzzy: {:.2%}\n".format(fuzzy_catch/num_fn))
            f.write("SafetyNet_FN_Bayes: {:.2%}\n".format(bayes_catch/num_fn))

if __name__ == "__main__":
    analyze_errors()