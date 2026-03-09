
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from src.data.load_data import prepare_data
from src.models.mlp_baseline import BaselineMLP
from src.interpretability.rules import apply_rules_dataframe
from src.interpretability.fuzzy import compute_p_fuzzy
from src.interpretability.bayesian_update import GaussianNaiveBayesLike

MOE_DIR = "experiments/fast_models"
OPTIMIZERS = ["adam", "sgd", "rmsprop"]

def evaluate_fast_hybrid():
    print("=== Evaluating Hybrid Architecture on Fast Models ===")
    
    # 1. Load Data
    # We use the full test set for evaluation to be comparable
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    
    # Prepare non-MLP components
    print("Fitting Bayesian Module on Full Training Data...")
    bayes = GaussianNaiveBayesLike(feature_names)
    bayes.fit(X_train, y_train, feature_names)
    
    print("Pre-computing Fuzzy and Bayes probabilities for Test Set...")
    # Unscale for Rules/Fuzzy
    X_test_unscaled = scaler.inverse_transform(X_test)
    df_test = pd.DataFrame(X_test_unscaled, columns=feature_names)
    
    # Fuzzy
    p_fuzzy = np.array(compute_p_fuzzy(df_test))
    
    # Bayes
    p_bayes = bayes.predict_proba(X_test, feature_names)
    
    # Rules
    rule_decisions = apply_rules_dataframe(df_test)
    
    input_dim = X_train.shape[1]
    
    results = {}

    for opt in OPTIMIZERS:
        model_path = os.path.join(MOE_DIR, f"baseline_mlp_{opt}.pth")
        print(f"\n>>> Processing Model: {opt.upper()} <<<")
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue
            
        # Load MLP
        model = BaselineMLP(input_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # MLP Predictions
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_prob_mlp = model(X_tensor).numpy().flatten()
            
        # --- Hybrid Fusion ---
        # Simple Average Fusion
        y_prob_hybrid = (y_prob_mlp + p_fuzzy + p_bayes) / 3.0
        
        # Rules Override
        # If Rule says High -> boost prob to min 0.9
        # If Rule says Low -> suppress prob to max 0.1
        for i, res in enumerate(rule_decisions):
            if res:
                if res['decision'] == 'high':
                    y_prob_hybrid[i] = max(y_prob_hybrid[i], 0.9)
                elif res['decision'] == 'low':
                    y_prob_hybrid[i] = min(y_prob_hybrid[i], 0.1)
                    
        y_pred_hybrid = (y_prob_hybrid > 0.5).astype(int)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred_hybrid)
        rec = recall_score(y_test, y_pred_hybrid)
        auc = roc_auc_score(y_test, y_prob_hybrid)
        f1 = f1_score(y_test, y_pred_hybrid)
        
        results[opt] = {
            "Accuracy": acc,
            "Recall": rec,
            "AUC": auc,
            "F1": f1
        }
        
        print(f"Hybrid {opt.upper()}: Acc={acc:.4f}, Recall={rec:.4f}, AUC={auc:.4f}")

    print("\n" + "="*50)
    print("     HYBRID ARCHITECTURE RESULTS (FAST MODELS)     ")
    print("="*50)
    print(f"{ 'Base Model':<10} {'Accuracy':<10} {'Recall':<10} {'AUC':<10} {'F1':<10}")
    print("-" * 60)
    for opt, res in results.items():
        print(f"{opt.upper():<10} {res['Accuracy']:.4f}     {res['Recall']:.4f}     {res['AUC']:.4f}     {res['F1']:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    evaluate_fast_hybrid()
