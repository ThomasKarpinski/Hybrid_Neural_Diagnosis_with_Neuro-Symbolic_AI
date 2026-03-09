import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from src.data.load_data import prepare_data
from src.models.mlp_baseline import BaselineMLP
from src.interpretability.rules import apply_rules_dataframe
from src.interpretability.fuzzy import compute_p_fuzzy
from src.interpretability.bayesian_update import GaussianNaiveBayesLike

MODEL_PATH = "experiments/best_models/baseline_mlp_hpo.pth"

def check_metrics():
    print("--- Checking Metrics ---")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    if isinstance(X_test, pd.DataFrame): X_test = X_test.values
    if isinstance(y_test, pd.Series): y_test = y_test.values

    # 2. Load MLP
    input_dim = X_train.shape[1]
    model = BaselineMLP(input_dim)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("Model not found! Using untrained (random) model.")
    
    model.eval() 
    
    # 3. Predict MLP
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_prob_mlp = model(X_tensor).numpy().flatten()
        y_pred_mlp = (y_prob_mlp > 0.5).astype(int)

    # 4. Hybrid Logic
    X_test_unscaled = scaler.inverse_transform(X_test)
    df_test = pd.DataFrame(X_test_unscaled, columns=feature_names)
    
    # Fuzzy
    p_fuzzy = np.array(compute_p_fuzzy(df_test))
    
    # Bayes
    bayes = GaussianNaiveBayesLike(feature_names)
    bayes.fit(X_train, y_train, feature_names)
    p_bayes = bayes.predict_proba(X_test, feature_names)
    
    # Fusion
    y_prob_hybrid = (y_prob_mlp + p_fuzzy + p_bayes) / 3.0
    
    # Rules Override
    rule_decisions = apply_rules_dataframe(df_test)
    for i, res in enumerate(rule_decisions):
        if res:
            if res['decision'] == 'high':
                y_prob_hybrid[i] = max(y_prob_hybrid[i], 0.9)
            elif res['decision'] == 'low':
                y_prob_hybrid[i] = min(y_prob_hybrid[i], 0.1)

    y_pred_hybrid = (y_prob_hybrid > 0.5).astype(int)

    # --- CALCULATE METRICS ---
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    # MLP Metrics
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    prec_mlp = precision_score(y_test, y_pred_mlp, zero_division=0)
    rec_mlp = recall_score(y_test, y_pred_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp)
    
    # ROC-AUC
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
    auc_mlp = auc(fpr_mlp, tpr_mlp)
    
    print(f"\nMLP Metrics:")
    print(f"  Acc: {acc_mlp:.4f}")
    print(f"  Prec: {prec_mlp:.4f}")
    print(f"  Rec: {rec_mlp:.4f}")
    print(f"  F1: {f1_mlp:.4f}")
    print(f"  ROC-AUC: {auc_mlp:.4f}")

    # Hybrid Metrics
    acc_hyb = accuracy_score(y_test, y_pred_hybrid)
    prec_hyb = precision_score(y_test, y_pred_hybrid, zero_division=0)
    rec_hyb = recall_score(y_test, y_pred_hybrid)
    f1_hyb = f1_score(y_test, y_pred_hybrid)
    
    fpr_hyb, tpr_hyb, _ = roc_curve(y_test, y_prob_hybrid)
    auc_hyb = auc(fpr_hyb, tpr_hyb)
    
    print(f"\nHybrid Metrics:")
    print(f"  Acc: {acc_hyb:.4f}")
    print(f"  Prec: {prec_hyb:.4f}")
    print(f"  Rec: {rec_hyb:.4f}")
    print(f"  F1: {f1_hyb:.4f}")
    print(f"  ROC-AUC: {auc_hyb:.4f}")
    
    # Safety Net Stats
    # "When the MLP produced a False Negative (missing a diabetic case), 
    # the Rule-Based system successfully flagged the patient as 'High Risk' in X% of cases."
    
    fn_indices = np.where((y_test == 1) & (y_pred_mlp == 0))[0]
    if len(fn_indices) > 0:
        rules_caught = 0
        for idx in fn_indices:
            res = rule_decisions[idx]
            if res and res['decision'] == 'high':
                rules_caught += 1
        print(f"Safety Net (Rules catching MLP False Negatives): {rules_caught}/{len(fn_indices)} = {rules_caught/len(fn_indices)*100:.2f}%")
    else:
        print("No False Negatives found.")

    # "Correction of False Alarms: Conversely, when the MLP produced a False Positive,
    # the Fuzzy Inference system correctly indicated 'Low Risk' in X% of cases."
    # Assuming "Low Risk" means p_fuzzy < 0.5 (or low risk category if fuzzy logic returns that, but here we have score).
    # The paper says "Fuzzy Inference system correctly indicated 'Low Risk'".
    # Let's assume this means fuzzy score < 0.5 or similar.
    
    fp_indices = np.where((y_test == 0) & (y_pred_mlp == 1))[0]
    if len(fp_indices) > 0:
        fuzzy_corrected = 0
        for idx in fp_indices:
            if p_fuzzy[idx] < 0.4: # Assuming Low Risk threshold
                fuzzy_corrected += 1
        print(f"False Alarm Correction (Fuzzy < 0.4 on MLP False Positives): {fuzzy_corrected}/{len(fp_indices)} = {fuzzy_corrected/len(fp_indices)*100:.2f}%")
    else:
        print("No False Positives found.")

    # Alignment
    # "MLP and Fuzzy Logic system showed high global alignment, disagreeing on only X% of the entire test set"
    y_pred_fuzzy = (p_fuzzy > 0.5).astype(int)
    disagreement = np.mean(y_pred_mlp != y_pred_fuzzy)
    print(f"Global Disagreement (MLP vs Fuzzy): {disagreement*100:.2f}%")

if __name__ == "__main__":
    check_metrics()
