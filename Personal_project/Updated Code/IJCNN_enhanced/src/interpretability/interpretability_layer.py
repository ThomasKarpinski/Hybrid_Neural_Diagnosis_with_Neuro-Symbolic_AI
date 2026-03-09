"""
Integration layer that combines:
 - Rule-based overrides (strong decisions)
 - Fuzzy inference (P_fuzzy)
 - MLP probability (P_MLP)
 - Bayesian posterior P(M|x)

Logic:
 1. If a rule triggers and it's decisive (high or low), and the model confidence is low, we can override.
 2. Compute P_fuzzy from fuzzy module.
 3. Combine P_MLP and P_fuzzy via P* = alpha * P_MLP + (1-alpha) * P_fuzzy
 4. Also compute Bayesian posterior for interpretability
 5. Return final probability, rule info, fuzzy score, bayes posterior, explanation string
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

from .rules import apply_rules_dataframe
from .fuzzy import compute_p_fuzzy
from .bayesian_update import GaussianNaiveBayesLike

def interpret_batch(
    X: Union[pd.DataFrame, np.ndarray],
    P_mlp: List[float],
    model_confidence_threshold: float = 0.6,
    alpha: float = 0.7,
    feature_names: Optional[List[str]] = None,
    feature_map: Optional[Dict[str, str]] = None,
    bayes_features: Optional[List[str]] = None,
    gnb_model: Optional[GaussianNaiveBayesLike] = None,
) -> List[Dict[str, Any]]:
    """
    For each sample, compute interpretability outputs.

    Parameters:
      X: features (DataFrame or numpy array)
      P_mlp: list/array of model probabilities (same order as X rows)
      model_confidence_threshold: below this threshold we consider model uncertain
      alpha: weight for P_MLP when combining with P_fuzzy
      bayes_features: features to use for Bayesian posterior (if None, default used)
      gnb_model: optional fitted GaussianNaiveBayesLike instance. If not provided, will be None for bayes posterior.

    Returns:
      list of dicts:
        {
          "p_mlp": float,
          "p_fuzzy": float,
          "p_combined": float,
          "p_bayes": float or None,
          "rule": dict or None,
          "final_decision": 0/1,
          "explanation": str
        }
    """
    if isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names required for numpy input")
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = X.copy()

    # apply rules
    rule_results = apply_rules_dataframe(df, feature_names=feature_names, feature_map=feature_map)
    p_fuzzy_list = compute_p_fuzzy(df, feature_names=feature_names, feature_map=feature_map)

    # prepare bayes posterior if model provided
    if gnb_model is not None:
        p_bayes_list = gnb_model.predict_proba(df, feature_names=feature_names, feature_map=feature_map)
    else:
        p_bayes_list = [None] * len(df)

    outputs = []
    for i, row in df.iterrows():
        p_mlp = float(P_mlp[i])
        p_fuzzy = float(p_fuzzy_list[i])
        rule = rule_results[i]
        p_bayes = p_bayes_list[i]
        # Combine fuzzy and mlp
        p_combined = float(np.clip(alpha * p_mlp + (1 - alpha) * p_fuzzy, 0.0, 1.0))

        rule_applied = False
        final_p = p_combined
        explanation = []

        # If a decisive rule exists and model confidence is low -> override or sharpen
        if rule is not None:
            explanation.append(f"Rule triggered: {rule['reason']} -> {rule['decision']}")
            # decide override logic:
            conf = abs(p_mlp - 0.5)  # model confidence centered at 0.5
            if conf < (model_confidence_threshold - 0.5):  # low confidence
                rule_applied = True
                if rule["decision"] == "high":
                    final_p = max(final_p, 0.9)
                elif rule["decision"] == "low":
                    final_p = min(final_p, 0.1)
                explanation.append("Rule applied because model uncertainty was high.")
            else:
                explanation.append("Rule available but model confident; rule is advisory only.")

        # incorporate bayesian posterior into explanation and optionally refine
        if p_bayes is not None:
            explanation.append(f"Bayes posterior P(M|x)={p_bayes:.3f}")
            # mild smoothing with bayes posterior (heuristic)
            final_p = float(0.85 * final_p + 0.15 * p_bayes)

        # final binary decision using 0.5 threshold
        final_decision = int(final_p >= 0.5)
        explanation = " | ".join(explanation) if explanation else "No rule triggered. Interpretation based on P_MLP and P_fuzzy."

        outputs.append({
            "p_mlp": p_mlp,
            "p_fuzzy": p_fuzzy,
            "p_combined": p_combined,
            "p_bayes": p_bayes,
            "rule": rule,
            "rule_applied": rule_applied,
            "final_p": final_p,
            "final_decision": final_decision,
            "explanation": explanation
        })

    return outputs

