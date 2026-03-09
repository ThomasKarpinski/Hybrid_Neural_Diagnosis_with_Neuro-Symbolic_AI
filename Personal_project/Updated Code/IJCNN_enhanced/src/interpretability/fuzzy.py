"""
Fuzzy inference module for two features: BMI and Age.
Defines triangular membership functions for low/medium/high.
Computes a fuzzy malignancy score P_fuzzy in [0,1].
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

def triangular_mf(x, a, b, c):
    """Triangular membership: peak at b, 0 at a and c"""
    x = float(x)
    if x <= a or x >= c:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b <= x < c:
        return (c - x) / (c - b)
    return 0.0

def fuzzify_bmi(bmi: float) -> Dict[str, float]:
    # Example cutoffs (tunable)
    return {
        "low": triangular_mf(bmi, 10, 18.5, 24.9),
        "medium": triangular_mf(bmi, 20, 27.5, 30),
        "high": triangular_mf(bmi, 25, 32.5, 60),
    }

def fuzzify_age(age: float) -> Dict[str, float]:
    # Age is 1-13 category
    return {
        "young": triangular_mf(age, 1, 2.5, 5),
        "mid": triangular_mf(age, 3, 7, 11),
        "old": triangular_mf(age, 8, 11, 13),
    }

def compute_p_fuzzy_from_row(row: Dict[str, Any]) -> float:
    """
    Compute fuzzy malignancy score for a single row using BMI and Age.
    Simple rule-based mapping:
      - high BMI AND old age -> high risk
      - low BMI AND young -> low risk
    Returns P_fuzzy in [0,1]
    """
    bmi = row.get("BMI", None)
    age = row.get("Age", None)
    if bmi is None or age is None:
        return 0.5  # neutral if missing

    bmi_f = fuzzify_bmi(float(bmi))
    age_f = fuzzify_age(float(age))

    # rule strengths:
    high_risk_strength = min(bmi_f["high"], age_f["old"])  # both high => strong
    med_risk_strength = min(bmi_f["medium"], age_f["mid"])
    low_risk_strength = min(bmi_f["low"], age_f["young"])

    # aggregate to [0,1] score: weights chosen heuristically
    score = 0.8 * high_risk_strength + 0.5 * med_risk_strength + 0.1 * low_risk_strength
    # normalize to [0,1] by maximum possible (0.8+0.5+0.1=1.4) though actual peaks rarely sum
    max_possible = 1.4
    return float(np.clip(score / max_possible, 0.0, 1.0))

def compute_p_fuzzy(
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    feature_map: Optional[Dict[str, str]] = None
) -> List[float]:
    """
    Vectorized: compute P_fuzzy for each row.
    """
    if isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names required for numpy input")
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = X.copy()

    if feature_map:
        inv_map = {v: k for k, v in feature_map.items()}
        df = df.rename(columns=inv_map)

    scores = []
    for _, row in df.iterrows():
        scores.append(compute_p_fuzzy_from_row(row.to_dict()))
    return scores

