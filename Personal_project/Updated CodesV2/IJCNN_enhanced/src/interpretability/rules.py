"""
Rule-based reasoning for CDC Diabetes indicators.

Rules are expressed in terms of canonical feature names:
  'BMI', 'Age', 'HighBP', 'HighChol', 'PhysActivity', 'Fruits', 'GenHlth', 'DiffWalk'

If your dataset uses different column names, provide a feature_map mapping these canonical
names to your dataset columns (see interpretability_layer.py).
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

# Example rules: returns 'high', 'low', or None (no decisive rule)
def example_rules(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Evaluate example rules for one sample (row is dict-like of feature values).
    Returns a dict {"rule_id": id, "decision": "high"/"low", "reason": str} or None.
    """

    # Rule 1: Obesity + older age -> high risk
    try:
        # Age in CDC is 1-13 (1=18-24, 13=80+). Cat 6 is 45-49.
        if (row.get("BMI", np.nan) is not None) and row["BMI"] >= 30 and row.get("Age", 0) >= 6:
            return {"rule_id": 1, "decision": "high", "reason": "BMI>=30 AND Age>=6 (45+)"}
    except Exception:
        pass

    # Rule 2: High blood pressure and high cholesterol
    try:
        if int(row.get("HighBP", 0)) == 1 and int(row.get("HighChol", 0)) == 1:
            return {"rule_id": 2, "decision": "high", "reason": "HighBP==1 AND HighChol==1"}
    except Exception:
        pass

    # Rule 3: No physical activity and no fruit intake -> increased risk
    try:
        if int(row.get("PhysActivity", 1)) == 0 and int(row.get("Fruits", 1)) == 0:
            return {"rule_id": 3, "decision": "high", "reason": "PhysActivity==0 AND Fruits==0"}
    except Exception:
        pass

    # Rule 4: Poor general health and difficulty walking -> high risk
    try:
        # GenHlth often coded 1..5 where 1=excellent,5=poor
        if row.get("GenHlth", 0) >= 4 and int(row.get("DiffWalk", 0)) == 1:
            return {"rule_id": 4, "decision": "high", "reason": "GenHlth>=4 AND DiffWalk==1"}
    except Exception:
        pass

    # Rule 5: Healthy lifestyle indicators -> low risk
    try:
        if row.get("BMI", 999) < 25 and int(row.get("PhysActivity", 0)) == 1 and row.get("GenHlth", 5) <= 2:
            return {"rule_id": 5, "decision": "low", "reason": "BMI<25 AND PhysActivity==1 AND GenHlth<=2"}
    except Exception:
        pass

    return None


def apply_rules_dataframe(
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    feature_map: Optional[Dict[str, str]] = None,
) -> List[Optional[Dict[str, Any]]]:
    """
    Apply example_rules to each row in X.

    Parameters:
      X: pandas DataFrame or numpy array
      feature_names: list of column names if X is numpy array
      feature_map: optional mapping canonical_name -> actual_column_name

    Returns:
      List of rule outputs (None if no rule applies).
    """
    if isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names required for numpy input")
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = X.copy()

    # rename columns by canonical names if feature_map provided
    if feature_map:
        inv_map = {v: k for k, v in feature_map.items()}  # actual->canonical
        df = df.rename(columns=inv_map)

    results = []
    for _, row in df.iterrows():
        res = example_rules(row.to_dict())
        results.append(res)
    return results

