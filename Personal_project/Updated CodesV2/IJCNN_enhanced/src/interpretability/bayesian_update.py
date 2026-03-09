"""
Bayesian posterior computation using class-conditional Gaussian likelihoods
for selected numeric features. Uses log-probabilities to avoid underflow.

Fits per-class mean and variance on training data, then computes:
  P(M|x) = P(x|M) P(M) / (P(x|M)P(M) + P(x|~M)P(~M))
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from math import log, exp

class GaussianNaiveBayesLike:
    def __init__(self, features: List[str]):
        self.features = features
        self.class_stats = {}  # class_value -> {feature: (mean,var)}
        self.class_priors = {}  # class_value -> prior prob

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[np.ndarray, List[int]],
            feature_names: Optional[List[str]] = None, feature_map: Optional[Dict[str,str]] = None):
        if isinstance(X, np.ndarray):
            if feature_names is None:
                raise ValueError("feature_names required for numpy input")
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()
        if feature_map:
            inv_map = {v:k for k,v in feature_map.items()}
            df = df.rename(columns=inv_map)

        y_arr = np.array(y).reshape(-1)
        classes, counts = np.unique(y_arr, return_counts=True)
        n = len(y_arr)
        for c, cnt in zip(classes, counts):
            self.class_priors[int(c)] = cnt / n
            sub = df[y_arr == c][self.features]
            stats = {}
            for f in self.features:
                col = sub[f].dropna().astype(float)
                if len(col) == 0:
                    stats[f] = (0.0, 1.0)
                else:
                    stats[f] = (col.mean(), max(col.var(ddof=0), 1e-6))
            self.class_stats[int(c)] = stats

    def _log_gaussian_pdf(self, x, mean, var):
        # log pdf of Gaussian
        # -0.5 * (log(2πσ^2) + (x-μ)^2 / σ^2)
        return -0.5 * (np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray],
                      feature_names: Optional[List[str]] = None,
                      feature_map: Optional[Dict[str,str]] = None) -> List[float]:
        if isinstance(X, np.ndarray):
            if feature_names is None:
                raise ValueError("feature_names required for numpy input")
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()
        if feature_map:
            inv_map = {v:k for k,v in feature_map.items()}
            df = df.rename(columns=inv_map)

        probs = []
        for _, row in df.iterrows():
            # compute log-likelihood for each class
            log_likes = {}
            for c, stats in self.class_stats.items():
                s = 0.0
                for f in self.features:
                    x = row[f]
                    mean, var = stats[f]
                    s += self._log_gaussian_pdf(float(x), mean, var)
                log_prior = np.log(self.class_priors.get(c, 1e-9))
                log_likes[c] = s + log_prior

            # stable normalization: subtract max
            max_log = max(log_likes.values())
            exp_sum = sum(np.exp(log_likes[c] - max_log) for c in log_likes)
            post_M = np.exp(log_likes.get(1, -1e9) - max_log) / exp_sum if 1 in log_likes else 0.0
            probs.append(float(post_M))
        return probs

