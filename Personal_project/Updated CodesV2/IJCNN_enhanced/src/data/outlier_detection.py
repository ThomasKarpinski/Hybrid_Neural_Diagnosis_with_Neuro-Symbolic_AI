import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


def zscore_outliers(X: np.ndarray, threshold: float = 3.0):
    """
    Returns boolean mask of outlier samples using z-score.
    Outlier = any feature beyond ±threshold SD.
    """
    zscores = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return np.any(np.abs(zscores) > threshold, axis=1)


def iqr_outliers(X: np.ndarray, k: float = 1.5):
    """
    Detects outliers using the Interquartile Range (IQR) method.
    Outlier = value < Q1 - k*IQR or > Q3 + k*IQR.
    """
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1

    lower = Q1 - k * IQR
    upper = Q3 + k * IQR

    mask = np.any((X < lower) | (X > upper), axis=1)
    return mask


def mad_outliers(X: np.ndarray, threshold: float = 3.5):
    """
    Detect outliers using Median Absolute Deviation (MAD),
    robust to skewed distributions.
    """
    median = np.median(X, axis=0)
    abs_dev = np.abs(X - median)
    mad = np.median(abs_dev, axis=0)

    modified_z = 0.6745 * abs_dev / mad
    return np.any(modified_z > threshold, axis=1)


def isolation_forest_outliers(X: np.ndarray, contamination='auto', random_state=42):
    """
    Detects outliers using Isolation Forest.
    Returns boolean mask where True indicates an outlier.
    """
    iso = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    preds = iso.fit_predict(X)
    return preds == -1


def remove_outliers(X: np.ndarray,
                    y: np.ndarray,
                    method: str = "zscore") -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes outliers from dataset using the specified method.
    method ∈ {"zscore", "iqr", "mad", "isolation_forest"}
    """
    if method == "zscore":
        mask = zscore_outliers(X, threshold=3.0)
    elif method == "iqr":
        mask = iqr_outliers(X, k=1.5)
    elif method == "mad":
        mask = mad_outliers(X, threshold=3.5)
    elif method == "isolation_forest":
        mask = isolation_forest_outliers(X, contamination='auto')
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Outliers detected: {np.sum(mask)}")

    X_clean = X[~mask]
    y_clean = y[~mask]

    print(f"Remaining samples: {len(y_clean)}")
    return X_clean, y_clean


def plot_outlier_distributions(X: np.ndarray, feature_names=None):
    """
    Optional diagnostic visualization.
    """
    df = pd.DataFrame(X, columns=feature_names if feature_names else None)

    plt.figure(figsize=(16, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=90)
    plt.title("Boxplots for Outlier Inspection")
    plt.tight_layout()
    plt.show()

