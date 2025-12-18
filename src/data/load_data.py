import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

def load_raw_cdc():
    """Fetch raw CDC Diabetes dataset from local file or ucimlrepo."""
    import os
    local_file = "diabetes_binary_health_indicators_BRFSS2015.csv"
    
    if os.path.exists(local_file):
        print(f"Loading data from local file: {local_file}")
        df = pd.read_csv(local_file)
        y = df["Diabetes_binary"]
        X = df.drop(columns=["Diabetes_binary"])
        return X, y
    
    print("Local file not found, fetching from ucimlrepo...")
    ds = fetch_ucirepo(id=891)
    X = pd.DataFrame(ds.data.features)
    y = pd.Series(ds.data.targets["Diabetes_binary"], name="target")
    return X, y


def normalize_features(X_train, X_test):
    """Standardize all features using training statistics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline:
    - Load data
    - Stratified split
    - Normalize features
    """
    X, y = load_raw_cdc()
    
    # Capture feature names
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    return (
        X_train_scaled, X_test_scaled,
        y_train.to_numpy(), y_test.to_numpy(),
        scaler,
        feature_names
    )

