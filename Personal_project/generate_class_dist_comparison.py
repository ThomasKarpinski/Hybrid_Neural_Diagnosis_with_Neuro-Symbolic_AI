
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers

def generate_comparison_plot():
    print("Loading data...")
    # X_train is already scaled
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    
    # 1. Original Distribution
    original_counts = pd.Series(y_train).value_counts().sort_index()
    print("Original counts:", original_counts.to_dict())
    
    # 2. Z-score (threshold=3.0)
    print("Applying Z-score...")
    # remove_outliers uses zscore with threshold=3.0 by default
    _, y_train_zscore = remove_outliers(X_train, y_train, method="zscore")
    zscore_counts = pd.Series(y_train_zscore).value_counts().sort_index()
    print("Z-score counts:", zscore_counts.to_dict())
    
    # 3. Isolation Forest
    print("Applying Isolation Forest...")
    # Using default contamination='auto' or a fixed small percentage?
    # Usually IF is used to find rare anomalies. 
    # Let's align with typical usage. 'auto' determines threshold based on decision function.
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    # Fit predicts -1 for outliers, 1 for inliers
    preds = iso.fit_predict(X_train)
    mask_iso = (preds == 1) # Keep inliers
    y_train_iso = y_train[mask_iso]
    iso_counts = pd.Series(y_train_iso).value_counts().sort_index()
    print(f"Isolation Forest kept {np.sum(mask_iso)} samples.")
    print("Isolation Forest counts:", iso_counts.to_dict())
    
    # Prepare DataFrame for plotting
    data = []
    
    # Class 0
    data.append({'Method': 'Original', 'Class': '0 (No Diabetes)', 'Count': original_counts.get(0, 0)})
    data.append({'Method': 'Z-score', 'Class': '0 (No Diabetes)', 'Count': zscore_counts.get(0, 0)})
    data.append({'Method': 'Isolation Forest', 'Class': '0 (No Diabetes)', 'Count': iso_counts.get(0, 0)})
    
    # Class 1
    data.append({'Method': 'Original', 'Class': '1 (Diabetes)', 'Count': original_counts.get(1, 0)})
    data.append({'Method': 'Z-score', 'Class': '1 (Diabetes)', 'Count': zscore_counts.get(1, 0)})
    data.append({'Method': 'Isolation Forest', 'Class': '1 (Diabetes)', 'Count': iso_counts.get(1, 0)})
    
    df_plot = pd.DataFrame(data)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(data=df_plot, x='Method', y='Count', hue='Class', palette='viridis')
    
    plt.title("Class Distribution Comparison: Z-score vs Isolation Forest", fontsize=14)
    plt.xlabel("Outlier Detection Method", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    plt.legend(title="Class")
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
        
    plt.tight_layout()
    save_path = "paper/figures/class_distribution_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    generate_comparison_plot()
