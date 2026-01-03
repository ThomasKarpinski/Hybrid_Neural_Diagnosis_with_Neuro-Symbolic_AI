import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.load_data import prepare_data, load_raw_cdc
from src.data.outlier_detection import remove_outliers

def generate_extra_plots():
    print("Generating Extra Data Plots...")
    os.makedirs("paper/figures", exist_ok=True)
    
    # Load data
    X_raw, y_raw = load_raw_cdc()
    
    # 1. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    # Select numeric columns
    numeric_df = X_raw.select_dtypes(include=[np.number])
    # Compute correlation
    corr = numeric_df.corr()
    
    # Plot
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig("paper/figures/corr_heatmap.png", dpi=300)
    plt.close()
    
    # 2. Feature Histograms
    print("Generating Feature Histograms (All Variables)...")
    # Select all numeric features
    features_to_plot = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    
    if features_to_plot:
        # Calculate grid size
        n_features = len(features_to_plot)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        X_raw[features_to_plot].hist(figsize=(20, 5 * n_rows), bins=20, layout=(n_rows, n_cols))
        plt.suptitle("Feature Distributions", y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig("paper/figures/feature_hist.png", dpi=300)
        plt.close()
    
    print("Extra plots generated.")

if __name__ == "__main__":
    generate_extra_plots()
