import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap import UMAP

# Add project root to path
sys.path.append(os.getcwd())

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.models.autoencoder import Autoencoder, get_embeddings

def generate_plots():
    print("Loading data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    
    os.makedirs("figures", exist_ok=True)
    sample_size = 5000

    # ---------------------------------------------------------
    # 1. ISO Analysis (PCA, K-Means, AE)
    # ---------------------------------------------------------
    print("Processing Isolation Forest (ISO)...")
    X_iso, y_iso = remove_outliers(X_train, y_train, method="isolation_forest")
    
    # Sampling for visualization speed and clarity
    idx_iso = np.random.choice(len(X_iso), min(len(X_iso), sample_size), replace=False)
    X_vis_iso = X_iso[idx_iso]
    y_vis_iso = y_iso[idx_iso]

    # PCA & K-Means
    pca = PCA(n_components=2)
    X_pca_iso = pca.fit_transform(X_vis_iso)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters_iso = kmeans.fit_predict(X_vis_iso)
    
    df_pca = pd.DataFrame(X_pca_iso, columns=['PC1', 'PC2'])
    df_pca['Label'] = y_vis_iso
    df_pca['Cluster'] = clusters_iso

    # Plot PCA Labels (ISO)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Label', alpha=0.6, palette='viridis')
    plt.title("PCA Visualization - True Labels (Isolation Forest)")
    plt.savefig("figures/unsupervised_pca_labels_iso.png", dpi=300)
    plt.close()

    # Plot PCA Clusters (ISO)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', alpha=0.6, palette='rocket')
    plt.title("PCA Visualization - K-Means Clusters (Isolation Forest)")
    plt.savefig("figures/unsupervised_pca_clusters_iso.png", dpi=300)
    plt.close()

    # Autoencoder Latent Space (ISO)
    print("Generating AE visualization (ISO)...")
    input_dim = X_train.shape[1]
    encoding_dim = 10
    ae_model = Autoencoder(input_dim, encoding_dim)
    ae_path = "experiments/best_models/autoencoder.pth"
    if os.path.exists(ae_path):
        ae_model.load_state_dict(torch.load(ae_path, map_location='cpu'))
        ae_embeddings = get_embeddings(ae_model, X_vis_iso)
        pca_ae = PCA(n_components=2)
        ae_2d = pca_ae.fit_transform(ae_embeddings)
        df_ae = pd.DataFrame(ae_2d, columns=['Dim1', 'Dim2'])
        df_ae['Label'] = y_vis_iso
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_ae, x='Dim1', y='Dim2', hue='Label', alpha=0.6, palette='viridis')
        plt.title("Autoencoder Latent Space (ISO cleaned data)")
        plt.savefig("figures/unsupervised_ae_labels_iso.png", dpi=300)
        plt.close()
    else:
        print("Warning: Autoencoder model not found.")

    # ---------------------------------------------------------
    # 2. UMAP Comparison (Z-score vs ISO)
    # ---------------------------------------------------------
    print("Processing Z-score for comparison...")
    X_z, y_z = remove_outliers(X_train, y_train, method="zscore")
    idx_z = np.random.choice(len(X_z), min(len(X_z), sample_size), replace=False)
    X_vis_z = X_z[idx_z]
    y_vis_z = y_z[idx_z]

    print("Running UMAP on Z-score data...")
    umap_model = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap_z = umap_model.fit_transform(X_vis_z)

    print("Running UMAP on ISO data...")
    X_umap_iso = umap_model.fit_transform(X_vis_iso)

    # Combined UMAP Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Z-score
    sns.scatterplot(x=X_umap_z[:, 0], y=X_umap_z[:, 1], hue=y_vis_z, alpha=0.5, palette='viridis', ax=ax1)
    ax1.set_title("UMAP Visualization (Z-score cleaning)")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    
    # ISO
    sns.scatterplot(x=X_umap_iso[:, 0], y=X_umap_iso[:, 1], hue=y_vis_iso, alpha=0.5, palette='viridis', ax=ax2)
    ax2.set_title("UMAP Visualization (Isolation Forest cleaning)")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    
    plt.suptitle("Manifold Comparison: Z-score vs Isolation Forest")
    plt.tight_layout()
    plt.savefig("figures/umap_comparison.png", dpi=300)
    plt.close()
    
    print("All plots generated successfully.")

if __name__ == "__main__":
    generate_plots()
