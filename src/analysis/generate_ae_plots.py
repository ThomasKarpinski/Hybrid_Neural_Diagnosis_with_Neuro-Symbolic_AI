import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import os
import sys
from sklearn.decomposition import PCA

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.models.autoencoder import Autoencoder, get_embeddings

def generate_ae_visualization():
    print("Generating Autoencoder Embedding Visualization...")
    os.makedirs("paper/figures", exist_ok=True)
    
    # 1. Load Data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    # Remove outliers to match training distribution
    X_clean, y_clean = remove_outliers(X_train, y_train)
    
    # Sample for plotting if too large
    if len(X_clean) > 5000:
        idx = np.random.choice(len(X_clean), 5000, replace=False)
        X_plot = X_clean[idx]
        y_plot = y_clean[idx]
    else:
        X_plot = X_clean
        y_plot = y_clean

    # 2. Load Model
    input_dim = X_train.shape[1]
    encoding_dim = 10 # Default in train_autoencoder
    model = Autoencoder(input_dim, encoding_dim)
    
    model_path = "experiments/best_models/autoencoder.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded AE from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Cannot plot embeddings.")
        return

    # 3. Get Embeddings
    embeddings = get_embeddings(model, X_plot)
    
    # 4. Reduce to 2D for visualization (PCA on Bottleneck)
    # Even though AE is non-linear, visualizing the bottleneck often requires 
    # another step if dim > 2. t-SNE is better but slower; PCA is standard for "viewing axes".
    pca_ae = PCA(n_components=2)
    embeddings_2d = pca_ae.fit_transform(embeddings)
    
    # 5. Plot
    df_plot = pd.DataFrame(embeddings_2d, columns=['Dim1', 'Dim2'])
    df_plot['Label'] = y_plot
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_plot, x='Dim1', y='Dim2', hue='Label', alpha=0.6, palette='viridis')
    plt.title("Autoencoder Latent Space (PCA-reduced from 10D)")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.tight_layout()
    plt.savefig("paper/figures/unsupervised_ae_labels.png", dpi=300)
    plt.close()
    print("Autoencoder plot saved.")

if __name__ == "__main__":
    generate_ae_visualization()
