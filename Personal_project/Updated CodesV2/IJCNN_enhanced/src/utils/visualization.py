import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def plot_class_distribution(y, save_path=None):
    """Plot diabetes vs non-diabetes distribution."""
    counts = pd.Series(y).value_counts()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("Class Distribution (Diabetes Binary)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def pairplot_features(X, y, sample_size=5000, save_path=None):
    """
    Pairwise relationships between a random subset of features.
    CDC dataset has many features, so we limit pairplot to 8.
    """
    df = pd.DataFrame(X)
    df["target"] = y

    # pick 8 features (or less if not enough)
    n_features = min(df.shape[1] - 1, 8) # -1 because of target column
    selected_features = df.columns[:n_features].tolist()
    
    if "target" in selected_features:
        selected_features.remove("target")

    # Sample if too large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    sns.pairplot(df[selected_features + ["target"]], hue="target", diag_kind="kde", corner=True)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def perform_unsupervised_analysis(X, y, save_prefix="experiments/unsupervised"):
    """
    Performs K-Means clustering and PCA visualization.
    """
    print("Running PCA...")
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print("Running K-Means...")
    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # DataFrame for plotting
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['True_Label'] = y
    df_pca['Cluster'] = clusters
    
    print("Generating plots...")
    # Plot 1: True Labels
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='True_Label', alpha=0.6, palette='viridis')
    plt.title("PCA Visualization - True Labels")
    if save_prefix:
        plt.savefig(f"{save_prefix}_pca_labels.png")
    plt.close()
    
    # Plot 2: K-Means Clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', alpha=0.6, palette='rocket')
    plt.title("PCA Visualization - K-Means Clusters")
    if save_prefix:
        plt.savefig(f"{save_prefix}_pca_clusters.png")
    plt.close()
    
    print(f"Unsupervised analysis plots saved to {save_prefix}*")


def plot_confusion_matrix(y_true, y_pred_probs, threshold=0.5, save_path=None):
    """
    Plots the confusion matrix.
    y_true: actual labels
    y_pred_probs: predicted probabilities
    threshold: probability threshold for classification
    """
    y_pred = (y_pred_probs > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    print(f"Confusion Matrix plot saved to {save_path}")
