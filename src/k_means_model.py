import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  # Needed for scaling before PCA/KMeans
from sklearn.decomposition import PCA  # Needed for PCA

from src.utils import evaluate_model  # Import evaluation helper


def train_and_evaluate_kmeans(X_train_base, X_test_base, y_train_base, y_test_base, random_state=42):
    """
    Trains and evaluates a K-means clustering model.
    Performs scaling and PCA before training K-means.

    Args:
        X_train_base (pd.DataFrame): Original training features.
        X_test_base (pd.DataFrame): Original test features.
        y_train_base (pd.Series): Original training target.
        y_test_base (pd.Series): Original test target.
        random_state (int): Seed for reproducibility.
    """
    print("\n--- Training K-means Clustering Model ---")

    # 1. Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_base)
    X_test_scaled = scaler.transform(X_test_base)
    print("Data scaled.")

    # 2. Apply PCA for dimensionality reduction to 2 components for visualization
    pca_kmeans = PCA(n_components=2, random_state=random_state)
    X_train_pca = pca_kmeans.fit_transform(X_train_scaled)
    X_test_pca = pca_kmeans.transform(X_test_scaled)  # Transform test data using the *same* PCA fit
    print(f"PCA applied. Reduced dimensions from {X_train_base.shape[1]} to 2.")

    # Initialize and train the K-means model
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=random_state, max_iter=300)
    print("Fitting K-means model...")
    kmeans.fit(X_train_pca)

    # Predict clusters for the test set
    clusters = kmeans.predict(X_test_pca)

    # --- Mapping K-means clusters to actual classes for evaluation ---
    # K-means assigns arbitrary labels (0 or 1). We need to map them to actual classes.
    # We'll map the cluster with more '1's (frauds) in the test set to the 'fraud' class.

    # Find which cluster (0 or 1) has more actual '1's (frauds) in the test set
    # This requires checking the actual labels (y_test_base) against the clusters.
    fraud_cluster_0_count = np.sum(y_test_base[clusters == 0] == 1)
    fraud_cluster_1_count = np.sum(y_test_base[clusters == 1] == 1)

    # Create a mapping: cluster_label -> predicted_class_label
    if fraud_cluster_0_count > fraud_cluster_1_count:
        cluster_to_class_mapping = {0: 1, 1: 0}
    else:
        cluster_to_class_mapping = {0: 0, 1: 1}

    # Apply the mapping to the predicted clusters
    y_pred = np.array([cluster_to_class_mapping[c] for c in clusters])

    # Evaluate the model
    evaluate_model(y_test_base, y_pred, "K-means Clustering (PCA-reduced)")

    # Visualize the clusters and centroids on the test data
    plt.figure(figsize=(10, 8))
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=clusters, cmap='viridis', s=5, alpha=0.5)
    # Plot the centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=3, color='red', label='Centroids')
    plt.title('K-means Clustering on Test Data (PCA-Reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
    print(
        "Visualization of K-means clusters on PCA-reduced test data. Note: The cluster labels (colors) are arbitrary.")
    print("The red 'X' marks are the centroids.")


if __name__ == '__main__':
    # This block demonstrates how to use the functions if run directly
    print("Running K-means model functions directly for demonstration.")
    # Dummy data for demonstration
    from sklearn.datasets import make_classification
    import pandas as pd
    from sklearn.model_selection import train_test_split

    X_dummy, y_dummy = make_classification(n_samples=1000, n_features=30, n_informative=10,
                                           n_redundant=5, n_classes=2, weights=[0.99, 0.01],
                                           flip_y=0, random_state=42)
    X_dummy = pd.DataFrame(X_dummy)
    y_dummy = pd.Series(y_dummy)

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dummy, y_dummy, test_size=0.3, random_state=42,
                                                                stratify=y_dummy)

    train_and_evaluate_kmeans(X_train_d, X_test_d, y_train_d, y_test_d)
