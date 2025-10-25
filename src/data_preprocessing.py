import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import os

# Import helper functions from utils.py
from src.utils import plot_feature_distribution


def load_and_explore_data(file_path):
    """
    Loads the credit card fraud dataset and performs initial exploration.

    Args:
        file_path (str): The path to the creditcard.csv file.

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): The loaded dataframe.
            - X (pd.DataFrame): Features dataframe.
            - y (pd.Series): Target series.
            - frauds (pd.DataFrame): Fraudulent transactions.
            - non_frauds (pd.DataFrame): Non-fraudulent transactions.
    """
    print("--- Data Loading and Initial Exploration ---")
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please ensure the dataset is in the correct directory.")
        return None, None, None, None, None

    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Info:")
    df.info()

    # Separate features (X) and target (y)
    X = df.iloc[:, :-1]
    y = df['Class']

    # Check the distribution of the target variable (Class)
    frauds = df.loc[df['Class'] == 1]
    non_frauds = df.loc[df['Class'] == 0]
    print(f"\nWe have {len(frauds)} fraud data points and {len(non_frauds)} non-fraudulent data points.")
    print(f"Fraudulent transactions make up {len(frauds) / len(df) * 100:.3f}% of the dataset.")

    # Visualize key feature distributions
    plot_feature_distribution(df, frauds, non_frauds, 'Amount')
    # plot_feature_distribution(df, frauds, non_frauds, 'V1') # Uncomment to visualize other features

    return df, X, y, frauds, non_frauds


def split_data(X, y, test_size=0.35, random_state=42):
    """
    Splits the data into training and testing sets, stratifying by the target variable.

    Args:
        X (pd.DataFrame): Features dataframe.
        y (pd.Series): Target series.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generator for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(f"\n--- Splitting Data (Test Size: {test_size * 100}%, Random State: {random_state}) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Frauds in training set: {y_train.sum()} ({y_train.sum() / len(y_train) * 100:.3f}%)")
    print(f"Frauds in test set: {y_test.sum()} ({y_test.sum() / len(y_test) * 100:.3f}%)")
    return X_train, X_test, y_train, y_test


def preprocess_for_nn_kmeans(X_train, X_test, y_train, n_components_pca=10, random_state=42):
    """
    Performs scaling, SMOTE oversampling, and PCA for Neural Network and K-means models.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        n_components_pca (int): Number of PCA components.
        random_state (int): Seed for random number generator for reproducibility.

    Returns:
        tuple: A tuple containing:
            - X_train_processed (np.array): Processed training features.
            - X_test_processed (np.array): Processed test features.
            - y_train_processed (np.array): Processed training target (after SMOTE).
            - pca_model (PCA): The fitted PCA model.
    """
    print(f"\n--- Preprocessing for NN/K-means (Scaling, SMOTE, PCA to {n_components_pca} components) ---")

    # 1. Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data scaled.")

    # 2. Apply SMOTE on the training data to address imbalance
    # SMOTE should only be applied to training data to prevent data leakage.
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print(f"SMOTE applied. Training set size BEFORE SMOTE: {len(X_train)} -> AFTER SMOTE: {len(X_train_smote)}")
    print(
        f"Fraud instances AFTER SMOTE: {y_train_smote.sum()} | Non-fraud instances AFTER SMOTE: {len(y_train_smote) - y_train_smote.sum()}")

    # 3. Apply PCA for dimensionality reduction
    # Fit PCA on the SMOTE-augmented training data
    pca = PCA(n_components=n_components_pca, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_smote)
    # Transform test data using the *same* PCA model fitted on training data
    X_test_pca = pca.transform(X_test_scaled)
    print(f"PCA applied. Reduced dimensions from {X_train.shape[1]} to {n_components_pca}.")

    print(f"Processed X_train shape: {X_train_pca.shape}")
    print(f"Processed X_test shape: {X_test_pca.shape}")

    return X_train_pca, X_test_pca, y_train_smote, pca

