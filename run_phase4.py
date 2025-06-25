import os
import sys
import warnings

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import data loading and preprocessing functions
from data_preprocessing import load_and_explore_data, split_data, preprocess_for_nn_kmeans

# Import Neural Network model functions (including new advanced training function)
from neural_network_model import train_and_evaluate_advanced_nn

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def main():
    """
    Main function for Phase 6: Advanced Neural Network Improvements.
    Loads data, preprocesses it, and trains/evaluates the advanced NN model.
    """
    print("==============================================")
    print("   Phase 6: Advanced Neural Network Model")
    print("==============================================")

    # --- Data Loading and Splitting (reusing Phase 1 logic) ---
    print("\n--- SECTION: Data Loading and Splitting ---")
    data_file_path = os.path.join(current_dir, 'data', 'creditcard.csv')
    df, X, y, _, _ = load_and_explore_data(data_file_path)

    if df is None:
        print("\nPhase 6 aborted: Data not loaded. Please check file path and dataset.")
        return

    # Using 0.33 test_size for NN, as in original notebook
    X_train_base, X_test_base, y_train_base, y_test_base = split_data(X, y, test_size=0.33, random_state=42)

    # --- Data Preprocessing for Neural Network ---
    # This will include Scaling, SMOTE, and PCA
    print("\n--- SECTION: Data Preprocessing for Neural Network ---")
    X_train_nn_processed, X_test_nn_processed, y_train_nn_processed, _ = preprocess_for_nn_kmeans(
        X_train_base, X_test_base, y_train_base, n_components_pca=10, random_state=42
    )

    # --- Training Advanced Neural Network ---
    print("\n" + "=" * 50)
    print("--- Running Advanced Neural Network Model ---")
    print("=" * 50)

    # Example configuration for the advanced NN
    # You can modify these parameters to experiment further
    nn_config = {
        'layers_config': [(128, 'relu'), (64, 'relu'), (32, 'relu')],  # More units
        'dropout_rate': 0.4,  # Increased dropout
        'learning_rate': 0.0001,  # Lower learning rate
        'epochs': 100,  # More epochs, relying on EarlyStopping
        'batch_size': 128  # Larger batch size
    }

    train_and_evaluate_advanced_nn(
        X_train_nn_processed, X_test_nn_processed, y_train_nn_processed, y_test_base,
        **nn_config  # Pass the dictionary as keyword arguments
    )

    print("\n==============================================")
    print("Phase 6 Complete: Advanced Neural Network Model Trained and Evaluated!")
    print("You can close any plot windows that appeared.")
    print("==============================================")


if __name__ == "__main__":
    main()
