import os
import sys
import warnings

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import data loading and splitting functions from data_preprocessing.py
from data_preprocessing import load_and_explore_data, split_data

# Import Logistic Regression model functions
from logistic_regression_model import train_and_evaluate_vanilla_lr, train_and_evaluate_smote_lr, \
    train_and_evaluate_balanced_lr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def main():
    """
    Main function for Phase 2: Logistic Regression Model Implementation.
    Loads data, splits it, and trains/evaluates different LR models.
    """
    print("==============================================")
    print("   Phase 2: Logistic Regression Models")
    print("==============================================")

    # --- Data Loading and Splitting (reusing Phase 1 logic) ---
    print("\n--- SECTION: Data Loading and Splitting ---")
    data_file_path = os.path.join(current_dir, 'data', 'creditcard.csv')
    df, X, y, _, _ = load_and_explore_data(data_file_path)

    if df is None:
        print("\nPhase 2 aborted: Data not loaded. Please check file path and dataset.")
        return

    X_train_base, X_test_base, y_train_base, y_test_base = split_data(X, y, test_size=0.35, random_state=42)

    # --- Logistic Regression Models ---
    print("\n" + "=" * 50)
    print("--- Running Logistic Regression Models ---")
    print("=" * 50)

    # --- Vanilla Logistic Regression ---
    train_and_evaluate_vanilla_lr(X_train_base, X_test_base, y_train_base, y_test_base)

    # --- Logistic Regression with SMOTE Over-sampling and Scaling ---
    # Note: The SMOTE and Scaling logic is now encapsulated within train_and_evaluate_smote_lr itself
    train_and_evaluate_smote_lr(X_train_base, X_test_base, y_train_base, y_test_base)

    # --- Logistic Regression with Balanced Class Weights ---
    train_and_evaluate_balanced_lr(X_train_base, X_test_base, y_train_base, y_test_base)

    print("\n==============================================")
    print("Phase 2 Complete: Logistic Regression Models Trained and Evaluated!")
    print("You can close any plot windows that appeared.")
    print("==============================================")


if __name__ == "__main__":
    main()
