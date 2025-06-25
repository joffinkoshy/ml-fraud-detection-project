import os
import sys
import warnings

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import data loading and splitting functions
from data_preprocessing import load_and_explore_data, split_data

# Import Logistic Regression model functions (tuned with pipeline)
from logistic_regression_model import tune_and_evaluate_lr_with_cv

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def main():
    """
    Main function for Phase 8: Logistic Regression - SMOTE in Pipeline & Advanced Tuning.
    Loads data, splits it, and trains/evaluates the LR model with SMOTE integrated into GridSearchCV.
    """
    print("==============================================")
    print("   Phase 8: LR - SMOTE in Pipeline & Tuning")
    print("==============================================")

    # --- Data Loading and Splitting ---
    print("\n--- SECTION: Data Loading and Splitting ---")
    data_file_path = os.path.join(current_dir, 'data', 'creditcard.csv')
    df, X, y, _, _ = load_and_explore_data(data_file_path)

    if df is None:
        print("\nPhase 8 aborted: Data not loaded. Please check file path and dataset.")
        return

    # Using 0.35 test_size for LR as in previous phases
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = split_data(X, y, test_size=0.35, random_state=42)

    # --- Running Tuned Logistic Regression Model with SMOTE in Pipeline ---
    print("\n" + "=" * 50)
    print("--- Running Tuned Logistic Regression Model (SMOTE in Pipeline) ---")
    print("=" * 50)

    tune_and_evaluate_lr_with_cv(X_train_lr, X_test_lr, y_train_lr, y_test_lr)

    print("\n==============================================")
    print("Phase 8 Complete: Tuned LR Pipeline Model Trained and Evaluated!")
    print("All plots saved to the 'plots/' directory if generated.")
    print("==============================================")


if __name__ == "__main__":
    main()
