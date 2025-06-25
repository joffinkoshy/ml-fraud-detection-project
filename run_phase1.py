import os
import sys
import pandas as pd
import warnings

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import data loading and splitting functions
from data_preprocessing import load_and_explore_data, split_data

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def main():
    """
    Main function for Phase 1: Project Setup and Data Loading.
    Loads data, explores it, and performs basic train-test split.
    """
    print("==============================================")
    print("   Phase 1: Project Setup and Data Loading")
    print("==============================================")

    # Define the path to the dataset
    data_file_path = os.path.join(current_dir, 'data', 'creditcard.csv')

    # Load and explore data
    df, X, y, frauds, non_frauds = load_and_explore_data(data_file_path)

    if df is None:
        print("\nPhase 1 failed: Data not loaded. Please check file path and dataset.")
        return

    # Perform basic train-test split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.35, random_state=42)

    print("\n==============================================")
    print("Phase 1 Complete: Data loaded, explored, and split successfully!")
    print("You can close any plot windows that appeared.")
    print("==============================================")


if __name__ == "__main__":
    main()
