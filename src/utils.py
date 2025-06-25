import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings

# Suppress general warnings that might arise from older versions of other packages
warnings.filterwarnings('ignore')


def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates the model's performance focusing on False Negative Rate (FNR).
    Prints key metrics and plots the normalized confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        model_name (str): Name of the model for display.
    """
    print(f"\n--- Evaluation for {model_name} ---")

    # Calculate confusion matrix using scikit-learn
    cm = confusion_matrix(y_true, y_pred)

    # Extracting TN, FP, FN, TP from the confusion matrix
    # For binary classification, cm is typically:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    print("Confusion Matrix:")
    print(f"[[{tn}, {fp}],")
    print(f" [{fn}, {tp}]]")

    total_test_points = len(y_true)

    # Calculate common metrics directly using sklearn.metrics or extracted values
    accuracy = accuracy_score(y_true, y_pred)

    # Precision: TP / (TP + FP)
    precision = precision_score(y_true, y_pred, zero_division=0)  # zero_division=0 handles cases where TP+FP is 0

    # Recall (Sensitivity/TPR): TP / (TP + FN)
    recall = recall_score(y_true, y_pred, zero_division=0)  # zero_division=0 handles cases where TP+FN is 0

    # F1-Score
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # False Negative Rate (FNR): proportion of actual positives (frauds) that were missed
    # FNR = FN / (FN + TP) -- this is (1 - Recall) for the positive class
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"True Positives (TP): {tp}")
    print(f"False Negatives (FN): {fn}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negative Rate (FNR): {fnr:.4f} (Lower is better for fraud detection)")
    print(f"Precision (Positive Class): {precision:.4f}")
    print(f"Recall (Sensitivity/TPR - Positive Class): {recall:.4f}")
    print(f"F1-Score (Positive Class): {f1:.4f}")

    # Plot normalized confusion matrix
    # Use seaborn if available for nicer plots, otherwise matplotlib's imshow
    try:
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Normalized Confusion Matrix for {model_name}')
        plt.show()
    except ImportError:
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['0', '1'], rotation=45)
        plt.yticks(tick_marks, ['0', '1'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # Add text annotations manually
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.show()


def plot_feature_distribution(df, frauds, non_frauds, feature_name):
    """
    Plots the distribution of a given feature with respect to the Class.

    Args:
        df (pd.DataFrame): The full dataframe.
        frauds (pd.DataFrame): Sub-dataframe containing only fraudulent transactions.
        non_frauds (pd.DataFrame): Sub-dataframe containing only non-fraudulent transactions.
        feature_name (str): The name of the feature to plot.
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

    ax1.scatter(frauds[feature_name], frauds['Class'], color='orange', label='Fraud', alpha=0.6)
    ax1.scatter(non_frauds[feature_name], non_frauds['Class'], color='blue', label='Normal', alpha=0.1)
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel('Class (0=Normal, 1=Fraud)')
    ax1.set_title(f'{feature_name} Distribution vs Class (Overall)')
    ax1.legend()

    ax2.scatter(frauds[feature_name], frauds['Class'], color='orange', label='Fraud', alpha=0.8)
    ax2.set_xlabel(feature_name)
    ax2.set_ylabel('Class (0=Normal, 1=Fraud)')
    ax2.set_title(f'{feature_name} Distribution vs Class (Zoomed on Fraud)')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    print(f"Visualization of '{feature_name}' distribution. Note the different patterns for fraud.")

