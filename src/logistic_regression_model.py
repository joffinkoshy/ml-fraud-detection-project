import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score
from imblearn.pipeline import Pipeline  # IMPORTANT CHANGE: Use imblearn's Pipeline

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from src.utils import evaluate_model


def train_and_evaluate_vanilla_lr(X_train, X_test, y_train, y_test, C=1e5, random_state=42):
    """
    Trains and evaluates a vanilla Logistic Regression model.
    """
    print("\n--- Training Vanilla Logistic Regression ---")
    logistic_vanilla = LogisticRegression(C=C, solver='liblinear', max_iter=1000, random_state=random_state)
    print("Fitting model...")
    logistic_vanilla.fit(X_train, y_train)

    y_pred = logistic_vanilla.predict(X_test)
    evaluate_model(y_test, y_pred, "Vanilla Logistic Regression")


def train_and_evaluate_smote_lr(X_train_base, X_test_base, y_train_base, y_test_base, C=1e5, random_state=42):
    """
    Performs scaling, SMOTE oversampling on training data, then trains and evaluates
    a Logistic Regression model.
    """
    print("\n--- Training Logistic Regression with SMOTE and Scaling ---")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_base)
    X_test_scaled = scaler.transform(X_test_base)
    print("Data scaled.")

    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train_base)
    print(f"SMOTE applied. Training set size BEFORE SMOTE: {len(X_train_base)} -> AFTER SMOTE: {len(X_train_smote)}")

    logistic_smote = LogisticRegression(C=C, solver='liblinear', max_iter=1000, random_state=random_state)
    print("Fitting model...")
    logistic_smote.fit(X_train_smote, y_train_smote)

    y_pred = logistic_smote.predict(X_test_scaled)
    evaluate_model(y_test_base, y_pred, "Logistic Regression (SMOTE & Scaled)")


def train_and_evaluate_balanced_lr(X_train, X_test, y_train, y_test, C=1e5, random_state=42):
    """
    Trains and evaluates a Logistic Regression model with 'balanced' class weights.
    """
    print("\n--- Training Logistic Regression with Balanced Class Weights ---")
    logistic_balanced = LogisticRegression(C=C, solver='liblinear', max_iter=1000, class_weight='balanced',
                                           random_state=random_state)
    print("Fitting model...")
    logistic_balanced.fit(X_train, y_train)

    y_pred = logistic_balanced.predict(X_test)
    evaluate_model(y_test, y_pred, "Logistic Regression (Balanced Weights)")


def tune_and_evaluate_lr_with_cv(X_train, X_test, y_train, y_test, random_state=42):
    """
    Performs hyperparameter tuning for Logistic Regression using GridSearchCV with
    cross-validation, optimizing for recall (minimizing FNR), within a pipeline
    that includes scaling and SMOTE.

    Args:
        X_train (pd.DataFrame): Training features (original).
        X_test (pd.DataFrame): Test features (original).
        y_train (pd.Series): Training target (original).
        y_test (pd.Series): Test target (original).
        random_state (int): Seed for reproducibility.

    Returns:
        object: The best trained Pipeline model.
    """
    print("\n--- Hyperparameter Tuning Logistic Regression with Cross-Validation (SMOTE in Pipeline) ---")
    print("Optimizing for Recall (higher recall = lower FNR)...")

    # Create a pipeline using imblearn.pipeline.Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=random_state, k_neighbors=5)),
        ('logisticregression', LogisticRegression(random_state=random_state, max_iter=1000))
    ])

    # Define the parameter grid for the pipeline steps
    param_grid = {
        'logisticregression__C': [0.01, 0.1, 1, 10, 100, 1000],
        'logisticregression__solver': ['liblinear']
    }

    scorer = make_scorer(recall_score, pos_label=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        verbose=2,
        n_jobs=-1
    )

    print("Starting GridSearchCV fit (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)

    print("\nGridSearchCV Results (SMOTE in Pipeline):")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score (Recall): {grid_search.best_score_:.4f}")

    best_pipeline_model = grid_search.best_estimator_

    print("\nEvaluating best Pipeline model on test set:")
    y_pred = best_pipeline_model.predict(X_test)
    evaluate_model(y_test, y_pred,
                   f"Tuned LR Pipeline (Best C={best_pipeline_model.named_steps['logisticregression'].C:.2f})")

    return best_pipeline_model

