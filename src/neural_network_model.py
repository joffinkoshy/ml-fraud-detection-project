import matplotlib;

matplotlib.use('Agg')  # IMPORTANT: Force non-interactive backend for saving plots
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf  # NEW: Import tensorflow for custom loss
from sklearn.utils import class_weight  # NEW: To calculate class weights

from src.utils import evaluate_model


# Existing function (train_and_evaluate_nn) is kept for completeness
def train_and_evaluate_nn(X_train_pca, X_test_pca, y_train_smote, y_test, random_state=42):
    """
    Trains and evaluates a Neural Network model with a fixed architecture.
    """
    print("\n--- Training Neural Network Model (Fixed Architecture) ---")

    model_nn = Sequential([
        Dense(30, input_dim=X_train_pca.shape[1], activation='relu'),
        Dense(27, activation='relu'),
        Dropout(0.2),
        Dense(20, activation='relu'),
        Dense(15, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model_nn.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    print("\nNeural Network Model Summary (Fixed Architecture):")
    model_nn.summary()

    history = model_nn.fit(
        X_train_pca, y_train_smote,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    y_pred_proba = model_nn.predict(X_test_pca)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    evaluate_model(y_test, y_pred, "Neural Network (Fixed Architecture, SMOTE, Scaling & PCA)")

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Neural Network Model Accuracy (Fixed Architecture)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show(block=True)  # Ensure this blocks
    plt.close()  # Explicitly close

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Neural Network Model Loss (Fixed Architecture)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show(block=True)  # Ensure this blocks
    plt.close()  # Explicitly close


# --- NEW: Custom Weighted Binary Cross-Entropy Loss function ---
def weighted_binary_crossentropy(y_true, y_pred, class_weights):
    """
    Custom binary cross-entropy loss function with dynamic class weights.
    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted probabilities.
        class_weights (dict): Dictionary of weights for each class (0 and 1).
                              e.g., {0: weight_for_class_0, 1: weight_for_class_1}
    Returns:
        tf.Tensor: Weighted binary cross-entropy loss.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Get weights for each sample based on its true class
    # If y_true is 1, use class_weights[1]; if y_true is 0, use class_weights[0]
    weights = y_true * class_weights[1] + (1.0 - y_true) * class_weights[0]

    # Calculate binary cross-entropy for each sample
    bce_loss = tf.keras.backend.binary_crossentropy(y_true, y_pred)

    # Apply weights to the loss and compute mean
    weighted_bce_loss = tf.reduce_mean(weights * bce_loss)
    return weighted_bce_loss


# --- Updated Advanced Neural Network Training with Custom Weighted Loss ---
def train_and_evaluate_advanced_nn(X_train_pca, X_test_pca, y_train_smote, y_test,
                                   layers_config=[(64, 'relu'), (32, 'relu'), (16, 'relu')],
                                   dropout_rate=0.4, learning_rate=0.0001,
                                   epochs=50, batch_size=64, random_state=42):
    """
    Trains and evaluates an advanced Neural Network model with configurable layers,
    dropout, and using EarlyStopping, ReduceLROnPlateau, and a Custom Weighted Loss.
    """
    print("\n--- Training Advanced Neural Network Model with Custom Weighted Loss ---")
    print(
        f"Configuration: Layers={layers_config}, Dropout={dropout_rate}, LR={learning_rate}, Epochs={epochs}, BatchSize={batch_size}")

    # Calculate class weights for the custom loss function
    # These weights are applied *in the loss function* during training.
    # This gives more weight to the minority class (fraud) for the loss calculation.
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',  # Balanced weights based on class frequencies
        classes=np.unique(y_train_smote),
        y=y_train_smote  # Use the SMOTE-augmented y_train to compute weights
    )
    # Convert to a dictionary with class labels as keys
    class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}
    print(f"Calculated Class Weights for Loss: {class_weights}")

    model_nn = Sequential()
    # Input layer
    model_nn.add(Dense(layers_config[0][0], input_dim=X_train_pca.shape[1], activation=layers_config[0][1]))

    # Hidden layers
    for units, activation in layers_config[1:]:
        model_nn.add(Dense(units, activation=activation))
        if dropout_rate > 0:
            model_nn.add(Dropout(dropout_rate))

    # Output layer
    model_nn.add(Dense(1, activation='sigmoid'))

    # Compile the model with the custom weighted_binary_crossentropy loss
    model_nn.compile(loss=lambda y_true, y_pred: weighted_binary_crossentropy(y_true, y_pred, class_weights),
                     optimizer=Adam(learning_rate=learning_rate),
                     metrics=['accuracy'])

    print("\nAdvanced Neural Network Model Summary:")
    model_nn.summary()

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,  # Reduced patience for faster debugging
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,  # Reduced patience for faster debugging
        min_lr=0.000005,
        verbose=1
    )

    callbacks = [early_stopping, reduce_lr]

    # Train the model
    print("\nFitting Advanced Neural Network model...")
    history = model_nn.fit(
        X_train_pca, y_train_smote,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate the model on the unseen test set
    loss, accuracy = model_nn.evaluate(X_test_pca, y_test, verbose=0)
    print(f"\nAdvanced Neural Network Test Loss: {loss:.4f}")
    print(f"Advanced Neural Network Test Accuracy: {accuracy:.4f}")

    # Make predictions (probabilities) on the test set
    y_pred_proba = model_nn.predict(X_test_pca)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Evaluate using the custom evaluation function
    evaluate_model(y_test, y_pred, "Advanced Neural Network (Custom Weighted Loss, SMOTE, Scaling & PCA)")

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Advanced Neural Network Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('plots/advanced_nn_accuracy.png')  # Save plot instead of showing
    plt.close()  # Close figure

    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Advanced Neural Network Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('plots/advanced_nn_loss.png')  # Save plot instead of showing
    plt.close()  # Close figure

