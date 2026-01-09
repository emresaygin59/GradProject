"""
Sign Language Recognition - Model Training Script

Description:
    This script loads the hand landmark dataset, trains a Multi-Layer Perceptron (MLP)
    Neural Network, and saves the trained model and performance graphs.

Features:
    - Data loading and preprocessing (One-Hot Encoding).
    - MLP Architecture with Dropout for regularization.
    - Early Stopping to prevent overfitting.
    - Visualization of Accuracy/Loss and Confusion Matrix.

Output:
    - Model: models/smart_home_model.h5
    - Labels: models/labels.pkl
    - Graphs: assets/training_graphs.png, assets/confusion_matrix.png

Author: Emre Saygin & Talha Erden
Date: January 2026
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURATION & CONSTANTS ---
# Project Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Input/Output Paths
DATA_FILE = os.path.join(BASE_DIR, 'dataset', 'hand_landmarks.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')  # New folder for images

# Output Files
MODEL_PATH = os.path.join(MODEL_DIR, 'smart_home_model.h5')
LABEL_PATH = os.path.join(MODEL_DIR, 'labels.pkl')
GRAPH_ACC_PATH = os.path.join(ASSETS_DIR, 'training_graphs.png')
GRAPH_CM_PATH = os.path.join(ASSETS_DIR, 'confusion_matrix.png')

# Hyperparameters
TEST_SIZE = 0.2
RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 5  # Early stopping patience

# Visualization Style
plt.style.use('ggplot')


def create_directories():
    """Creates necessary output directories if they don't exist."""
    for directory in [MODEL_DIR, ASSETS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[INFO] Created directory: {directory}")


def load_and_preprocess_data():
    """Loads CSV data, encodes labels, and splits into train/val sets."""
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Dataset not found at: {DATA_FILE}")
        print("Please run 'collect_landmarks.py' first.")
        exit(1)

    print("[INFO] Loading dataset...")
    # Load dataset (assuming no header in CSV)
    df = pd.read_csv(DATA_FILE, header=None)

    # Split Features (X) and Labels (y)
    X = df.iloc[:, 1:].values  # Columns 1 to end (Landmark coordinates)
    y = df.iloc[:, 0].values  # Column 0 (Labels)

    print(f"[INFO] Dataset Loaded. Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # Encode string labels to integers (e.g., 'LIGHT_ON' -> 0)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Save label mappings for inference script
    with open(LABEL_PATH, 'wb') as f:
        pickle.dump(label_encoder.classes_, f)
    print(f"[INFO] Labels saved to: {LABEL_PATH}")

    # Split into Train and Validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, label_encoder


def build_model(input_shape, num_classes):
    """Defines the MLP Neural Network architecture."""
    model = Sequential([
        Input(shape=(input_shape,)),  # Input Layer (63 features)
        Dense(128, activation='relu'),  # Hidden Layer 1: Feature extraction
        Dropout(0.3),  # Regularization
        Dense(64, activation='relu'),  # Hidden Layer 2: Refinement
        Dropout(0.2),  # Regularization
        Dense(num_classes, activation='softmax')  # Output Layer
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def save_performance_plots(history, model, X_test, y_test, label_encoder):
    """Generates and saves accuracy/loss graphs and confusion matrix."""
    print("[INFO] Generating performance graphs...")

    # --- 1. Accuracy & Loss Plot ---
    plt.figure(figsize=(14, 6))

    # Subplot: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, linestyle='--')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Subplot: Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(GRAPH_ACC_PATH, dpi=300)
    print(f" -> Saved Graph: {GRAPH_ACC_PATH}")

    # --- 2. Confusion Matrix ---
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(GRAPH_CM_PATH, dpi=300)
    print(f" -> Saved Matrix: {GRAPH_CM_PATH}")


def main():
    print(">>> Training Process Started...")
    create_directories()

    # 1. Prepare Data
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()

    # 2. Build Model
    model = build_model(input_shape=X_train.shape[1], num_classes=len(label_encoder.classes_))

    # 3. Train Model
    print("[INFO] Starting training...")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    # 4. Save Model
    model.save(MODEL_PATH)
    print(f"[SUCCESS] Model saved to '{MODEL_PATH}'")

    # 5. Evaluate & Visualize
    save_performance_plots(history, model, X_test, y_test, label_encoder)

    print("\n>>> [DONE] Training pipeline completed successfully.")


if __name__ == "__main__":
    main()