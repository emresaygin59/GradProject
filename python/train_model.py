import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# --- PATH SETTINGS ---
# Determine the absolute path of the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Define paths
DATA_FILE = os.path.join(BASE_DIR, 'dataset', 'hand_landmarks.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'smart_home_model.h5')
LABEL_FILE = os.path.join(MODEL_DIR, 'labels.pkl')

# Graph Style (Academic vision)
plt.style.use('ggplot') 

def train():
    print(">>> TRAIN PROCESS STARTED... (Generating Professional Graphs)")
    
    # Ensure models directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Data Loading
    try:
        df = pd.read_csv(DATA_FILE, header=None)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {DATA_FILE}.\nPlease run collect_landmarks.py first.")
        return

    X = df.iloc[:, 1:].values  # Coordinates
    y = df.iloc[:, 0].values   # Labels

    # 2. Labelling (Encoding)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Save the labels (for inference use)
    with open(LABEL_FILE, 'wb') as f:
        pickle.dump(le.classes_, f)

    # 3. Data Splitting (80% Train, 20% Validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # 4. Model Architecture (ANN / MLP)
    model = Sequential([
        Input(shape=(63,)),              # 21 points * 3 (x,y,z)
        Dense(128, activation='relu'),   # Hidden Layer 1
        Dropout(0.2),                    # Overfitting preventer
        Dense(64, activation='relu'),    # Hidden Layer 2
        Dense(len(le.classes_), activation='softmax') # Output Layer
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training Model...")
    # Start the training and save history
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    model.save(MODEL_FILE)
    print(f"\nSUCCESS: Model saved to '{MODEL_FILE}'.")

    # --- 5. GRAPH GENERATION ---
    print("Generating graphs...")
    
    # A) Accuracy & Loss Graph
    plt.figure(figsize=(14, 6))

    # Accuracy Subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, linestyle='--')
    plt.title('Model Accuracy over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Loss Subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
    plt.title('Model Loss over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Cross-Entropy)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_graphs.png', dpi=300) 
    print("-> 'training_graphs.png' saved.")

    # B) Confusion Matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                annot_kws={"size": 12})
    
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("-> 'confusion_matrix.png' saved.")
    
    print("\nDONE! Check your folder for the PNG files.")

if __name__ == "__main__":
    train()