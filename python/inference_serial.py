"""
Sign Language Based Smart Home Control System
Inference Module

Description:
    This script captures video from the webcam, detects hand landmarks using MediaPipe,
    classifies the gesture using a trained ANN model, and sends commands to an Arduino
    via Serial communication.

Author: Emre Saygin & Talha Erden
Date: January 2026
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import pickle
import sys
from tensorflow.keras.models import load_model
from typing import Tuple, Any

# --- CONFIGURATION & CONSTANTS ---
# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'smart_home_model.h5')
LABEL_PATH = os.path.join(BASE_DIR, 'models', 'labels.pkl')

# Serial Settings
COM_PORT = 'COM3'  # NOTE: Users must update this port based on their system
BAUD_RATE = 9600
SERIAL_TIMEOUT = 1

# Detection Settings
CONFIDENCE_THRESHOLD = 0.85  # Increased slightly for better precision

# Colors (BGR Format)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)

# Command Mapping (Label -> Arduino Signal)
COMMAND_MAP = {
    "LIGHT_ON": b'1',
    "LIGHT_OFF": b'0',
    "FAN_ON": b'3',
    "FAN_OFF": b'2'
}


def setup_serial_connection(port: str, baud: int) -> serial.Serial:
    """Establishes a serial connection with the Arduino."""
    try:
        ser = serial.Serial(port, baud, timeout=SERIAL_TIMEOUT)
        time.sleep(2)  # Wait for Arduino to reset
        print(f"[SUCCESS] Connected to Arduino on {port}.")
        return ser
    except serial.SerialException as e:
        print(f"[WARNING] Arduino not found on {port}. Simulation mode enabled.")
        print(f"Details: {e}")
        return None


def load_resources() -> Tuple[Any, list]:
    """Loads the trained Keras model and label encoder."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        print(f"[CRITICAL ERROR] Model files not found at: {BASE_DIR}/models/")
        print("Please run 'train_model.py' first.")
        sys.exit(1)

    try:
        model = load_model(MODEL_PATH)
        with open(LABEL_PATH, 'rb') as f:
            classes = pickle.load(f)
        print("[SUCCESS] Model and labels loaded successfully.")
        return model, classes
    except Exception as e:
        print(f"[ERROR] Failed to load resources: {e}")
        sys.exit(1)


def draw_ui(frame, text: str, color: Tuple[int, int, int]):
    """Draws the prediction text and background on the frame."""
    # Background rectangle for text
    cv2.rectangle(frame, (0, 0), (350, 60), COLOR_BLACK, -1)
    # Status Text
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # Instruction Text
    cv2.putText(frame, "Press 'q' to quit", (10, 470), cv2.FONT_HERSHEY_PLAIN, 1, COLOR_TEXT, 1)


def main():
    # 1. Initialize Resources
    model, classes = load_resources()
    ser = setup_serial_connection(COM_PORT, BAUD_RATE)

    # 2. Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    # 3. Start Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible.")
        return

    print("[INFO] System Ready. Starting main loop...")
    last_sent_command = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        # Mirror frame and convert to RGB
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Visual: Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLOR_GREEN, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=COLOR_RED, thickness=2, circle_radius=2)
                )

                # Data: Extract coordinates
                lm_list = []
                for lm in hand_landmarks.landmark:
                    lm_list.extend([lm.x, lm.y, lm.z])

                # AI: Predict Gesture
                prediction = model.predict(np.array([lm_list]), verbose=0)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)
                label = classes[class_index]

                # UI: Determine color based on command type (ON=Green, OFF=Red)
                status_color = COLOR_GREEN if "ON" in label else COLOR_RED
                display_text = f"{label} ({int(confidence * 100)}%)"

                # Logic: Serial Communication
                if confidence > CONFIDENCE_THRESHOLD:
                    # Get corresponding Arduino command from dictionary
                    current_command = COMMAND_MAP.get(label)

                    if current_command and current_command != last_sent_command:
                        if ser:
                            try:
                                ser.write(current_command)
                                print(f"[ACTION] Command Sent: {label} -> {current_command}")
                            except serial.SerialException as e:
                                print(f"[ERROR] Serial Write Failed: {e}")

                        last_sent_command = current_command
                else:
                    # If confidence is low, show as Uncertain
                    display_text = "..."
                    status_color = (128, 128, 128)  # Gray

                draw_ui(frame, display_text, status_color)

        cv2.imshow('Smart Home Control System (Emre Saygin)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()
    print("[INFO] System Terminated Safely.")


if __name__ == "__main__":
    main()