import os
import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import pickle
from tensorflow.keras.models import load_model

# --- PATH SETTINGS ---
# Determine the absolute path of the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Define paths for models
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'smart_home_model.h5')
LABEL_PATH = os.path.join(BASE_DIR, 'models', 'labels.pkl')

COM_PORT = 'COM3'   # Change this to your Arduino port
BAUD_RATE = 9600

# --- ARDUINO CONNECTION ---
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino reset
    print(f"SUCCESS: Arduino connected via {COM_PORT}.")
except:
    ser = None
    print("WARNING: Arduino not connected! Running in simulation mode (Display only).")

# --- LOAD MODEL & LABELS ---
print("Loading model and labels...")
try:
    model = load_model(MODEL_PATH)
    with open(LABEL_PATH, 'rb') as f:
        classes = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR: Could not load model or labels. {e}")
    exit()

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)
last_command = ""

print("System is ready. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image (mirror effect) and convert to RGB
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on screen
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # Prepare landmark data for the model
            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.extend([lm.x, lm.y, lm.z])

            # Make prediction
            prediction = model.predict(np.array([lm_list]), verbose=0)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)
            label = classes[class_index]

            # Display prediction on screen
            text = f"{label} ({int(confidence * 100)}%)"
            
            # Determine color based on state (Green for ON, Red for OFF)
            color = (0, 255, 0) if "ON" in label else (0, 0, 255)

            cv2.rectangle(frame, (0, 0), (350, 60), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # --- SEND COMMAND TO ARDUINO ---
            # Send command if confidence > 80%
            if confidence > 0.8:
                command_to_send = None
                
                if label == "LIGHT_ON": command_to_send = b'1'
                elif label == "LIGHT_OFF": command_to_send = b'0'
                elif label == "FAN_ON": command_to_send = b'3'
                elif label == "FAN_OFF": command_to_send = b'2'
                
                # Send only if command changed to avoid flooding serial
                if command_to_send and ser:
                    try:
                        ser.write(command_to_send)
                    except Exception as e:
                        print(f"Serial Error: {e}")

    cv2.imshow('Smart Home Control System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()