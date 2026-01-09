"""
Sign Language Dataset Generator

Description:
    This script iterates through raw image folders (data/), detects hand landmarks
    using MediaPipe, and saves the extracted 3D coordinates to a CSV file.
    This CSV file is then used to train the Neural Network.

Features:
    - Automatically traverses label directories.
    - Extracts 21 landmarks (x, y, z) per image (Total 63 features).
    - Ignores non-image files and handles read errors gracefully.

Output:
    - dataset/hand_landmarks.csv

Author: Emre Saygin & Talha Erden
Date: January 2026
"""

import os
import cv2
import mediapipe as mp
import csv
import sys

# --- CONFIGURATION & PATHS ---
# Determine project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Input/Output Directories
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data')  # Folder containing images
OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset')  # Folder for CSV output
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'hand_landmarks.csv')

# MediaPipe Settings
MP_STATIC_MODE = True
MP_MAX_HANDS = 1
MP_MIN_CONFIDENCE = 0.5
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def setup_directories():
    """Checks input directory and creates output directory."""
    if not os.path.exists(RAW_DATA_DIR):
        print(f"[ERROR] Raw data directory not found at: {RAW_DATA_DIR}")
        print("Please create a 'data' folder and put your image classes inside.")
        sys.exit(1)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Created output directory: {OUTPUT_DIR}")


def extract_landmarks(image_path, hands_processor):
    """
    Reads an image and extracts hand landmarks using MediaPipe.
    Returns: List of 63 coordinates [x1, y1, z1, ...] or None if failed.
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert BGR to RGB (MediaPipe requirement)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        results = hands_processor.process(image_rgb)

        if results.multi_hand_landmarks:
            # Get the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]

            # Flatten coordinates: [x1, y1, z1, x2, y2, z2, ...]
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            return landmark_list

    except Exception as e:
        print(f"[WARNING] Skipping corrupt image {os.path.basename(image_path)}: {e}")

    return None


def main():
    print(">>> [INIT] Dataset Generation Started...")
    setup_directories()

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=MP_STATIC_MODE,
        max_num_hands=MP_MAX_HANDS,
        min_detection_confidence=MP_MIN_CONFIDENCE
    )

    total_samples = 0

    # Open CSV file for writing
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Iterate through each label folder in 'data/'
        labels = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

        if not labels:
            print("[WARNING] No class folders found in 'data/' directory.")
            return

        for label in labels:
            label_path = os.path.join(RAW_DATA_DIR, label)
            print(f">>> [PROCESSING] Class: '{label}'...")

            class_count = 0

            for img_name in os.listdir(label_path):
                # Filter only image files
                if not img_name.lower().endswith(VALID_EXTENSIONS):
                    continue

                img_path = os.path.join(label_path, img_name)

                # Extract features
                landmarks = extract_landmarks(img_path, hands)

                if landmarks:
                    # Write row: [LABEL, features...]
                    writer.writerow([label] + landmarks)
                    class_count += 1
                    total_samples += 1

            print(f"    -> Added {class_count} samples.")

    hands.close()
    print(f"\n>>> [SUCCESS] Dataset creation complete.")
    print(f"    -> Total Samples: {total_samples}")
    print(f"    -> Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()