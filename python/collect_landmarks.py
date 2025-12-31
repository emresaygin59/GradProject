import os
import cv2
import mediapipe as mp
import csv

# --- PATH SETTINGS ---
# Determine the absolute path of the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Define paths relative to the project root
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'hand_landmarks.csv')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def create_dataset():
    # Check if raw data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Raw data directory '{DATA_DIR}' not found!")
        return

    # Create dataset directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)

        # Iterate through label folders (e.g., LIGHT_ON, LIGHT_OFF, etc.)
        for label in os.listdir(DATA_DIR):
            label_path = os.path.join(DATA_DIR, label)
            if not os.path.isdir(label_path):
                continue

            print(f"Processing: {label}...")
            image_count = 0

            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)

                # Prevent image reading errors
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # 21 landmarks * 3 coordinates (x, y, z) = 63 features
                            landmark_list = []
                            for lm in hand_landmarks.landmark:
                                landmark_list.extend([lm.x, lm.y, lm.z])

                            # Write row: [LABEL, x1, y1, z1, x2, ...]
                            writer.writerow([label] + landmark_list)
                            image_count += 1

                except Exception as e:
                    print(f"Error ({img_name}): {e}")

            print(f" -> {label}: {image_count} samples added.")

    print(f"\nSUCCESS: Data saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    create_dataset()