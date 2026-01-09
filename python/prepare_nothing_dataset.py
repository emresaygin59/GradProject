"""
Dataset Augmentation Script - 'NOTHING' Class Generator

Description:
    This script generates a negative class ('NOTHING' or Background) for the
    gesture recognition model. It does this by sampling random images from
    unused classes in the original ASL dataset (e.g., dynamic gestures like J, Z
    or unused letters) to create a diverse background set.

Features:
    - Automatically balances the dataset to a target image count.
    - Prevents class overlap by excluding active command classes.
    - Renames files to avoid collisions.

Author: Emre Saygin & Talha Erden
Date: January 2026
"""

import os
import random
import shutil
import sys

# --- CONFIGURATION ---
# Path Management
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# NOTE: The source dataset path usually depends on where the user downloaded the Kaggle dataset.
# We assume it is located in the project root for portability, but this can be changed.
SOURCE_DATASET_DIR = os.path.join(BASE_DIR, "asl_dataset", "asl_alphabet_train", "asl_alphabet_train")
PROJECT_DATA_DIR = os.path.join(BASE_DIR, 'data')

TARGET_CLASS_NAME = "NOTHING"
TARGET_IMAGE_COUNT = 8500

# Classes to EXCLUDE (These are the gestures we actually use for commands)
# We don't want to accidentally put a 'Light On' gesture into the 'Nothing' class.
EXCLUDED_CLASSES = ["A", "B", "C", "D", "E", "F"]  # Update this list based on your actual command classes


def validate_paths():
    """Checks if source and destination directories are valid."""
    if not os.path.exists(SOURCE_DATASET_DIR):
        print(f"[ERROR] Source dataset not found at: {SOURCE_DATASET_DIR}")
        print("Please download the ASL dataset and update 'SOURCE_DATASET_DIR' in the script.")
        sys.exit(1)

    destination_dir = os.path.join(PROJECT_DATA_DIR, TARGET_CLASS_NAME)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"[INFO] Created destination directory: {destination_dir}")

    return destination_dir


def get_valid_source_folders():
    """Returns a list of folders from the source dataset that are NOT in the excluded list."""
    all_folders = [f for f in os.listdir(SOURCE_DATASET_DIR)
                   if os.path.isdir(os.path.join(SOURCE_DATASET_DIR, f))]

    valid_folders = [f for f in all_folders if f not in EXCLUDED_CLASSES]

    if not valid_folders:
        print("[ERROR] No valid source folders found. Check your EXCLUDED_CLASSES list.")
        sys.exit(1)

    return valid_folders


def generate_dataset():
    print(">>> [INIT] Generating 'NOTHING' Class Dataset...")

    destination_dir = validate_paths()
    source_folders = get_valid_source_folders()

    # Calculate how many images to take from each folder to maintain balance
    images_per_folder = TARGET_IMAGE_COUNT // len(source_folders)

    print(f"[INFO] Source Folders Found: {len(source_folders)}")
    print(f"[INFO] Sampling Plan: ~{images_per_folder} images per folder.")

    total_copied = 0

    for folder_name in source_folders:
        source_path = os.path.join(SOURCE_DATASET_DIR, folder_name)

        try:
            all_images = os.listdir(source_path)
        except Exception as e:
            print(f"[WARNING] Could not access folder '{folder_name}': {e}")
            continue

        # Logic: If folder has enough images, sample randomly. If not, take all.
        if len(all_images) >= images_per_folder:
            selected_images = random.sample(all_images, images_per_folder)
        else:
            selected_images = all_images

        # Copy process
        for img_name in selected_images:
            src_file = os.path.join(source_path, img_name)

            # Rename: Prefix with folder name to ensure uniqueness (e.g., 'SPACE_image1.jpg')
            new_name = f"{folder_name}_{img_name}"
            dst_file = os.path.join(destination_dir, new_name)

            shutil.copy(src_file, dst_file)
            total_copied += 1

        print(f"    -> Copied {len(selected_images)} samples from '{folder_name}'")

    print("-" * 40)
    print(f"[SUCCESS] Generation Complete.")
    print(f"          Total Images in '{TARGET_CLASS_NAME}': {total_copied}")
    print(f"          Location: {destination_dir}")


if __name__ == "__main__":
    generate_dataset()