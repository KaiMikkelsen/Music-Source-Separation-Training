import os
import shutil
import random

def split_and_copy_folders(source_base_path, train_dest, validation_dest, train_count, validation_count):
    """
    Randomly splits folders from specified source directories and copies them
    to training and validation destination directories.

    Args:
        source_base_path (str): The base path containing 'fold_2' and 'fold_3'.
        train_dest (str): The destination directory for training folders.
        validation_dest (str): The destination directory for validation folders.
        train_count (int): The number of folders to copy to the training directory.
        validation_count (int): The number of folders to copy to the validation directory.
    """
    source_fold_2 = os.path.join(source_base_path, "fold_2")
    source_fold_3 = os.path.join(source_base_path, "fold_3")

    # Ensure destination directories exist
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(validation_dest, exist_ok=True)
    print(f"Ensured '{train_dest}' and '{validation_dest}' exist.")

    all_folders = []

    # Get folders from fold_2
    if os.path.exists(source_fold_2):
        for item in os.listdir(source_fold_2):
            item_path = os.path.join(source_fold_2, item)
            if os.path.isdir(item_path):
                all_folders.append(item_path)
    else:
        print(f"Warning: Source directory '{source_fold_2}' does not exist.")

    # Get folders from fold_3
    if os.path.exists(source_fold_3):
        for item in os.listdir(source_fold_3):
            item_path = os.path.join(source_fold_3, item)
            if os.path.isdir(item_path):
                all_folders.append(item_path)
    else:
        print(f"Warning: Source directory '{source_fold_3}' does not exist.")

    if not all_folders:
        print("No folders found in the specified source directories. Exiting.")
        return

    print(f"Found {len(all_folders)} folders in total.")

    # Shuffle the list of all folders to randomize selection
    random.shuffle(all_folders)

    # Determine the split point
    if len(all_folders) < (train_count + validation_count):
        print(f"Warning: Not enough folders ({len(all_folders)}) to satisfy the requested counts ({train_count} train, {validation_count} validation). Adjusting counts.")
        train_count = min(train_count, len(all_folders))
        validation_count = min(validation_count, len(all_folders) - train_count)

    train_folders = all_folders[:train_count]
    validation_folders = all_folders[train_count : train_count + validation_count]

    print(f"Selected {len(train_folders)} folders for training and {len(validation_folders)} for validation.")

    # Copy folders to the training directory
    print("\nCopying folders to training directory...")
    for i, folder_path in enumerate(train_folders):
        folder_name = os.path.basename(folder_path)
        dest_path = os.path.join(train_dest, folder_name)
        try:
            shutil.copytree(folder_path, dest_path)
            print(f"  [{i+1}/{len(train_folders)}] Copied '{folder_name}' to '{train_dest}'")
        except shutil.Error as e:
            print(f"  Error copying '{folder_name}' to '{train_dest}': {e}")
        except OSError as e:
            print(f"  OS Error copying '{folder_name}' to '{train_dest}': {e}")

    # Copy folders to the validation directory
    print("\nCopying folders to validation directory...")
    for i, folder_path in enumerate(validation_folders):
        folder_name = os.path.basename(folder_path)
        dest_path = os.path.join(validation_dest, folder_name)
        try:
            shutil.copytree(folder_path, dest_path)
            print(f"  [{i+1}/{len(validation_folders)}] Copied '{folder_name}' to '{validation_dest}'")
        except shutil.Error as e:
            print(f"  Error copying '{folder_name}' to '{validation_dest}': {e}")
        except OSError as e:
            print(f"  OS Error copying '{folder_name}' to '{validation_dest}': {e}")

    print("\nFolder splitting and copying complete!")

# --- Configuration ---
SOURCE_BASE_PATH = "/home/kaim/scratch/MUSDB18_ALL/"
TRAIN_DESTINATION = os.path.join(SOURCE_BASE_PATH, "MUSDB18_HQ_1/train")
VALIDATION_DESTINATION = os.path.join(SOURCE_BASE_PATH, "MUSDB18_HQ_1/validation")
TRAIN_COUNT = 86
VALIDATION_COUNT = 14 # The remaining 14 folders

# Run the script
if __name__ == "__main__":
    split_and_copy_folders(
        SOURCE_BASE_PATH,
        TRAIN_DESTINATION,
        VALIDATION_DESTINATION,
        TRAIN_COUNT,
        VALIDATION_COUNT
    )
