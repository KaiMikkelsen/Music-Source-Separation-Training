import os
import shutil
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Original dataset path (all songs)
data_dir = "/home/kaim/scratch/moisesdb23_labelnoise_v1.0"

# Output splits path (where the new datasets will be created)
output_dir = "/home/kaim/scratch/labelnoise_splits"

# List all folders in your dataset directory (ensure there are enough folders)
all_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

# Ensure the dataset has at least 150 folders
if len(all_folders) < 203:
    print(len(all_folders))
    logging.error("Dataset should have at least 240 folders.")
    exit(1)
else:
    logging.info(f"Found {len(all_folders)} folders in the dataset.")

# Shuffle the dataset to ensure randomness
np.random.seed(42)
shuffled_folders = np.random.permutation(all_folders)

# Split into test, train, validation sets
test_folders = shuffled_folders[:68]  # First 50 for testing
train_folders = shuffled_folders[68:184]  # Next 86 for training
valid_folders = shuffled_folders[184:203]  # Last 14 for validation

# Second split: Different combination for Dataset 2
shuffled_folders_2 = np.random.permutation(all_folders)
test_folders_2 = shuffled_folders_2[:68]
train_folders_2 = shuffled_folders_2[68:184]
valid_folders_2 = shuffled_folders_2[184:203]

# Third split: Different combination for Dataset 3
shuffled_folders_3 = np.random.permutation(all_folders)
test_folders_3 = shuffled_folders_3[:68]
train_folders_3 = shuffled_folders_3[68:184]
valid_folders_3 = shuffled_folders_3[184:203]

# Helper function to create directories and move folders
def create_and_move_folders(dataset_name, train_folders, valid_folders, test_folders, source_dir, output_dir):
    logging.info(f"Creating {dataset_name}...")
    
    # Create a new directory for the dataset
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        logging.info(f"Created directory: {dataset_dir}")
    
    # Create train, valid, and test subdirectories
    for subset in ['train', 'valid', 'test']:
        subset_dir = os.path.join(dataset_dir, subset)
        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)
            logging.info(f"Created subdirectory: {subset_dir}")

    # Move folders into the corresponding subdirectories
    for folder in train_folders:
        src = os.path.join(source_dir, folder)
        dst = os.path.join(dataset_dir, 'train', folder)
        shutil.copytree(src, dst)
        logging.info(f"Copied {folder} to {dst}")
    
    for folder in valid_folders:
        src = os.path.join(source_dir, folder)
        dst = os.path.join(dataset_dir, 'valid', folder)
        shutil.copytree(src, dst)
        logging.info(f"Copied {folder} to {dst}")

    for folder in test_folders:
        src = os.path.join(source_dir, folder)
        dst = os.path.join(dataset_dir, 'test', folder)
        shutil.copytree(src, dst)
        logging.info(f"Copied {folder} to {dst}")

# Create 3 different datasets
logging.info("Starting dataset creation process...")
create_and_move_folders('dataset_1', train_folders, valid_folders, test_folders, data_dir, output_dir)
create_and_move_folders('dataset_2', train_folders_2, valid_folders_2, test_folders_2, data_dir, output_dir)
create_and_move_folders('dataset_3', train_folders_3, valid_folders_3, test_folders_3, data_dir, output_dir)

logging.info("Dataset creation process completed successfully.")