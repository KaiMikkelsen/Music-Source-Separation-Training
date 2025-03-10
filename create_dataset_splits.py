import os
import shutil
import numpy as np

# Original dataset path (all songs)
data_dir = "/home/kaim/projects/def-ichiro/kaim/data/MUSDB18HQ_ALL_SONGS"

# Output splits path (where the new datasets will be created)
output_dir = "/home/kaim/projects/def-ichiro/kaim/data/MUSDB18HQ_SPLITS"

# List all folders in your dataset directory (ensure there are enough folders)
all_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

# Ensure the dataset has at least 150 folders
assert len(all_folders) >= 150, "Dataset should have at least 150 folders."

# Shuffle the dataset to ensure randomness
np.random.seed(42)
shuffled_folders = np.random.permutation(all_folders)

# Split into test, train, validation sets
test_folders = shuffled_folders[:50]  # First 50 for testing
train_folders = shuffled_folders[50:136]  # Next 86 for training
valid_folders = shuffled_folders[136:150]  # Last 14 for validation

# Second split: Different combination for Dataset 2
shuffled_folders_2 = np.random.permutation(all_folders)
test_folders_2 = shuffled_folders_2[:50]
train_folders_2 = shuffled_folders_2[50:136]
valid_folders_2 = shuffled_folders_2[136:150]

# Third split: Different combination for Dataset 3
shuffled_folders_3 = np.random.permutation(all_folders)
test_folders_3 = shuffled_folders_3[:50]
train_folders_3 = shuffled_folders_3[50:136]
valid_folders_3 = shuffled_folders_3[136:150]

# Helper function to create directories and move folders
def create_and_move_folders(dataset_name, train_folders, valid_folders, test_folders, source_dir, output_dir):
    # Create a new directory for the dataset
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Create train, valid, and test subdirectories
    for subset in ['train', 'valid', 'test']:
        subset_dir = os.path.join(dataset_dir, subset)
        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)

    # Move folders into the corresponding subdirectories
    for folder in train_folders:
        shutil.copytree(os.path.join(source_dir, folder), os.path.join(dataset_dir, 'train', folder))
    for folder in valid_folders:
        shutil.copytree(os.path.join(source_dir, folder), os.path.join(dataset_dir, 'valid', folder))
    for folder in test_folders:
        shutil.copytree(os.path.join(source_dir, folder), os.path.join(dataset_dir, 'test', folder))

# Create 3 different datasets
create_and_move_folders('dataset_1', train_folders, valid_folders, test_folders, data_dir, output_dir)
create_and_move_folders('dataset_2', train_folders_2, valid_folders_2, test_folders_2, data_dir, output_dir)
create_and_move_folders('dataset_3', train_folders_3, valid_folders_3, test_folders_3, data_dir, output_dir)