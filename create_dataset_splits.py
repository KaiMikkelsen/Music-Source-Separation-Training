import os
import shutil
import numpy as np

# Assuming your dataset is organized as a directory of files
data_dir = "/home/kaim/projects/def-ichiro/kaim/data/MUSDB18HQ_ALL_SONGS"  # Original dataset path

# List all files in your dataset directory (ensure there are enough files)
all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

print(f"Total files in dataset: {len(all_files)}")
# Ensure the dataset has at least 150 files
assert len(all_files) >= 150, "Dataset should have at least 150 files."

# Shuffle the dataset to ensure randomness
np.random.seed(42)
shuffled_files = np.random.permutation(all_files)

# Split into test, train, validation sets
test_files = shuffled_files[:50]  # First 50 for testing
train_files = shuffled_files[50:136]  # Next 86 for training
valid_files = shuffled_files[136:150]  # Last 14 for validation

# Second split: Different combination for Dataset 2
shuffled_files_2 = np.random.permutation(all_files)
test_files_2 = shuffled_files_2[:50]
train_files_2 = shuffled_files_2[50:136]
valid_files_2 = shuffled_files_2[136:150]

# Third split: Different combination for Dataset 3
shuffled_files_3 = np.random.permutation(all_files)
test_files_3 = shuffled_files_3[:50]
train_files_3 = shuffled_files_3[50:136]
valid_files_3 = shuffled_files_3[136:150]

# Helper function to create directories and move files
def create_and_move_files(dataset_name, train_files, valid_files, test_files, source_dir):
    # Create a new directory for the dataset
    dataset_dir = os.path.join("/home/kaim/projects/def-ichiro/kaim/data/MUSDB18HQ_SPLITS", dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Create train, valid, and test subdirectories
    for subset in ['train', 'valid', 'test']:
        subset_dir = os.path.join(dataset_dir, subset)
        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)

    # Move files into the corresponding subdirectories
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(dataset_dir, 'train', file))
    for file in valid_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(dataset_dir, 'valid', file))
    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(dataset_dir, 'test', file))

# Create 3 different datasets
create_and_move_files('dataset_1', train_files, valid_files, test_files, data_dir)
create_and_move_files('dataset_2', train_files_2, valid_files_2, test_files_2, data_dir)
create_and_move_files('dataset_3', train_files_3, valid_files_3, test_files_3, data_dir)