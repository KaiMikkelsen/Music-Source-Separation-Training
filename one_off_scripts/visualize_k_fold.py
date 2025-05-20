import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import KFold

# Define the dataset size and cross-validation parameters
n_samples = 390  # Total number of songs
n_splits = 3     # 3-fold cross-validation
fold_size = n_samples // n_splits  # Size of each fold (150 // 3 = 50)
validation_size = 36 # 14 songs for validation

# Create dummy data (indices) to represent the dataset
X = np.arange(n_samples)

# Create a KFold object for the main 3 splits
kf_test_split = KFold(n_splits=n_splits, shuffle=False) # Shuffle=False to keep folds contiguous for simpler visualization logic

# Function to plot the cross-validation splits including train, validation, and test sets
def plot_custom_kfold(kf, X, ax, n_splits, validation_size, title):
    """
    Plots the indices for a custom cross-validation scenario with train, validation, and test sets.

    Parameters:
    kf: KFold object for the main splits (test set)
    X: Feature set (or indices in this case)
    ax: Matplotlib axis object
    n_splits: Number of folds in the cross-validation
    validation_size: Number of samples to reserve for the validation set
    title: Title of the plot
    """

    # Using a colormap that clearly distinguishes three values: 0 (train), 1 (validation), 2 (test)
    # 'viridis', 'plasma', 'inferno', 'magma', 'cividis' are good options for sequential.
    # 'coolwarm' is diverging, which can also work. Let's use 'viridis' and map values explicitly.
    cmap = plt.get_cmap('viridis')
    colors = [cmap(0.1), cmap(0.5), cmap(0.9)] # Distinct colors for train, validation, test

    # Use the split method to get the indices for the main folds (which will serve as test sets)
    test_folds_indices = list(kf.split(X))

    for i_split in range(n_splits):
        # The current fold is the test set for this iteration
        test_index = test_folds_indices[i_split][1]
        # The remaining folds are the initial training/validation pool
        train_val_pool_indices = np.setdiff1d(np.arange(n_samples), test_index)

        # Randomly sample indices for the validation set from the training/validation pool
        # Ensure reproducibility if needed by setting a random state here or globally
        rng = np.random.RandomState(i_split) # Use a different random state for each fold for variation in validation split
        rng.shuffle(train_val_pool_indices) # Shuffle the pool before splitting
        validation_index = train_val_pool_indices[:validation_size]
        train_index = train_val_pool_indices[validation_size:]


        # Create an array of NaNs and fill in training, validation, and testing indices
        indices = np.full(len(X), np.nan)
        # Use different values to represent train, validation, and test
        indices[test_index] = 2  # Represent test set with value 2
        indices[validation_index] = 1 # Represent validation set with value 1
        indices[train_index] = 0 # Represent training set with value 0


        # Plot the training, validation, and testing indices
        ax_x = range(len(indices))
        ax_y = [i_split + 0.5] * len(indices)

        # Scatter plot with different colors based on the index value
        # We'll map the index values (0, 1, 2) to the defined colors
        scatter = ax.scatter(ax_x, ax_y, c=indices, marker="_",
                           lw=10, cmap='viridis', vmin=-0.5, vmax=2.5) # Use 'viridis' colormap

    # Set y-ticks and labels
    y_ticks = np.arange(n_splits) + 0.5
    ax.set(yticks=y_ticks, yticklabels=range(n_splits),
           xlabel="Song Index", ylabel="Fold",
           ylim=[n_splits, -0.2], xlim=[0, n_samples])

    # Set plot title
    ax.set_title(title, fontsize=14)

    # Create legend patches manually with explicit colors
    legend_patches = [
                      Patch(color=cmap(0.9), label="Testing set (130 songs)"), # Map color based on value 2
                      Patch(color=cmap(0.5), label="Validation set (36 songs)"), # Map color based on value 1
                      Patch(color=cmap(0.1), label="Training set (224 songs)") # Map color based on value 0
                      ]


    ax.legend(handles=legend_patches, loc=(1.03, 0.7))


# Create figure and axis for the plot
fig, ax = plt.subplots(figsize=(9, 4)) # Adjusted figure size for better visibility of legend

# Plot the custom K-Fold visualization
plot_custom_kfold(kf_test_split, X, ax, n_splits, validation_size, "3-Fold Cross-Validation on MUSBDB18-HQ + MoisesDB (390 songs)")

plt.tight_layout()
fig.subplots_adjust(right=0.7) # Adjust layout to make space for the legend
plt.show()