# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch # For custom legend
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# --- Function to visualize K-Fold splits ---
def plot_kfold_splits(cv, X, y, ax, n_splits, x_axis_limit=None):
    """
    Plots the indices for a K-Fold cross-validation object.

    Parameters:
    cv (KFold): The KFold cross-validation object.
    X (array-like): The feature set.
    y (array-like): The target variable (optional, but good practice for stratified folds).
    ax (matplotlib.axes.Axes): The matplotlib axes object to plot on.
    n_splits (int): The number of folds.
    x_axis_limit (int, optional): Maximum limit for the x-axis (number of samples).
                                 Defaults to the total number of samples in X.
    """
    cmap_data = plt.cm.Paired # Color map for data points if we were to plot them
    cmap_cv = plt.cm.coolwarm # Color map for train/test splits

    if x_axis_limit is None:
        x_axis_limit = len(X)

    # Iterate through the splits
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Create an array of NaNs and fill in training/testing indices
        # 0 for training, 1 for testing
        indices = np.full(len(X), np.nan)
        indices[train_idx] = 0
        indices[test_idx] = 1

        # Plot the training and testing indices for this fold
        # Each fold is a horizontal bar showing train/test parts
        ax.scatter(
            range(len(indices)),  # X-coordinates: sample index
            [i + 0.5] * len(indices),  # Y-coordinates: constant for this fold
            c=indices,          # Color based on train/test
            marker="_",         # Marker style
            lw=15,              # Marker line width (thickness of the bar)
            cmap=cmap_cv,       # Colormap
            vmin=-0.2,          # Ensure consistent coloring
            vmax=1.2,           # Ensure consistent coloring
        )

    # Formatting the plot
    ax.set_yticks(np.arange(n_splits) + 0.5)
    ax.set_yticklabels([f"Fold {i}" for i in range(n_splits)])
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Fold Number")
    ax.set_ylim(n_splits, -0.5) # Invert y-axis to have Fold 0 at the top
    ax.set_xlim(0, x_axis_limit)
    ax.set_title(f"{n_splits}-Fold Cross-Validation Splits", fontsize=14)

    # Add a legend
    # cmap_cv(0.0) should correspond to vmin (training set color)
    # cmap_cv(1.0) should correspond to vmax (testing set color)
    # Note: The exact color mapping might depend on the cmap_cv range and vmin/vmax.
    # For 'coolwarm', 0 is typically blue (train) and 1 is red (test).
    legend_patches = [
        Patch(color=cmap_cv(0.05), label="Training Set"), # Adjust fraction for desired color
        Patch(color=cmap_cv(0.95), label="Testing Set"),  # Adjust fraction for desired color
    ]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    return ax

# --- 1. Generate a synthetic dataset ---
X, y = make_classification(
    n_samples=100,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=42
)
print(f"Generated dataset shape: Features X = {X.shape}, Target y = {y.shape}\n")

# --- 2. Define K-Fold Cross-Validation parameters ---
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
print(f"Initialized KFold with {n_splits} splits.\n")

# --- Visualize the K-Fold splits ---
fig, ax = plt.subplots(figsize=(10, 4)) # Adjust figsize as needed
plot_kfold_splits(kf, X, y, ax, n_splits)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for legend if it's outside
plt.show() # Display the plot

# --- 3. Iterate through the K-Folds and demonstrate splits (optional: for detailed output) ---
print("\n--- Detailed Fold Information ---")
fold_counter = 0
accuracies_manual = [] # To store accuracies from manual loop
for train_index, test_index in kf.split(X):
    fold_counter += 1
    print(f"--- Fold {fold_counter}/{n_splits} ---")
    print(f"  Number of training samples: {len(train_index)}")
    print(f"  Number of testing samples:  {len(test_index)}")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_manual.append(accuracy)
    print(f"  Accuracy for Fold {fold_counter}: {accuracy:.4f}\n")

print(f"Manually calculated accuracies: {accuracies_manual}")
print(f"Mean accuracy (manual loop): {np.mean(accuracies_manual):.4f}")
print(f"Std deviation (manual loop): {np.std(accuracies_manual):.4f}\n")


# --- 4. Using cross_val_score for a more concise evaluation (Alternative) ---
model_for_cv = LogisticRegression(solver='liblinear', random_state=42)
cv_procedure = KFold(n_splits=n_splits, shuffle=True, random_state=42) # Same as kf
scores = cross_val_score(model_for_cv, X, y, cv=cv_procedure, scoring='accuracy')

print(f"\n--- Using cross_val_score (more concise) ---")
print(f"Scores for each fold: {scores}")
print(f"Mean accuracy (cross_val_score): {np.mean(scores):.4f}")
print(f"Standard deviation of accuracy (cross_val_score): {np.std(scores):.4f}")

print("\nScript finished.")
