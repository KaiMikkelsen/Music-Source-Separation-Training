#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6        # Adjust based on your cluster's CPU/GPU ratio
#SBATCH --mem=125G               # Adjust memory as needed
#SBATCH --time=2-00:00           # DD-HH:MM:SS
#SBATCH --account=def-ichiro
#SBATCH --output=slurm_logs/slurm-%j.out  # Use Job ID for unique output files

# Define source directories to be zipped
# Add more directories to this array if needed
SOURCE_DIRS=(
    "/home/kaim/scratch/moises_4_stem"
    "/home/kaim/scratch/MOISESDB+MUSDB18HQ"
)

# Define the destination directory for the zip files
DEST_DIR="/home/kaim/nearline/def-ichiro/kaim"

echo "Starting the zipping and moving process..."

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Could not create destination directory '$DEST_DIR'."
    exit 1
fi

# Loop through each source directory
for SOURCE_DIR in "${SOURCE_DIRS[@]}"; do
    # Extract just the directory name for the zip filename
    DIR_NAME=$(basename "$SOURCE_DIR")
    ZIP_FILENAME="${DIR_NAME}_backup_$(date +%Y%m%d_%H%M%S).zip"

    echo "--- Processing directory: $SOURCE_DIR ---"

    # Check if source directory exists
    if [ ! -d "$SOURCE_DIR" ]; then
        echo "Error: Source directory '$SOURCE_DIR' not found. Skipping."
        continue # Move to the next directory in the loop
    fi

    # Create the zip archive for the current directory
    # The -r option zips directories recursively
    # The -q option makes it quiet (no verbose output during zipping)
    # The -j option "junk paths" means only the contents of the folder are zipped directly into the archive root.
    # If you want the folder itself (e.g., moises_4_stem/) inside the zip, remove the -j option.
    echo "Creating zip archive: $ZIP_FILENAME from $SOURCE_DIR..."
    zip -r -q "$ZIP_FILENAME" "$SOURCE_DIR"

    if [ $? -eq 0 ]; then
        echo "Zip archive '$ZIP_FILENAME' created successfully."
    else
        echo "Error: Failed to create zip archive for '$SOURCE_DIR'. Skipping move."
        continue # Move to the next directory in the loop
    fi

    # Move the zip file to the destination directory
    echo "Moving '$ZIP_FILENAME' to '$DEST_DIR'..."
    mv -v "$ZIP_FILENAME" "$DEST_DIR/"

    if [ $? -eq 0 ]; then
        echo "Successfully moved '$ZIP_FILENAME' to '$DEST_DIR'."
    else
        echo "Error: Failed to move '$ZIP_FILENAME' to '$DEST_DIR'."
        # You might want to remove the zip file here if the move failed, or keep it for debugging
        # rm "$ZIP_FILENAME" # Uncomment this line to delete the zip if move fails
        exit 1 # Exit if moving fails for any zip
    fi

    echo "--- Finished processing: $SOURCE_DIR ---"
done

echo "All specified directories processed."
