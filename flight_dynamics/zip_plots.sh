#!/bin/bash

# Check if the user actually provided an experiment name
if [ -z "$1" ]; then
    echo "Error: Please provide an experiment name."
    echo "Usage: ./zip_plots.sh <experiment_name>"
    echo "Example: ./zip_plots.sh flexible_rat"
    exit 1
fi

EXPERIMENT_NAME=$1
TARGET_DIR="flight_dynamics/runs/${EXPERIMENT_NAME}"
TEMP_DIR="${TARGET_DIR}/flat_plots"
OUTPUT_FILE="${TARGET_DIR}/${EXPERIMENT_NAME}_all_plots.tar.gz"

# Check if the experiment directory actually exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '${TARGET_DIR}' does not exist!"
    exit 1
fi

echo "==> Gathering and renaming HTML plots from ${TARGET_DIR}..."

# Create a fresh temporary directory
mkdir -p "$TEMP_DIR"

# Find all html files and process them one by one
find "$TARGET_DIR" -name "*.html" | while read filepath; do
    # Extract the name of the specific model folder (e.g., flexible_rat1_2026...)
    parent_dir=$(basename $(dirname "$filepath"))
    
    # Extract the original filename (e.g., trajectory_plot_0.html)
    filename=$(basename "$filepath")
    
    # Combine them so the file is completely unique
    new_filename="${parent_dir}_${filename}"
    
    # Copy the file into the temporary directory with its new name
    cp "$filepath" "${TEMP_DIR}/${new_filename}"
done

echo "==> Zipping the flattened directory..."
# The -C flag tells tar to jump into the TARGET_DIR before zipping, 
# so your zip file just contains the "flat_plots" folder and not the whole /cs/labs/... path!
tar -czvf "$OUTPUT_FILE" -C "$TARGET_DIR" $(basename "$TEMP_DIR")

echo "==> Cleaning up..."
rm -rf "$TEMP_DIR"

echo "==> Success! Flattened zip saved to: ${OUTPUT_FILE}"