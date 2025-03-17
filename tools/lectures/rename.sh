#!/bin/bash

# Define the target directory
TARGET_DIR="translated"

# Create the directory if it does not exist
mkdir -p "$TARGET_DIR"

# Move files with _cn.md to the target directory
mv *_cn.md "$TARGET_DIR/" 2>/dev/null

# Change to the target directory
cd "$TARGET_DIR" || exit 1

# Rename files only if they are inside the directory
for file in *_cn.md; do
    [ -e "$file" ] || continue  # Skip if no files match
    mv "$file" "${file/_cn.md/.md}"
done