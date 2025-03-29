#!/bin/bash

# Set maximum number of Python files to run concurrently
CONCURRENT_LIMIT=$2

# Directory containing Python files to execute
DIRECTORY=$1

# Find all Python files recursively in the directory
find $DIRECTORY -name '*.py' | \
(
  # Initialize counter for currently running processes
  current_running=0

  # Process each Python file found
  while IFS= read -r file; do
    # Launch Python file in specified Conda environment (background process)
    conda run -n Chimera python "$file" &
    ((current_running++))

    # When reaching concurrency limit, wait for any process to complete
    if (( current_running >= CONCURRENT_LIMIT )); then
      # Wait for any single background process to finish
      wait -n
      ((current_running--))
    fi
  done

  # Wait for all remaining background processes to complete
  wait
)