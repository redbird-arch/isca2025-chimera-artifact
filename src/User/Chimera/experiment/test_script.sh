#!/bin/bash

# Check if path argument was provided
if [ -z "$1" ]; then
    echo "Error: Please provide a directory path as argument."
    exit 1
fi

# Get directory path argument
directory="$1"

# Verify directory exists
if [ ! -d "$directory" ]; then
    echo "Error: Invalid directory path - directory does not exist: $directory"
    exit 1
fi

# Activate conda environment
source activate Chimera

# Iterate through all .py files in directory
for file in "$directory"/*.py
do
    # Verify file exists and is Python file
    if [ -f "$file" ]; then
        echo "Executing: $file"
        
        # Execute Python file and record process ID
        python "$file" &
        pid=$!
        
        # Wait for 2 seconds of execution
        sleep 2
        
        # Check if process is still running
        if kill -0 $pid 2>/dev/null; then
            # If process is still running, kill it and continue to next file
            kill $pid
            echo "Process $file was terminated - proceeding to next file."
        else
            # If process completed before timeout, error and stop
            echo "Error: Process $file completed before timeout - stopping execution."
            break
        fi
        
    fi
done

echo "Test execution completed."