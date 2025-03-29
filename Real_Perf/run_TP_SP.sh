#!/bin/bash

# Define the output file path
OUTPUT_FILE="TP+SP.txt"

# Check if the output file already exists and remove it to start fresh
if [ -f "$OUTPUT_FILE" ]; then
    rm "$OUTPUT_FILE"
fi

# Loop to run the Python script 100 times
for i in {1..100}
do
    echo "Run $i:" >> "$OUTPUT_FILE"
    OMP_NUM_THREADS=16 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  TP_SP.py >> "$OUTPUT_FILE"
    echo "-----------------------------------" >> "$OUTPUT_FILE"
done