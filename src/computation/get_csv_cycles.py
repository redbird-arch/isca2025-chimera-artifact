# File name  :    get_csv_cycles.py
# Author     :    xiaocuicui
# Time       :    2025/03/28 12:04:54
# Version    :    V1.0
# Abstract   :        

import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))


import pandas as pd


def sum_total_cycles(file_path: str) -> int:
    """
    Reads a CSV file and returns the sum of the "Total Cycles" column.
    
    Parameters:
        file_path (str): The path to the CSV file.
    
    Returns:
        int: The sum of the "Total Cycles" column.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Remove any extra spaces in column names
    df.columns = df.columns.str.strip()
    
    # Sum the "Total Cycles" column
    return df["Total Cycles"].sum()




# Example usage:
if __name__ == "__main__":
    file_path = './backend/runfile/moe_1point3_256_moe_weight_backward/GoogleTPU_v1_ws/COMPUTE_REPORT.csv'  # Replace with your actual file path
    total = sum_total_cycles(file_path)
    print("Sum of the Total Cycles column:", total)
