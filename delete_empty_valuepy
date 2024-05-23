import os
import pandas as pd

def load_csv_data(root_folder):
    data_matrices = {}  # Dictionary to store data matrices, keyed by folder name

    # Walk through each directory in the root folder
    for dirpath, dirnames, filenames in os.walk(root_folder):
        has_csv = False
        for filename in filenames:
            if filename.endswith('.csv'):
                has_csv = True
                csv_path = os.path.join(dirpath, filename)
                df = pd.read_csv(csv_path)

                # Check if the CSV contains the value 3
                if 3 in df.values:
                    print(f"Skipping {csv_path} because it contains the value 3.")
                    break
        else:
            # If no CSV file contains the value 3 and at least one CSV exists, load the data
            if has_csv:
                data_matrices[os.path.basename(dirpath)] = df
                print(f"Loaded data from {csv_path}")

    return data_matrices

# Define the path to the root directory containing the folders
root_folder = r'/Users/lunaxu/Desktop/seg_pics'

# Load the data
data_matrices = load_csv_data(root_folder)

# Print the first 5 rows of each data matrix
for folder_name, matrix in data_matrices.items():
    print(f"First 5 rows from {folder_name}:")
    print(matrix.head())
