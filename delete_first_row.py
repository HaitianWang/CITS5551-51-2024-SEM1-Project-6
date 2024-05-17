import os
import pandas as pd

# Function to check if a string can be converted to an integer
def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# Function to process each label matrix file and conditionally remove the first row
def process_label_matrix(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)
    
    # Check if the first row contains any non-integer values
    first_row = df.iloc[0]
    if any(not is_integer(value) for value in first_row):
        # Remove the first row
        df = df.iloc[1:]
        print(f'Removed first row from {file_path}')
    else:
        print(f'First row is valid in {file_path}, no removal needed')

    # Save the modified data back to the same file
    df.to_csv(file_path, header=False, index=False)

# Base path to the dataset
base_path = r'E:\\UWA\\GENG 5551\\2021 09 06 Test Split'

# Traverse the base directory
for root, dirs, files in os.walk(base_path):
    for dir_name in dirs:
        if dir_name.startswith('smalldata_'):
            dir_path = os.path.join(root, dir_name)
            print(f'Processing folder: {dir_path}')
            for file_name in os.listdir(dir_path):
                if file_name.startswith('label_matrix_') and file_name.endswith('.csv'):
                    file_path = os.path.join(dir_path, file_name)
                    process_label_matrix(file_path)

print("Processing complete.")
