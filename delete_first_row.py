import os
import pandas as pd

# Function to process each label matrix file and remove the first row
def process_label_matrix(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)
    
    # Remove the first row
    df = df.iloc[1:]
    
    # Save the modified data back to the same file
    df.to_csv(file_path, header=False, index=False)

# Base path to the dataset
base_path = r'E:\UWA\GENG 5551\2021 09 06 Test Split'

# Traverse the base directory
for root, dirs, files in os.walk(base_path):
    for dir_name in dirs:
        if dir_name.startswith('smalldata_'):
            dir_path = os.path.join(root, dir_name)
            for file_name in os.listdir(dir_path):
                if file_name.startswith('label_matrix_') and file_name.endswith('.csv'):
                    file_path = os.path.join(dir_path, file_name)
                    process_label_matrix(file_path)
                    print(f'Processed {file_path}')

print("Processing complete.")
