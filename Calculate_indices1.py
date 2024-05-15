import os
import numpy as np
import rasterio
import pandas as pd


# Define the function to calculate indices
def calculate_indices(blue, green, red, nir, re):
    indices = {}

    # Excess Green (ExG)
    ExG = 2 * green - red - blue
    indices['ExG'] = ExG

    # Excess Red (ExR)
    ExR = 1.4 * red - green
    indices['ExR'] = ExR

    # Photochemical Reflectance Index (PRI)
    PRI = (green - blue) / (green + blue)
    indices['PRI'] = PRI

    # Modified Green Red Vegetation Index (MGRVI)
    MGRVI = (green ** 2 - red ** 2) / (green ** 2 + red ** 2)
    indices['MGRVI'] = MGRVI

    # Normalized Difference Vegetation Index (NDVI)
    NDVI = (nir - red) / (nir + red)
    indices['NDVI'] = NDVI

    # Green Normalized Difference Vegetation Index (GNDVI)
    GNDVI = (nir - green) / (nir + green)
    indices['GNDVI'] = GNDVI

    # Soil Adjusted Vegetation Index (SAVI)
    L = 0.5  # The L value is a constant
    SAVI = (nir - red) * (1 + L) / (nir + red + L)
    indices['SAVI'] = SAVI

    # Modified Soil Adjusted Vegetation Index (MSAVI)
    MSAVI = 0.5 * (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red)))
    indices['MSAVI'] = MSAVI

    return indices


# Function to process each folder
def process_folder(folder_path, hor, cor):
    try:
        # Read the .tif files
        blue = rasterio.open(os.path.join(folder_path, f'Blue_{hor}_{cor}.tif')).read(1)
        green = rasterio.open(os.path.join(folder_path, f'Green_{hor}_{cor}.tif')).read(1)
        red = rasterio.open(os.path.join(folder_path, f'Red_{hor}_{cor}.tif')).read(1)
        nir = rasterio.open(os.path.join(folder_path, f'NIR_{hor}_{cor}.tif')).read(1)
        re = rasterio.open(os.path.join(folder_path, f'RedEdge_{hor}_{cor}.tif')).read(1)

        # Calculate indices
        indices = calculate_indices(blue, green, red, nir, re)

        # Save indices to .csv files
        for index_name, index_data in indices.items():
            df = pd.DataFrame(index_data)
            csv_path = os.path.join(folder_path, f'{index_name}_{hor}_{cor}.csv')
            df.to_csv(csv_path, index=False, header=False)
            print(f'Saved {csv_path}')

    except Exception as e:
        print(f'Error processing {folder_path}: {e}')


# Main function to traverse directories and process folders
def main(base_path):
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            if dir.startswith('smalldata_'):
                folder_path = os.path.join(root, dir)
                parts = dir.split('_')
                hor = parts[1]
                cor = parts[2]
                print(f'Processing folder: {folder_path}')
                process_folder(folder_path, hor, cor)


# Specify the base path where the folders are located 路径改这里
base_path = '/Users/lunaxu/Desktop/seg_pics_color_copy2'

# Run the main function
main(base_path)
