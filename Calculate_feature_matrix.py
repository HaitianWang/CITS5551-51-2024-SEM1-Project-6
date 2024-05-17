import os
import numpy as np
import rasterio
import re

def calculate_indices(red, green, blue, nir, red_edge):
    indices = {}
    # Adding small constant epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Safely compute indices with added epsilon
    indices['NDVI'] = (nir - red) / (nir + red + epsilon)
    indices['GNDVI'] = (nir - green) / (nir + green + epsilon)
    indices['EVI'] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + epsilon)
    indices['SAVI'] = ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5
    indices['MSAVI'] = 0.5 * (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red)))
    indices['ExG'] = 2 * green - red - blue
    indices['ExR'] = 1.3 * red - green
    indices['PRI'] = (green - blue) / (green + blue + epsilon)
    indices['MGRVI'] = (green ** 2 - red ** 2) / (green ** 2 + red ** 2 + epsilon)
    indices['REIP'] = (red_edge - red) / (red_edge + red + epsilon)
    indices['CI'] = (nir / red_edge) - 1
    indices['OSAVI'] = (1.16 * (nir - red)) / (nir + red + 0.16)
    indices['TVI'] = np.sqrt(indices['NDVI'] + 0.5)
    indices['MCARI'] = ((red_edge - red) - 0.2 * (red_edge - green) * (red_edge / red))
    indices['TCARI'] = 3 * ((red_edge - red) - 0.2 * (red_edge - green) * (red_edge / red))
    return indices

def process_folder(folder_path, folder_suffix):
    # Load files using dynamically constructed file names based on folder suffix
    red = rasterio.open(os.path.join(folder_path, f'Red_{folder_suffix}.tif')).read(1)
    green = rasterio.open(os.path.join(folder_path, f'Green_{folder_suffix}.tif')).read(1)
    blue = rasterio.open(os.path.join(folder_path, f'Blue_{folder_suffix}.tif')).read(1)
    nir = rasterio.open(os.path.join(folder_path, f'NIR_{folder_suffix}.tif')).read(1)
    red_edge = rasterio.open(os.path.join(folder_path, f'RedEdge_{folder_suffix}.tif')).read(1)

    indices = calculate_indices(red, green, blue, nir, red_edge)
    for index_name, index_array in indices.items():
        np.savetxt(os.path.join(folder_path, f"{index_name}_{folder_suffix}.csv"), index_array, delimiter=",")

def main(base_folder):
    for root, dirs, files in os.walk(base_folder):
        # Using regular expression to extract folder suffix from the directory name
        match = re.search(r"smalldata_(\d+_\d+)$", root)
        if match:
            folder_suffix = match.group(1)
            if all(f'{band}_{folder_suffix}.tif' in files for band in ['Red', 'Green', 'Blue', 'NIR', 'RedEdge']):
                print(f"Processing folder: {root}")
                process_folder(root, folder_suffix)

# Path to the directory containing all the dataset folders
base_folder = r'C:\Simon\Master of Professional Engineering\Software Design Project\Code Area\ImageData_Split'
main(base_folder)
