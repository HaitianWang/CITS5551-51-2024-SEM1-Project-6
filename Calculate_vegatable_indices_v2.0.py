import os
import numpy as np
import rasterio
import pandas as pd

# Define the function to calculate indices
def calculate_indices(blue, green, red, nir, re):
    indices = {}

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

    # Excess Green (ExG)
    ExG = 2 * green - red - blue
    indices['ExG'] = ExG

    # Excess Red (ExR)
    ExR = 1.3 * red - green
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

    # Enhanced Vegetation Index (EVI)
    EVI = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    indices['EVI'] = EVI

    # Red Edge Inflection Point Index (REIP)
    REIP = 700 + 40 * (((red + re) / 2 - green) / (re - green))
    indices['REIP'] = REIP

    # Chlorophyll Index (CI)
    CI = (nir / red) - 1
    indices['CI'] = CI

    # Optimized Soil Adjusted Vegetation Index (OSAVI)
    OSAVI = (nir - red) / (nir + red + 0.16)
    indices['OSAVI'] = OSAVI

    # Transformed Vegetation Index (TVI)
    TVI = np.sqrt(NDVI + 0.5)
    indices['TVI'] = TVI

    # Modified Chlorophyll Absorption in Reflectance Index (MCARI)
    MCARI = (re - red) - 0.2 * (re - green) * (re / red)
    indices['MCARI'] = MCARI

    # Transformed Chlorophyll Absorption in Reflectance Index (TCARI)
    TCARI = 3 * ((re - red) - 0.2 * (re - green) * (re / red))
    indices['TCARI'] = TCARI

    return indices

# Function to save array as a .tif file
def save_as_tif(data, profile, path):
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)
    print(f'Saved {path}')

# Function to process each folder
def process_folder(folder_path, hor, cor):
    try:
        # Read the .tif files
        blue_src = rasterio.open(os.path.join(folder_path, f'Blue_{hor}_{cor}.tif'))
        green_src = rasterio.open(os.path.join(folder_path, f'Green_{hor}_{cor}.tif'))
        red_src = rasterio.open(os.path.join(folder_path, f'Red_{hor}_{cor}.tif'))
        nir_src = rasterio.open(os.path.join(folder_path, f'NIR_{hor}_{cor}.tif'))
        re_src = rasterio.open(os.path.join(folder_path, f'RedEdge_{hor}_{cor}.tif'))

        blue = blue_src.read(1)
        green = green_src.read(1)
        red = red_src.read(1)
        nir = nir_src.read(1)
        re = re_src.read(1)

        # Calculate indices
        indices = calculate_indices(blue, green, red, nir, re)

        # Save indices to .csv and .tif files
        for index_name, index_data in indices.items():
            csv_path = os.path.join(folder_path, f'{index_name}_{hor}_{cor}.csv')
            tif_path = os.path.join(folder_path, f'{index_name}_{hor}_{cor}.tif')

            # Check if files already exist
            if not os.path.exists(csv_path):
                df = pd.DataFrame(index_data)
                df.to_csv(csv_path, index=False, header=False)
                print(f'Saved {csv_path}')
            else:
                print(f'{csv_path} already exists, skipping.')

            if not os.path.exists(tif_path):
                profile = blue_src.profile
                profile.update(dtype=rasterio.float32, count=1)
                save_as_tif(index_data.astype(rasterio.float32), profile, tif_path)
            else:
                print(f'{tif_path} already exists, skipping.')

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
base_path = r'E:\UWA\GENG 5551\2021 09 06 Test Split'

# Run the main function
main(base_path)
