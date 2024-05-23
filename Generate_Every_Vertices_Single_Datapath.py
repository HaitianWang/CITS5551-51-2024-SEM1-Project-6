import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rgb_path = r'E:\UWA\GENG 5551\2021 09 06 Test Split\smalldata_4_15\RGB_4_15.tif'
red_path = r'E:\UWA\GENG 5551\2021 09 06 Test Split\smalldata_4_15\Red_4_15.tif' # Input your own Red channel file path
green_path = r'E:\UWA\GENG 5551\2021 09 06 Test Split\smalldata_4_15\Green_4_15.tif'
blue_path = r'E:\UWA\GENG 5551\2021 09 06 Test Split\smalldata_4_15\Blue_4_15.tif'
red_edge_path = r'E:\UWA\GENG 5551\2021 09 06 Test Split\smalldata_4_15\RedEdge_4_15.tif'
nir_path = r'E:\UWA\GENG 5551\2021 09 06 Test Split\smalldata_4_15\NIR_4_15.tif'  # Input your own NIR channel file path

output_folder = r'E:\UWA\GENG 5551\Temp'
os.makedirs(output_folder, exist_ok=True)

with rasterio.open(red_path) as red_ds:
    red_array = red_ds.read(1).astype(np.float32)
    profile = red_ds.profile

with rasterio.open(green_path) as green_ds:
    green_array = green_ds.read(1).astype(np.float32)

with rasterio.open(blue_path) as blue_ds:
    blue_array = blue_ds.read(1).astype(np.float32)

with rasterio.open(red_edge_path) as red_edge_ds:
    red_edge_array = red_edge_ds.read(1).astype(np.float32)

with rasterio.open(nir_path) as nir_ds:
    nir_array = nir_ds.read(1).astype(np.float32)



def calculate_indices(R, G, B, RE, NIR):
    ExG = 2 * G - R - B
    ExR = 1.4 * R - G
    PRI = (R - G) / (R + G + np.finfo(np.float32).eps)
    MGRVI = (G**2 - R**2) / (G**2 + R**2 + np.finfo(np.float32).eps)
    GNDVI = (NIR - G) / (NIR + G + np.finfo(np.float32).eps)
    
    L = 0.5  # Soil Adjusted Vegetation Index correction factor
    SAVI = (1 + L) * (NIR - R) / (NIR + R + L + np.finfo(np.float32).eps)
    MSAVI = (2 * NIR + 1 - np.sqrt((2 * NIR + 1)**2 - 8 * (NIR - R))) / 2
    NDVI = (NIR - R) / (NIR + R + np.finfo(np.float32).eps)
    
    return ExG, ExR, PRI, MGRVI, GNDVI, SAVI, MSAVI, NDVI


ExG, ExR, PRI, MGRVI, GNDVI, SAVI, MSAVI, NDVI = calculate_indices(red_array, green_array, blue_array, red_edge_array, nir_array)

index_names = ['ExG', 'ExR', 'PRI', 'MGRVI', 'GNDVI', 'SAVI', 'MSAVI', 'NDVI']
index_arrays = [ExG, ExR, PRI, MGRVI, GNDVI, SAVI, MSAVI, NDVI]

for index_name, index_array in zip(index_names, index_arrays):

    df = pd.DataFrame(index_array)

    output_path = os.path.join(output_folder, f'{index_name}.csv')
    df.to_csv(output_path, index=False, header=False)

print("All indices have been calculated and saved as CSV files successfully.")
