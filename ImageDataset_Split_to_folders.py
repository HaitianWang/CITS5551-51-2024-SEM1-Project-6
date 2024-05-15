import rasterio
from rasterio import windows
import pandas as pd
from itertools import product
import os

def get_tiles(ds, width, height):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in offsets:
        if col_off + width > nols or row_off + height > nrows:
            continue  # Skip partial tiles at the edge
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform, (col_off, row_off)

def save_tile(tile, transform, out_path, folder_name, src):
    file_path = os.path.join(folder_name, out_path)
    os.makedirs(folder_name, exist_ok=True)  # Ensure the folder exists
    with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=tile.height,
            width=tile.width,
            count=src.count,
            dtype=src.dtypes[0],
            crs=src.crs,
            transform=transform,
    ) as dst:
        dst.write(src.read(window=tile))

def split_image(image_path, output_folder, tile_size_x, tile_size_y):
    with rasterio.open(image_path) as src:
        for window, transform, (col_off, row_off) in get_tiles(src, tile_size_x, tile_size_y):
            hor_index = col_off // tile_size_x + 1
            ver_index = row_off // tile_size_y + 1
            folder_name = os.path.join(output_folder, f"smalldata_{hor_index}_{ver_index}")
            base_name = os.path.basename(image_path).replace('.tif', '')
            out_path = f"{base_name}_{hor_index}_{ver_index}.tif"
            save_tile(window, transform, out_path, folder_name, src)

def split_csv(csv_path, output_folder, tile_size_x, tile_size_y, width, height):
    df = pd.read_csv(csv_path)
    num_tiles_x = width // tile_size_x
    num_tiles_y = height // tile_size_y
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            folder_name = os.path.join(output_folder, f"smalldata_{i+1}_{j+1}")
            os.makedirs(folder_name, exist_ok=True)
            sub_df = df.iloc[j*tile_size_y:(j+1)*tile_size_y, i*tile_size_x:(i+1)*tile_size_x]
            sub_df.to_csv(os.path.join(folder_name, f"label_matrix_{i+1}_{j+1}.csv"), index=False)

# Main execution block
base_folder = r'E:\UWA\GENG 5551\2021 09 06 Test Images & Labels' #输入文件夹地址
output_folder = r'E:\UWA\GENG 5551\2021 09 06 Test Split' #输出文件夹地址
tile_size_x = 512
tile_size_y = 512

# Assuming all images have the same dimensions
example_image_path = os.path.join(base_folder, 'Red.tif')
with rasterio.open(example_image_path) as src:
    width, height = src.width, src.height

images = ['Blue.tif', 'Green.tif', 'NIR.tif', 'Red.tif', 'RedEdge.tif', 'RGB.tif']
for image in images:
    split_image(os.path.join(base_folder, image), output_folder, tile_size_x, tile_size_y)

csv_path = r'E:\UWA\GENG 5551\2021 09 06 Test Images & Labels\GENG 5551 - NDVI Label v3.csv' #csv地址
split_csv(csv_path, output_folder, tile_size_x, tile_size_y, width, height)
