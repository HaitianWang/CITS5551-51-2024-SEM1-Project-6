import rasterio
from rasterio import windows
from itertools import product  # Make sure to import product

def get_tiles(ds, width, height):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

def save_tile(tile, transform, out_path, src):
    with rasterio.open(
        out_path,
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
        for window, transform in get_tiles(src, tile_size_x, tile_size_y):
            out_path = f"{output_folder}/tile_{window.col_off}_{window.row_off}.tif"
            save_tile(window, transform, out_path, src)

# Use raw string literals for Windows paths
image_path = r'C:\UWA\GENG 5551\RGB Original\2020 07 30 Kondinin E2 RGB.tif'
output_folder = r'C:\UWA\GENG 5551\2020 RGB Split'
tile_size_x = 256  # Tile width in pixels
tile_size_y = 256  # Tile height in pixels
split_image(image_path, output_folder, tile_size_x, tile_size_y)
