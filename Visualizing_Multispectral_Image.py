import rasterio
import numpy as np
import matplotlib.pyplot as plt

def read_band(image_path, band_index):
    """
    Reads a specific band from a multispectral image.

    Parameters:
    - image_path: Path to the multispectral .tif file.
    - band_index: The index of the band to read (1-indexed).

    Returns:
    - A NumPy array containing the data of the specified band.
    """
    with rasterio.open(image_path) as src:
        return src.read(band_index)

def normalize(array):
    """
    Normalizes a NumPy array to 0-1 scale.
    """
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def create_false_color_composite(image_path, bands, output_path=None):
    """
    Creates a false-color composite using specified bands.

    Parameters:
    - image_path: Path to the multispectral .tif file.
    - bands: A tuple of band indices to map to RGB channels.
    - output_path: Optional path to save the composite image.

    This function visualizes a false-color composite by mapping the specified bands to the RGB channels,
    normalizing them for visualization, and optionally saving the composite image.
    """
    # Read the specified bands
    red = read_band(image_path, bands[0])
    green = read_band(image_path, bands[1])
    blue = read_band(image_path, bands[2])
    
    # Normalize the bands
    red_normalized = normalize(red)
    green_normalized = normalize(green)
    blue_normalized = normalize(blue)
    
    # Stack bands into an RGB image
    rgb = np.dstack((red_normalized, green_normalized, blue_normalized))
    
    # Display the image
    plt.imshow(rgb)
    plt.axis('off')  # No axes for a cleaner look
    plt.show()
    
    # Save the composite image if an output path is provided
    if output_path:
        plt.imsave(output_path, rgb)

# Corrected example usage with a specified output file name
image_path = r'C:\Simon\Master of Professional Engineering\Software Design Project\Image Dataset\2020 07 30 Kondinin E2_Red\tile_256_3840.tif'
bands = (4, 3, 2)  # Make sure these band indices match your data
output_path = r'C:\Simon\Master of Professional Engineering\Software Design Project\Image Dataset\2020 07 30 Kondinin E2_Red\false_color_composite.png'
with rasterio.open(image_path) as src:
    print(f"Number of bands in image: {src.count}")
#create_false_color_composite(image_path, bands, output_path)
















