import rasterio
import numpy as np
import matplotlib.pyplot as plt

def read_band(image_path):
    """
    Reads the single band from a .tif file.

    Parameters:
    - image_path: Path to the .tif file.

    Returns:
    - A NumPy array containing the data of the band.
    """
    with rasterio.open(image_path) as src:
        return src.read(1)  # Assuming the image has only one band

def normalize(array):
    """
    Normalizes a NumPy array to 0-1 scale.
    """
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def visualize_and_save_band(image_path, output_path=None):
    """
    Visualizes the single band of an image and optionally saves the visualization.

    Parameters:
    - image_path: Path to the .tif file.
    - output_path: Optional path to save the visualization.
    """
    # Read and normalize the band
    band = read_band(image_path)
    band_normalized = normalize(band)
    
    # Display the image using a grayscale colormap
    plt.imshow(band_normalized, cmap='gray')
    plt.axis('off')  # No axes for a cleaner look
    plt.colorbar()  # Optional: to display the scale of pixel values
    plt.show()
    
    # Save the visualization if an output path is provided
    if output_path:
        plt.imsave(output_path, band_normalized, cmap='gray')

# Usage
image_path = r'C:\Simon\Master of Professional Engineering\Software Design Project\Image Dataset\2020 07 30 Kondinin E2_Red\tile_256_3840.tif'
output_path = r'C:\Simon\Master of Professional Engineering\Software Design Project\Image Dataset\2020 07 30 Kondinin E2_Red\false_color_composite\enhanced_visualization.png'
visualize_and_save_band(image_path, output_path)








