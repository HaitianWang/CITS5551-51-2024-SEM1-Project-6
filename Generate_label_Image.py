import rasterio
import numpy as np
import matplotlib.pyplot as plt

red_path = "result_Red.tif"  # Red channel file path
nir_path = "result_NIR.tif"  # NIR channel file path

# Open red channel TIF file
with rasterio.open(red_path) as red_ds:
    red_array = red_ds.read(1).astype(np.float32)
    profile = red_ds.profile

# Open NIR TIF file
with rasterio.open(nir_path) as nir_ds:
    nir_array = nir_ds.read(1).astype(np.float32)

# Calculate NDVI
ndvi = (nir_array - red_array) / (nir_array + red_array + 1e-8)

# Map NDVI to predefined categories
ndvi_mapped = np.where(ndvi > 0.54, 0, np.where(ndvi >= 0.41, 1, 2))

# Define color mapping
colors = {0: 'green', 1: 'red', 2: 'yellow'}

# Create color-mapped image
ndvi_color_mapped = np.zeros((ndvi_mapped.shape[0], ndvi_mapped.shape[1], 3), dtype=np.uint8)

# Display information about mapped arrays
print("Shape of ndvi_mapped:", ndvi_mapped.shape)
print("Shape of ndvi_color_mapped:", ndvi_color_mapped.shape)

# Color the values based on the mapping
for i, color in colors.items():
    # Get the RGB values of the color
    color_rgb = np.array(plt.cm.colors.to_rgba(color)[:3]) * 255
    # Apply colors based on the index and values
    ndvi_color_mapped[ndvi_mapped == i] = color_rgb

# Display the color image
plt.imshow(ndvi_color_mapped)
plt.axis('off')
plt.show()
