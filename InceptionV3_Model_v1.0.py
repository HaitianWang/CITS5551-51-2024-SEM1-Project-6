import os
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Conv2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to load a .tif file and return it as a numpy array
def load_tif(file_path):
    print(f"Loading TIF file from {file_path}")
    with rasterio.open(file_path) as src:
        image = src.read(1).astype(np.float32)
    return image

# Function to load a label matrix from a .csv file
def load_label_matrix(file_path):
    print(f"Loading label matrix from {file_path}")
    label_matrix = pd.read_csv(file_path, header=None).values
    return label_matrix

# Function to create a dataset from multiple indices and their masks
def create_dataset(base_path):
    indices = ['ExG', 'ExR', 'PRI', 'MGRVI', 'GNDVI', 'SAVI', 'MSAVI', 'NDVI', 'EVI', 'REIP', 'CI', 'OSAVI', 'TVI', 'MCARI', 'TCARI']
    images = []
    masks = []

    print(f"Traversing base directory: {base_path}")
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name.startswith('smalldata_'):
                dir_path = os.path.join(root, dir_name)
                print(f"Processing folder: {dir_path}")
                for file_name in os.listdir(dir_path):
                    if file_name.endswith('.tif'):
                        print(f"Processing file: {file_name}")
                        # Create a multi-channel image
                        channels = []
                        for index in indices:
                            index_path = os.path.join(dir_path, f"{index}_{file_name.split('_')[1]}_{file_name.split('_')[2].split('.')[0]}.tif")
                            if os.path.exists(index_path):
                                channels.append(load_tif(index_path))
                            else:
                                print(f"File {index_path} does not exist, adding zero array")
                                channels.append(np.zeros((512, 512)))  # Add a zero array if the file doesn't exist
                        
                        # Stack channels to create a multi-channel image
                        multi_channel_image = np.stack(channels, axis=-1)
                        images.append(multi_channel_image)
                        
                        # Load corresponding label matrix
                        label_matrix_file = f"label_matrix_{file_name.split('_')[1]}_{file_name.split('_')[2].split('.')[0]}.csv"
                        label_matrix_path = os.path.join(dir_path, label_matrix_file)
                        if os.path.exists(label_matrix_path):
                            label_matrix = load_label_matrix(label_matrix_path)
                            masks.append(label_matrix)
                        else:
                            print(f"Label matrix {label_matrix_path} does not exist")

    print(f"Finished creating dataset. Number of images: {len(images)}, Number of masks: {len(masks)}")
    return np.array(images), np.array(masks)

# Load the dataset
base_path = r'E:\UWA\GENG 5551\2021 09 06 Test Split Select'
X, y = create_dataset(base_path)

# Normalize the input images
print("Normalizing input images")
X = X / np.max(X)

# One-hot encode masks
num_classes = 4  # 0: vegetation, 1: weeds, 2: bare ground, 3: invalid
print("One-hot encoding masks")
y = to_categorical(y, num_classes=num_classes)

# Split the dataset into training and validation sets
print("Splitting dataset into training and validation sets")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')

def visualize_samples(X, y, num_samples=3):
    plt.figure(figsize=(15, num_samples * 5))
    indices = np.random.choice(range(X.shape[0]), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.imshow(X[idx, :, :, 0], cmap='gray')  # Show the first channel
        plt.title(f'Input Image {idx}')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(np.argmax(y[idx], axis=-1), cmap='gray')  # Show the mask
        plt.title(f'Mask {idx}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_samples(X_train, y_train)
