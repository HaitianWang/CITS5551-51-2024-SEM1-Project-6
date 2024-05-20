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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Available GPUs: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs available. Check your CUDA and cuDNN installation.")



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
base_path = '/home/haitian/CITIS5551/Dataset/20210906Select'
X, y = create_dataset(base_path)

# 打印数据形状以进行调试
print(f'Number of images: {X.shape[0]}, Number of masks: {y.shape[0]}')

# 确保数据集不为空
if X.size == 0 or y.size == 0:
    raise ValueError("The dataset is empty. Please check the data directory and file paths.")

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
    # 创建一个简单的图形并保存
    plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig("output_plot.png")  # 保存图形到文件
    print("Plot saved as output_plot.png")

visualize_samples(X_train, y_train)

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Conv2D, UpSampling2D, concatenate, Input, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 检查并设置GPU设备
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 创建模型的代码
def create_inceptionv3_fcn(input_shape, num_classes):
    print("Creating InceptionV3 model")

    # Custom input layer to match the input shape
    inputs = Input(shape=input_shape)

    # Adjusting inputs to match InceptionV3 expected input (3 channels)
    def slice_tensor(x, start, end):
        return x[:, :, :, start:end]

    channels_per_slice = 3
    num_slices = input_shape[-1] // channels_per_slice
    if input_shape[-1] % channels_per_slice != 0:
        raise ValueError("Number of channels in input must be divisible by 3.")

    slice_outputs = []
    base_models = []  # Store base models to access their layers later
    for i in range(num_slices):
        x_slice = Lambda(slice_tensor, arguments={'start': i*channels_per_slice, 'end': (i+1)*channels_per_slice})(inputs)
        base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=x_slice)
        for layer in base_model.layers:
            layer._name = f"{layer.name}_slice_{i}"
        slice_outputs.append(base_model.output)
        base_models.append(base_model)  # Store the base model

    # Concatenate slice outputs along the channel axis
    x = concatenate(slice_outputs, axis=-1)

    # Create the upsampling path
    layer_shapes = {
        'mixed10': (14, 14),
        'mixed5': (30, 30),
        'mixed2': (61, 61)
    }

    for i in range(num_slices):
        for layer_name in ['mixed10', 'mixed5', 'mixed2']:
            layer = base_models[i].get_layer(f"{layer_name}_slice_{i}").output
            target_shape = layer_shapes[layer_name]
            print(f"Upsampling from shape {x.shape} to match layer {layer_name} with shape {layer.shape}")

            # Calculate the number of times to upsample
            while (x.shape[1] * 2 <= target_shape[0]) and (x.shape[2] * 2 <= target_shape[1]):
                x = UpSampling2D(size=(2, 2))(x)
                print(f"Shape after upsampling: {x.shape}")

            # Ensure the shapes match before concatenation
            if x.shape[1] != target_shape[0] or x.shape[2] != target_shape[1]:
                x = Conv2D(layer.shape[-1], (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = tf.nn.relu(x)
                print(f"Shape after Conv2D adjustment: {x.shape}")

            # Final adjustment to ensure exact shape match
            x = tf.image.resize(x, (target_shape[0], target_shape[1]))
            print(f"Shape after final adjustment: {x.shape}")

            print(f"Layer {layer_name} shape: {layer.shape}")
            print(f"x shape before concatenation with {layer_name}: {x.shape}")
            x = concatenate([x, layer])
            print(f"x shape after concatenation with {layer_name}: {x.shape}")
            x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            print(f"Shape after concatenation and Conv2D: {x.shape}")

    # Final upsampling to match input size
    while x.shape[1] < 512:
        x = UpSampling2D(size=(2, 2))(x)
        if x.shape[1] > 512:
            x = tf.image.resize(x, (512, 512))
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        print(f"Shape after final upsampling and Conv2D: {x.shape}")

    # Ensure the final output shape matches input size
    x = tf.image.resize(x, (512, 512))

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(x)
    print(f"Shape after output layer: {outputs.shape}")

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# 定义模型
input_shape = (512, 512, 15)  # 根据输入形状调整（15 个指数）
num_classes = 4  # 0: 植被，1: 杂草，2: 空地，3: 无效
model = create_inceptionv3_fcn(input_shape, num_classes)

# 编译模型
print("Compiling the model")
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
print("Starting training")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=4)

# 保存模型
model.save('inceptionv3_fcn_model.h5')
print("Model saved as inceptionv3_fcn_model.h5")

# 函数：可视化预测结果
def visualize_predictions(X, y_true, y_pred, num_samples=3):
    plt.figure(figsize=(15, num_samples * 5))
    indices = np.random.choice(range(X.shape[0]), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.imshow(X[idx, :, :, 0], cmap='gray')  # 显示第一个通道
        plt.title(f'Input Image {idx}')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(np.argmax(y_true[idx], axis=-1), cmap='gray')  # 显示真实掩码
        plt.title(f'True Mask {idx}')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, 3 * i + 3)
        plt.imshow(np.argmax(y_pred[idx], axis=-1), cmap='gray')  # 显示预测掩码
        plt.title(f'Predicted Mask {idx}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 在验证集上进行预测
print("Predicting on validation set")
y_pred = model.predict(X_val)

# 可视化预测结果
visualize_predictions(X_val, y_val, y_pred)
