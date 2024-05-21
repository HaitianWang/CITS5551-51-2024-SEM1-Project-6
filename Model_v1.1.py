import os
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Conv2D, UpSampling2D, concatenate, Input, BatchNormalization, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 检查 GPU
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
    # sys.exit(1)

# 加载 .tif 文件并返回 numpy 数组
def load_tif(file_path):
    print(f"Loading TIF file from {file_path}")
    with rasterio.open(file_path) as src:
        image = src.read(1).astype(np.float32)
    return image

# 加载标签矩阵
def load_label_matrix(file_path):
    print(f"Loading label matrix from {file_path}")
    label_matrix = pd.read_csv(file_path, header=None).values
    return label_matrix

# 创建数据集
def create_dataset(base_path):
    # indices = ['ExG', 'ExR', 'PRI', 'MGRVI', 'GNDVI', 'SAVI', 'MSAVI', 'NDVI', 'EVI', 'REIP', 'CI', 'OSAVI', 'TVI', 'MCARI', 'TCARI']
    indices = ['ExG', 'ExR', 'PRI', 'MGRVI', 'SAVI', 'MSAVI', 'EVI', 'REIP', 'CI', 'OSAVI', 'TVI', 'MCARI', 'TCARI']
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
                        # 创建多通道图像
                        channels = []
                        for index in indices:
                            index_path = os.path.join(dir_path, f"{index}_{file_name.split('_')[1]}_{file_name.split('_')[2].split('.')[0]}.tif")
                            if os.path.exists(index_path):
                                channels.append(load_tif(index_path))
                            else:
                                print(f"File {index_path} does not exist, adding zero array")
                                channels.append(np.zeros((512, 512)))  # 文件不存在时添加零数组
                        
                        # 堆叠通道以创建多通道图像
                        multi_channel_image = np.stack(channels, axis=-1)
                        images.append(multi_channel_image)
                        
                        # 加载对应的标签矩阵
                        label_matrix_file = f"label_matrix_{file_name.split('_')[1]}_{file_name.split('_')[2].split('.')[0]}.csv"
                        label_matrix_path = os.path.join(dir_path, label_matrix_file)
                        if os.path.exists(label_matrix_path):
                            label_matrix = load_label_matrix(label_matrix_path)
                            masks.append(label_matrix)
                        else:
                            print(f"Label matrix {label_matrix_path} does not exist")

    print(f"Finished creating dataset. Number of images: {len(images)}, Number of masks: {len(masks)}")
    return np.array(images), np.array(masks)

# 加载数据集
base_path = '/home/haitian/Github/CITS5551-51-2024-SEM1-Project-6/2021 Test'
X, y = create_dataset(base_path)

# 打印数据形状以进行调试
print(f'Number of images: {X.shape[0]}, Number of masks: {y.shape[0]}')

# 确保数据集不为空
if X.size == 0 or y.size == 0:
    raise ValueError("The dataset is empty. Please check the data directory and file paths.")

# 归一化输入图像
print("Normalizing input images")
X = X / np.max(X)

# One-hot 编码标签
num_classes = 4  # 0: vegetation, 1: weeds, 2: bare ground, 3: invalid
print("One-hot encoding masks")
y = to_categorical(y, num_classes=num_classes)

# 拆分数据集为训练集和验证集
print("Splitting dataset into training and validation sets")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')

# 可视化样本
def visualize_samples(X, y, num_samples=3):
    plt.figure(figsize=(15, num_samples * 5))
    indices = np.random.choice(range(X.shape[0]), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.imshow(X[idx, :, :, 0], cmap='gray')  # 显示第一个通道
        plt.title(f'Input Image {idx}')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(np.argmax(y[idx], axis=-1), cmap='gray')  # 显示掩码
        plt.title(f'Mask {idx}')
        plt.axis('off')
    
    plt.tight_layout()
    # 创建一个简单的图形并保存
    plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig("output_plot.png")  # 保存图形到文件
    print("Plot saved as output_plot.png")

visualize_samples(X_train, y_train)

# 自定义层用于调整张量大小
class ResizeLayer(Layer):
    def __init__(self, target_height, target_width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize(inputs, (self.target_height, self.target_width))

# 自定义层用于应用 ReLU 激活
class ReluLayer(Layer):
    def call(self, inputs):
        return tf.nn.relu(inputs)

# 创建基于 InceptionV3 的全卷积网络 (FCN)
def create_inceptionv3_fcn(input_shape, num_classes):
    print("Creating InceptionV3 model")

    # 自定义输入层以匹配输入形状
    inputs = Input(shape=input_shape)

    # 创建基础 InceptionV3 模型，不使用预训练权重
    base_model = InceptionV3(weights=None, include_top=False, input_tensor=inputs)
    
    # 提取感兴趣的层
    mixed2 = base_model.get_layer('mixed2').output
    mixed5 = base_model.get_layer('mixed5').output
    mixed10 = base_model.get_layer('mixed10').output

    # 函数：上采样并连接层
    def upsample_and_concatenate(x, target_layer):
        target_shape = target_layer.shape[1:3]
        print(f"Upsampling from shape {x.shape} to match layer with shape {target_shape}")

        # 计算上采样次数
        while (x.shape[1] * 2 <= target_shape[0]) and (x.shape[2] * 2 <= target_shape[1]):
            x = UpSampling2D(size=(2, 2))(x)
            print(f"Shape after upsampling: {x.shape}")

        # 确保连接前形状匹配
        if x.shape[1] != target_shape[0] or x.shape[2] != target_shape[1]:
            x = Conv2D(target_layer.shape[-1], (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = ReluLayer()(x)
            print(f"Shape after Conv2D adjustment: {x.shape}")

        # 最后调整以确保形状匹配
        x = ResizeLayer(target_shape[0], target_shape[1])(x)
        print(f"Shape after final adjustment: {x.shape}")

        print(f"Layer shape: {target_layer.shape}")
        print(f"x shape before concatenation: {x.shape}")
        x = concatenate([x, target_layer])
        print(f"x shape after concatenation: {x.shape}")
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        print(f"Shape after concatenation and Conv2D: {x.shape}")
        return x

    # 初始处理
    x = base_model.output

    # 与 mixed10 上采样并连接
    x = upsample_and_concatenate(x, mixed10)

    # 与 mixed5 上采样并连接
    x = upsample_and_concatenate(x, mixed5)

    # 与 mixed2 上采样并连接
    x = upsample_and_concatenate(x, mixed2)

    # 最后上采样以匹配输入大小
    while x.shape[1] < 512:
        x = UpSampling2D(size=(2, 2))(x)
        if x.shape[1] > 512:
            x = ResizeLayer(512, 512)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        print(f"Shape after final upsampling and Conv2D: {x.shape}")

    # 确保最终输出形状匹配输入大小
    x = ResizeLayer(512, 512)(x)

    # 输出层
    outputs = Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(x)

    print(f"Shape after output layer: {outputs.shape}")

    # 定义模型
    model = Model(inputs=inputs, outputs=outputs)

    return model

# 定义模型参数
input_shape = (512, 512, 13)  # 根据你的数据调整输入形状（15 通道）
num_classes = 4  # 0: Vegetation, 1: Weed, 2: Bare Soil, 3: Invalid
model = create_inceptionv3_fcn(input_shape, num_classes)

# 编译模型
print("Compiling the model")
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# # 训练数据（替换为实际数据）
# X_train = np.random.rand(160, 512, 512, 15)  # 示例训练数据
# y_train = np.random.randint(0, 4, (160, 512, 512, 4))  # 示例训练标签

# # 验证数据（替换为实际数据）
# X_val = np.random.rand(50, 512, 512, 15)  # 示例验证数据
# y_val = np.random.randint(0, 4, (50, 512, 512, 4))  # 示例验证标签

# 训练模型
print("Starting training")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=8)

# 保存模型
model.save('inceptionv3_fcn_model.h5')
print("Model saved as inceptionv3_fcn_model.h5")

# 可视化预测结果
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

# 预测验证集
print("Predicting on validation set")
y_pred = model.predict(X_val)

# 可视化预测结果
visualize_predictions(X_val, y_val, y_pred)
