import os
import rasterio
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer

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
base_path = '/home/haitian/Github/CITS5551-51-2024-SEM1-Project-6/2021 09 06 Test Split'
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

# 定义 ResizeLayer
class ResizeLayer(Layer):
    def __init__(self, target_shape, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_shape)

    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({"target_shape": self.target_shape})
        return config

    @classmethod
    def from_config(cls, config):
        target_shape = config.pop("target_shape")
        return cls(target_shape=target_shape, **config)

# 定义计算 IoU 的函数
def calculate_iou(y_true, y_pred, num_classes):
    iou_list = []
    for cls in range(num_classes):
        true_class = y_true == cls
        pred_class = y_pred == cls
        intersection = np.sum(np.logical_and(true_class, pred_class))
        union = np.sum(np.logical_or(true_class, pred_class))
        if union == 0:
            iou = 0  # 如果没有类别在图像中，IOU 设置为 0
        else:
            iou = intersection / union
        iou_list.append(iou)
    return np.mean(iou_list)

# 定义计算 mIoU 的函数
def calculate_miou(y_true, y_pred, num_classes):
    miou_list = []
    for i in range(y_true.shape[0]):
        iou = calculate_iou(y_true[i], y_pred[i], num_classes)
        miou_list.append(iou)
    return np.mean(miou_list)

# 加载模型
with tf.keras.utils.custom_object_scope({'ResizeLayer': ResizeLayer}):
    model = load_model('inceptionv3_fcn_model.h5', compile=False)

# 拆分训练和验证数据集
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用模型预测
y_pred = model.predict(X_val)

# 将预测结果转换为类别标签
y_true_labels = np.argmax(y_val, axis=-1)
y_pred_labels = np.argmax(y_pred, axis=-1)

# 计算并输出mIoU
miou = calculate_miou(y_true_labels, y_pred_labels, num_classes)
print(f"Mean Intersection over Union (mIoU): {miou}")
