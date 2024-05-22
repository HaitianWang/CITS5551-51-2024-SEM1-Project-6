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

import rasterio
def load_tif(file_path):
    print(f"Loading TIF file from {file_path}")
    with rasterio.open(file_path) as src:
        image = src.read(1).astype(np.float32)
    return image
import os
import re

base_path = 'testset'
pattern = re.compile(r'smalldata_(\d+)_(\d+)')  # 正则表达式匹配 smalldata_ 后面的两个数字
indices = ['ExG', 'ExR', 'PRI', 'MGRVI', 'SAVI', 'MSAVI', 'EVI', 'REIP', 'NDVI', 'GNDVI', 'CI', 'OSAVI', 'TVI', 'MCARI', 'TCARI']

def read_X(dir=base_path,indices=indices):
    images=[]
    labels=[]
    for root, dirs, files in os.walk(dir):
        for dir_name in dirs:
            match = pattern.match(dir_name)
            if match:
                # 提取 smalldata_ 后面的数字
                group_number = match.group(1)
                sub_group_number = match.group(2)
                dir_path = os.path.join(root, dir_name)
                print(f"Processing folder: {dir_path} (Group: {group_number}, Sub-group: {sub_group_number})")
                channels = []
                for file_name in os.listdir(dir_path):
                    if file_name.endswith('.tif'):
                        for index in indices:
                            if file_name.startswith(index):
                                print(f"Processing file: {file_name} with feature: {index}")
                                # 创建多通道图像
                                
                                # 在这里进行进一步的文件处理
                                # 例如，可以读取 CSV 文件并执行某些操作
                                file_path = os.path.join(dir_path, file_name)
                                channels.append(load_tif(file_path))

                                # 读取CSV文件的示例代码
                                # import pandas as pd
                                # df = pd.read_csv(file_path)
                                # channels.append(df)  # 假设你要将数据添加到 channels 列表中
                         
                         
                    if file_name.startswith("label_matrix"):
                        file_path = os.path.join(dir_path, file_name)
                        label_matrix = pd.read_csv(file_path, header=None).values
                        

                                   
                images.append(channels)
                labels.append(label_matrix)
    return np.array(images), np.array(labels)                     
def convert_to_one_hot(y):
    # 获取输入数组的形状
    n, h, w = y.shape
    
    # 初始化一个形状为 (n, h, w, 4) 的全零数组
    y_one_hot = np.zeros((n, h, w, 4), dtype=int)
    
    # 使用高级索引将原数组的值转化为one-hot编码
    for i in range(4):
        y_one_hot[..., i] = (y == i)
    
    return y_one_hot


def get_predicted_labels(predictions):
    """
    将预测概率数组转换为标签数组。
    
    参数:
    predictions: 形状为 (n, 512, 512, 4) 的预测概率数组
    
    返回:
    形状为 (n, 512, 512) 的标签数组，每个点表示其最有可能的类别
    """
    predicted_labels = np.argmax(predictions, axis=-1)
    return predicted_labels


def one_hot_to_labels(y_one_hot):
    """
    将 one-hot 编码的数组还原为标签数组。
    
    参数:
    y_one_hot: 形状为 (n, 512, 512, 4) 的 one-hot 编码数组
    
    返回:
    形状为 (n, 512, 512) 的标签数组
    """
    # 使用 np.argmax 找到第四个维度的最大值的索引
    y_labels = np.argmax(y_one_hot, axis=-1)
    
    return y_labels

base_path = 'testset'
X,y=read_X(base_path)

# 归一化输入数据到 [0, 1]
X = X / np.max(X)
X = X.transpose((0, 2, 3, 1))
y_one_hot=convert_to_one_hot(y)

def preprocess_image(image, label):
    return image, label  # 不进行resize，保持原始尺寸

def load_dataset(images, labels, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = load_dataset(X, y_one_hot)

    

# 定义 ResizeLayer 自定义层
class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_height, target_width):
        super(ResizeLayer, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize(inputs, (self.target_height, self.target_width))

input_tensor = Input(shape=(512, 512, 15))

# 增加一个卷积层将输入转换为InceptionV3可接受的3通道输入
x = Conv2D(3, (1, 1), padding='same', activation='relu')(input_tensor)
print(x.shape)# 512

# 使用预训练的InceptionV3模型，不包含顶部的分类层
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

# 连接自定义输入层到基础模型
x = base_model(x)

# 使用卷积层保持空间维度一致
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
print(x.shape)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
print(x.shape)

# 添加最终的卷积层
x = Conv2D(4, (1, 1), padding='same')(x)
print(x.shape)
x = ResizeLayer(512, 512)(x)
print(x.shape)

# 应用Softmax激活函数确保输出符合概率分布
predictions = tf.keras.layers.Softmax(axis=-1)(x)

model = Model(inputs=input_tensor, outputs=predictions)

# 冻结预训练模型的卷积层
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# print(predictions.shape)
# print(np.sum(predictions[0, :, :, :], axis=-1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss') <= 0.1099 and logs.get('loss') <= 0.1099:
            print('\n\n Reached The Destination!')
            self.model.stop_training = True

callbacks = myCallback()
history = model.fit(
    train_dataset,
    epochs=1,
    # callbacks=[callbacks]
)

model.save('./inceptionv3_fcn_model_23_5.h5')
print("Model saved as inceptionv3_fcn_model_23_5.h5")
