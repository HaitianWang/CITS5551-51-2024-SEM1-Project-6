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
pattern = re.compile(r'smalldata_(\d+)_(\d+)')  
indices = ['ExG', 'ExR', 'PRI', 'MGRVI', 'SAVI', 'MSAVI', 'EVI', 'REIP', 'NDVI', 'GNDVI', 'CI', 'OSAVI', 'TVI', 'MCARI', 'TCARI']

def read_X(dir=base_path,indices=indices):
    images=[]
    labels=[]
    for root, dirs, files in os.walk(dir):
        for dir_name in dirs:
            match = pattern.match(dir_name)
            if match:

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

                                


                                file_path = os.path.join(dir_path, file_name)
                                channels.append(load_tif(file_path))


                                # import pandas as pd
                                # df = pd.read_csv(file_path)
                                # channels.append(df)  
                         
                         
                    if file_name.startswith("label_matrix"):
                        file_path = os.path.join(dir_path, file_name)
                        label_matrix = pd.read_csv(file_path, header=None).values
                        

                                   
                images.append(channels)
                labels.append(label_matrix)
    return np.array(images), np.array(labels)                     
def convert_to_one_hot(y):

    n, h, w = y.shape
    

    y_one_hot = np.zeros((n, h, w, 4), dtype=int)
    

    for i in range(4):
        y_one_hot[..., i] = (y == i)
    
    return y_one_hot


def get_predicted_labels(predictions):

    predicted_labels = np.argmax(predictions, axis=-1)
    return predicted_labels


def one_hot_to_labels(y_one_hot):

    y_labels = np.argmax(y_one_hot, axis=-1)
    
    return y_labels

base_path = 'testset'
X,y=read_X(base_path)


X = X / np.max(X)
X = X.transpose((0, 2, 3, 1))
y_one_hot=convert_to_one_hot(y)

def preprocess_image(image, label):
    return image, label  

def load_dataset(images, labels, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = load_dataset(X, y_one_hot)

    


class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_height, target_width):
        super(ResizeLayer, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize(inputs, (self.target_height, self.target_width))

input_tensor = Input(shape=(512, 512, 15))


x = Conv2D(3, (1, 1), padding='same', activation='relu')(input_tensor)
print(x.shape)# 512


base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(512, 512, 3))


x = base_model(x)


x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
print(x.shape)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
print(x.shape)


x = Conv2D(4, (1, 1), padding='same')(x)
print(x.shape)
x = ResizeLayer(512, 512)(x)
print(x.shape)


predictions = tf.keras.layers.Softmax(axis=-1)(x)

model = Model(inputs=input_tensor, outputs=predictions)


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
