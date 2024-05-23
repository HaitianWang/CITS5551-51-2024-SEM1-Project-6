import os
import rasterio
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer


def load_tif(file_path):
    print(f"Loading TIF file from {file_path}")
    with rasterio.open(file_path) as src:
        image = src.read(1).astype(np.float32)
    return image


def load_label_matrix(file_path):
    print(f"Loading label matrix from {file_path}")
    label_matrix = pd.read_csv(file_path, header=None).values
    return label_matrix


def create_dataset(base_path):
    indices = ['ExG', 'ExR', 'PRI', 'MGRVI',  'SAVI', 'MSAVI',  'EVI', 'REIP', 'CI', 'OSAVI', 'TVI', 'MCARI', 'TCARI']
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

                        channels = []
                        for index in indices:
                            index_path = os.path.join(dir_path, f"{index}_{file_name.split('_')[1]}_{file_name.split('_')[2].split('.')[0]}.tif")
                            if os.path.exists(index_path):
                                channels.append(load_tif(index_path))
                            else:
                                print(f"File {index_path} does not exist, adding zero array")
                                channels.append(np.zeros((512, 512)))  
                        

                        multi_channel_image = np.stack(channels, axis=-1)
                        images.append(multi_channel_image)
                        

                        label_matrix_file = f"label_matrix_{file_name.split('_')[1]}_{file_name.split('_')[2].split('.')[0]}.csv"
                        label_matrix_path = os.path.join(dir_path, label_matrix_file)
                        if os.path.exists(label_matrix_path):
                            label_matrix = load_label_matrix(label_matrix_path)
                            masks.append(label_matrix)
                        else:
                            print(f"Label matrix {label_matrix_path} does not exist")

    print(f"Finished creating dataset. Number of images: {len(images)}, Number of masks: {len(masks)}")
    return np.array(images), np.array(masks)


base_path = './testset'
X, y = create_dataset(base_path)


print(f'Number of images: {X.shape[0]}, Number of masks: {y.shape[0]}')


if X.size == 0 or y.size == 0:
    raise ValueError("The dataset is empty. Please check the data directory and file paths.")


print("Normalizing input images")
X = X / np.max(X)


num_classes = 4  # 0: vegetation, 1: weeds, 2: bare ground, 3: invalid
print("One-hot encoding masks")
y = to_categorical(y, num_classes=num_classes)



def calculate_accuracy(y_true, y_pred):
    accuracies = []
    for i in range(y_true.shape[0]):
        true_labels = y_true[i].flatten()
        pred_labels = y_pred[i].flatten()
        
        correct_predictions = np.sum(true_labels == pred_labels)
        total_pixels = true_labels.size
        
        accuracy = correct_predictions / total_pixels
        accuracies.append(accuracy)
    
    return np.mean(accuracies)


class ResizeLayer(Layer):
    def __init__(self, target_height, target_width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize(inputs, (self.target_height, self.target_width))
    

class ReluLayer(Layer):
    def call(self, inputs):
        return tf.nn.relu(inputs)
custom_objects = {
    'ResizeLayer': ResizeLayer(512,512),
    'ReluLayer': ReluLayer
}



with tf.keras.utils.custom_object_scope(custom_objects):
    model = load_model('inceptionv3_fcn_model.h5',compile=False)


from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, y_val = create_dataset(base_path)


y_pred = model.predict(X_val)

y_true_labels = np.argmax(y_val, axis=-1)
y_pred_labels = np.argmax(y_pred, axis=-1)
def calculate_miou(y_true, y_pred, num_classes):
    iou_list = []
    for c in range(num_classes):
        true_class = (y_true == c)
        pred_class = (y_pred == c)
        
        intersection = np.sum(true_class & pred_class)
        union = np.sum(true_class | pred_class)
        print(c)
        if union == 0:
            iou = 1.0  # If there is no ground truth or predicted instance in this class
        else:
            iou = intersection / union
            print(iou)
        
        iou_list.append(iou)
    
    miou = np.mean(iou_list)
    return miou


miou = calculate_miou(y_true_labels, y_pred_labels, num_classes)
print(f"Mean Intersection over Union (mIoU): {miou}")


miou = calculate_miou(y_true_labels, y_pred_labels, num_classes)
print(f"Mean Intersection over Union (mIoU): {miou}")



print(calculate_accuracy(y_true_labels, y_pred_labels))


