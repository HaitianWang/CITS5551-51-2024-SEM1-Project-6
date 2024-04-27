from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Input
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np
from sklearn.metrics import classification_report
import time 

import os

directory = r'F:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\Dataset\archive\WeedCrop.v1i.yolov5pytorch\trainv2'
contents = os.listdir(directory)
num_of_dirs = len([name for name in contents if os.path.isdir(os.path.join(directory, name))])

print("Contents of the directory:")
for item in contents:
    print(item)

print(f"\nNumber of directories: {num_of_dirs}")


from PIL import Image
import os

# Define the directory path
directory_path = r'F:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\Dataset\archive\WeedCrop.v1i.yolov5pytorch\trainv2'

# List all files in the directory
file_names = os.listdir(directory_path)

# Load images from the directory
images = []
for file_name in file_names:
    if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
        image_path = os.path.join(directory_path, file_name)
        image = Image.open(image_path)
        images.append(image)

# Process the images as required
# ...

# Example: Showing the first image
if images:
    images[0].show()
else:
    print("No images found in the directory.")

batch_size = 128
num_epochs = 3
image_size = (139, 139)
num_classes = 2

# Load the InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(*image_size, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
class_outputs = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=class_outputs)

# Compile the model
from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Load the training data with aggressive data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_dataset = train_datagen.flow_from_directory(
    r'F:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\Dataset\archive\WeedCrop.v1i.yolov5pytorch\trainv2',
    target_size=(139, 139),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the validation data with moderate data augmentation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_dataset = val_datagen.flow_from_directory(
    r'F:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\Dataset\archive\WeedCrop.v1i.yolov5pytorch\valid',
    target_size=(139, 139),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define learning rate scheduler
def lr_scheduler(epoch):
    if epoch < 10:
        return 0.001
    elif 10 <= epoch < 20:
        return 0.0001
    else:
        return 0.00001

lr_schedule = LearningRateScheduler(lr_scheduler)

# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define model checkpoint to save the best model
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
#checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
# Define ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=[lr_schedule, early_stop, checkpoint, reduce_lr]
)

# Save the model in native Keras format
model.save(r'C:\A\plant_disease_model_inceptionv2.keras')

import joblib

# Save the model using joblib
joblib.dump(model, r'C:\A\plant_disease_model_inceptionv2.pkl')


# Evaluate the model on the test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_dataset = test_datagen.flow_from_directory(
    r'F:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\Dataset\archive\WeedCrop.v1i.yolov5pytorch\testv2',
    target_size=(139, 139),
    batch_size=batch_size,
    class_mode='categorical'
)

test_labels = test_dataset.classes
test_labels = to_categorical(test_labels, num_classes=num_classes)

start_time = time.time()
y_pred = model.predict(test_dataset)
y_pred_bool = np.argmax(y_pred, axis=1)
rounded_labels = np.argmax(test_labels, axis=1)

print(classification_report(y_pred_bool, rounded_labels, digits=4))
print("Time taken to predict the model: " + str(time.time() - start_time))

# Save the model
model.save(r'C:\A\plant_disease_model_inceptionv2.h5')