import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import time

model = load_model(r'C:\A\plant_disease_model_inception.keras')

test_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 128
num_classes = 2

test_dataset = test_datagen.flow_from_directory(
    r'F:\DESKTOP-78O0QB5\Documents\UWA\GENG 5551\Dataset\archive\WeedCrop.v1i.yolov5pytorch\test',
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
