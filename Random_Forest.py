from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np

def load_data(annotation_path, images_folder):
    # Parse the XML annotations file
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    data = []
    labels = []

    for image in root.findall('.//image'):
        # Load the image using OpenCV
        image_name = image.get('name')
        image_path = os.path.join(images_folder, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Image not found: {image_path}")
            continue
        img = cv2.resize(img, (224, 224))  # Resize for uniformity
        img = img.astype('float32') / 255.0  # Normalize pixel values

        # Iterate over each 'mask' and 'polygon' tag and extract the 'label' attribute
        for obj in image.findall('.//mask') + image.findall('.//polygon'):
            label = obj.get('label')
            if label:
                data.append(img)
                labels.append(label)

    return data, labels


# Set the path to your annotations and images
annotations_xml = r'C:\Simon\Master of Professional Engineering\Software Design Project\Image Dataset\2020 07 30 Kondinin E2 RGB Split\Try_labelling\annotations.xml'
images_folder = r'C:\Simon\Master of Professional Engineering\Software Design Project\Image Dataset\2020 07 30 Kondinin E2 RGB Split\Try_labelling'

# Use the load_data function to load your annotated images
data, labels = load_data(annotations_xml, images_folder)

# Flatten the images if you are using a traditional machine learning model like RandomForest
data = np.array(data).reshape(len(data), -1)
labels = np.array(labels)

# Encode labels to integers
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Initialize the classifier model
model = RandomForestClassifier(n_estimators=100)

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Print a classification report
print(classification_report(y_test, predictions, target_names=le.classes_))
