import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define parameters
IMG_SIZE = 128  # Resize all images to 128x128
GESTURES = ["rock", "paper", "scissors"]
DATASET_DIR = "dataset"

X, y = [], []

# Load and preprocess images
for label, gesture in enumerate(GESTURES):
    path = os.path.join(DATASET_DIR, gesture)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
        img = img / 255.0  # Normalize
        X.append(img)
        y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# One-hot encode labels (for classification)
y = to_categorical(y, num_classes=len(GESTURES))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset loaded: {X_train.shape[0]} training images, {X_test.shape[0]} test images.")
