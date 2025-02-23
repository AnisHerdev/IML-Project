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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 output classes (rock, paper, scissors)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=15, batch_size=32)

# Save the trained model
model.save("rock_paper_scissors_model.h5")

print("Model training complete and saved as 'rock_paper_scissors_model.h5'")
