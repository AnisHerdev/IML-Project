import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("rock_paper_scissors_model.h5")

# Define gestures
GESTURES = ["Rock", "Paper", "Scissors"]

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to avoid mirror effect
    frame = cv2.flip(frame, 1)

    # Preprocess the frame
    img = cv2.resize(frame, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    predicted_gesture = GESTURES[class_index]

    # Display the prediction on the frame
    cv2.putText(frame, f"Gesture: {predicted_gesture}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Rock Paper Scissors - Real-Time Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
