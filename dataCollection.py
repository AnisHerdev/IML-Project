import cv2
import os

# Define labels for the gestures
GESTURES = ["rock", "paper", "scissors"]
DATASET_DIR = "dataset"

# Create directories if they don’t exist
for gesture in GESTURES:
    os.makedirs(os.path.join(DATASET_DIR, gesture), exist_ok=True)

# Start video capture
cap = cv2.VideoCapture(0)

print("Press 'r' for Rock, 'p' for Paper, 's' for Scissors, 'q' to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for better alignment
    frame = cv2.flip(frame, 1)

    # Display the frame
    cv2.putText(frame, "Press 'r', 'p', 's' to capture images", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Capture Gestures", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in [ord('r'), ord('p'), ord('s')]:
        label = GESTURES[['r', 'p', 's'].index(chr(key))]
        save_path = os.path.join(DATASET_DIR, label)

        # Count existing images
        count = len(os.listdir(save_path))
        filename = f"{save_path}/{count+1}.jpg"

        # Save the image
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

cap.release()
cv2.destroyAllWindows()