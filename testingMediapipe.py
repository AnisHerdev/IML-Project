import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

def classify_hand_shape(hand_landmarks):
    # Extract landmark coordinates
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Calculate distances between landmarks
    thumb_index_dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    index_middle_dist = ((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2) ** 0.5
    middle_ring_dist = ((middle_tip.x - ring_tip.x) ** 2 + (middle_tip.y - ring_tip.y) ** 2) ** 0.5
    ring_pinky_dist = ((ring_tip.x - pinky_tip.x) ** 2 + (ring_tip.y - pinky_tip.y) ** 2) ** 0.5

    # Determine hand shape based on distances
    if thumb_index_dist < 0.1 and index_middle_dist < 0.1 and middle_ring_dist < 0.1 and ring_pinky_dist < 0.1:
        return 'Stone'
    elif thumb_index_dist > 0.2 and index_middle_dist > 0.2 and middle_ring_dist > 0.2 and ring_pinky_dist > 0.2:
        return 'Paper'
    elif index_middle_dist > 0.2 and middle_ring_dist < 0.1 and ring_pinky_dist > 0.2:
        return 'Scissors'
    else:
        return 'Unknown'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    # Draw hand landmarks and classify hand shape
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_shape = classify_hand_shape(hand_landmarks)
            cv2.putText(frame, hand_shape, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
