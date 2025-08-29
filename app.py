import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

# Simulated streetlight status/brightness variable
light_status = 'OFF'
prev_state = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process image for pose estimation
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    # Draw pose landmarks on frame
    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # If pose is detected
    if results.pose_landmarks:
        light_status = 'ON'
        color = (0, 255, 0)
    else:
        light_status = 'OFF'
        color = (0, 0, 255)

    # Display the simulated light status
    cv2.rectangle(frame, (10, 10), (200, 60), color, -1)
    cv2.putText(frame, f'Light: {light_status}', (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Gait-based Smart Light (Laptop Demo)', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
