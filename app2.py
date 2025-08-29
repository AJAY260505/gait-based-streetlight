import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Setup MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Parameters for gait classification
ankle_history = deque(maxlen=10)
SPEED_THRESHOLDS = {'slow': 0.004, 'medium': 0.013}

# For unstable gait (stride variability)
VARIABILITY_THRESHOLD = 0.005

# Crowd size - placeholder (MediaPipe Pose supports only 1 person in python)
crowd_count = 1  # In real deployment: would use a multi-person detection system

# Timing for FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    frame_resized = cv2.resize(frame, (640, 480))

    # Process image for pose estimation
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    gait_status = "No Person"
    light_mode = "OFF"
    box_color = (0, 0, 255)

    # Default for no detection
    walking_speed = 0
    crowd_count = 0
    unstable = False

    # Only proceed if pose detected
    if results.pose_landmarks:
        # Draw pose
        mp_draw.draw_landmarks(
            frame_resized, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get ankle keypoint (y normalized value)
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        ankle_avg_y = (left_ankle.y + right_ankle.y) / 2

        # Append to history for stride/speed estimation
        ankle_history.append(ankle_avg_y)

        # Estimate walking speed (simple: distance/frame)
        if len(ankle_history) == ankle_history.maxlen:
            diffs = [abs(a - b) for a, b in zip(list(ankle_history)[1:], list(ankle_history)[:-1])]
            walking_speed = np.mean(diffs)
            stride_variability = np.std(diffs)

            # Gait classification
            if stride_variability > VARIABILITY_THRESHOLD:
                gait_status = "UNSTABLE/ASSIST MODE"
                light_mode = "Assist: Wide, Projected Path"
                box_color = (0, 255, 255)
            elif walking_speed < SPEED_THRESHOLDS['slow']:
                gait_status = "SLOW"
                light_mode = "Warm, Wide Light"
                box_color = (0, 128, 255)
            elif walking_speed < SPEED_THRESHOLDS['medium']:
                gait_status = "NORMAL"
                light_mode = "Standard Adaptive"
                box_color = (0, 255, 0)
            else:
                gait_status = "FAST/RUNNING"
                light_mode = "Bright, Extended"
                box_color = (255, 255, 0)
        else:
            gait_status = "Analysing"
            light_mode = "ON"
            box_color = (200, 200, 200)

        # Since we cannot do crowd count with single-person pose, placeholder:
        crowd_count = 1  # In real system, use YOLO/Pose with people counting
    else:
        ankle_history.clear()

    # Simulate group/crowd logic (show as text only)
    group_mode = "Normal"
    if crowd_count >= 3:
        group_mode = "Crowd!"
        light_mode = "Crowd: Wide, Anticipatory"
        box_color = (255, 0, 0)

    # Show all info on the frame
    cv2.rectangle(frame_resized, (10, 10), (420, 120), box_color, -1)
    cv2.putText(frame_resized, f'Light Mode: {light_mode}', (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(frame_resized, f'Gait: {gait_status}', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
    cv2.putText(frame_resized, f'Crowd Mode: {group_mode}', (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Show FPS (frame rate)
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(frame_resized, f'FPS: {fps}', (550, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 255), 2)

    cv2.imshow('Revolutionary Gait-Based Smart Streetlight', frame_resized)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
