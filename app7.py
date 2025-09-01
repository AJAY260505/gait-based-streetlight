import os
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import requests
from collections import deque
from scipy.signal import medfilt
from dotenv import load_dotenv

# For YOLO person detection (multi-gait, only if enabled)
from ultralytics import YOLO

# Enable multiple gait/person detection
ENABLE_YOLO = True

# Load YOLOv8 model for person detection if enabled
if ENABLE_YOLO:
    yolo_model = YOLO("yolov8n.pt")  # Use any YOLO model you prefer

load_dotenv()
API_KEY = os.getenv("OWM_API_KEY")

def get_my_location():
    try:
        r = requests.get("http://ip-api.com/json", timeout=3)
        result = r.json()
        lat = result['lat']
        lon = result['lon']
        city = result['city']
        return lat, lon, city
    except Exception as e:
        print(f"Location error: {e}")
        return 12.9716, 77.5946, 'Bangalore'

def get_weather(api_key, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        weather_main = data['weather'][0]['main'].lower()
        if 'rain' in weather_main:
            return 'rain'
        elif 'fog' in weather_main or 'mist' in weather_main or 'haze' in weather_main:
            return 'fog'
        elif 'clear' in weather_main:
            return 'clear'
        elif 'cloud' in weather_main:
            return 'cloudy'
        elif 'wind' in weather_main:
            return 'windy'
        else:
            return weather_main
    except Exception as e:
        print(f"Weather API error: {e}")
        return 'clear'

def get_ambient_light_level():
    hour = time.localtime().tm_hour
    if 6 <= hour < 18:
        return 'day'
    elif 18 <= hour < 20:
        return 'evening'
    else:
        return 'night'

def get_sound_level():
    return random.randint(40, 110)

def is_panic_run_detected(walking_speed):
    return walking_speed > 0.018 and get_sound_level() > 100

def user_assist_button():
    return random.random() < 0.01

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

WINDOW_SIZE = 21 # for median filtering (odd)
SPEED_THRESHOLDS = {'slow': 0.003, 'fast': 0.015}
VARIABILITY_THRESHOLD = 0.22
SOME_MIN_SPEED = 0.01

lat, lon, detected_city = get_my_location()
print(f"Detected location: {detected_city} ({lat}, {lon})")

weather = get_weather(API_KEY, lat, lon)
weather_last_update = time.time()

prev_time = 0
city_data_log = []

def remove_spikes(diff_list, k=5):
    arr = np.array(diff_list)
    if len(arr) < k:
        return arr  # Not enough data yet
    return medfilt(arr, kernel_size=k)

def analyze_gait(ankle_history):
    result = {
        "gait_status": "No Person",
        "light_mode": "OFF",
        "cue": "",
        "mode_color": (0,0,255),
        "assist_mode": False,
        "walking_speed": 0,
        "stride_var": 0
    }
    if len(ankle_history) >= 2:
        diffs = [abs(a - b) for a, b in zip(list(ankle_history)[1:], list(ankle_history)[:-1])]
        smoothed_diffs = remove_spikes(diffs, k=5)
        walking_speed = float(np.mean(smoothed_diffs))
        stride_var = float(np.std(smoothed_diffs))

        # Gait classification logic
        if stride_var > VARIABILITY_THRESHOLD and walking_speed > SOME_MIN_SPEED:
            result["gait_status"] = "UNSTABLE"
            result["light_mode"] = "Assist Mode: Wide + Guidance Cues"
            result["cue"] = "Project stripes/arrows"
            result["mode_color"] = (0, 255, 255)
            result["assist_mode"] = True
        elif walking_speed > SPEED_THRESHOLDS['fast']:
            result["gait_status"] = "FAST/RUN"
            result["light_mode"] = "Bright, Forward Extension"
            result["cue"] = "Illuminate several lights ahead"
            result["mode_color"] = (255, 255, 0)
        elif walking_speed < SPEED_THRESHOLDS['slow']:
            result["gait_status"] = "SLOW"
            result["light_mode"] = "Warm, Diffuse"
            result["cue"] = "Soft guidance"
            result["mode_color"] = (0, 128, 255)
        else:
            result["gait_status"] = "NORMAL"
            result["light_mode"] = "Adaptive"
            result["mode_color"] = (0,255,0)

        if is_panic_run_detected(walking_speed):
            result["light_mode"] = "Panic/Evacuation"
            result["cue"] = "Flash Red, Notify Control!"
            result["mode_color"] = (0,0,255)

        result["walking_speed"] = walking_speed
        result["stride_var"] = stride_var

    return result

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv2.resize(frame, (640, 480))
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # ---- Multi-person detection and gait analysis ----
    persons_results = []
    if ENABLE_YOLO:
        detections = yolo_model.predict(frame_resized, classes=[0], verbose=False)[0]  # class 0 is person in YOLO
        for det in detections.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
            person_crop = img_rgb[y1:y2, x1:x2] if y1<y2 and x1<x2 else img_rgb
            if person_crop.shape[0] < 40 or person_crop.shape[1] < 40:
                continue  # skip small detections (likely noise)
            single_pose = mp_pose.Pose()
            result = single_pose.process(person_crop)
            ankle_history = deque(maxlen=WINDOW_SIZE)
            # use the average ankle position in crop coordinates
            if result.pose_landmarks:
                mp_draw.draw_landmarks(person_crop, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                left_ankle = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                avg_ankle_y = (left_ankle.y + right_ankle.y) / 2
                ankle_history.append(avg_ankle_y)
                gait = analyze_gait(ankle_history)
                persons_results.append({
                    "box": (x1,y1,x2,y2),
                    **gait
                })
                # Draw box with color based on status
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), gait["mode_color"], 2)
                cv2.putText(frame_resized, f"{gait['gait_status']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gait["mode_color"], 2)
            else:
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame_resized, "NO GAIT", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # ----- Single-person logic (fallback for no YOLO) -----
    else:
        ankle_history = deque(maxlen=WINDOW_SIZE)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame_resized, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            ankle_avg_y = (left_ankle.y + right_ankle.y) / 2
            ankle_history.append(ankle_avg_y)
            gait = analyze_gait(ankle_history)
            persons_results.append({
                "box": None,
                **gait
            })
        else:
            persons_results.append({"gait_status":"No Person","light_mode":"Standby","cue":"","mode_color":(0,0,255)})

    # Main overlay - aggregate logic for display/log
    ambient = get_ambient_light_level()
    if time.time() - weather_last_update > 60:
        weather = get_weather(API_KEY, lat, lon)
        weather_last_update = time.time()
    sound = get_sound_level()
    accessibility_mode = user_assist_button()

    # Compose frame overlay, showing all people/gaits
    summary = "; ".join([
        f"[{i+1}] {p['gait_status']} (Speed: {p.get('walking_speed',0):.4f})"
        for i, p in enumerate(persons_results)
    ])
    cv2.rectangle(frame_resized, (10, 10), (630, 120), (100, 200, 200), -1)
    cv2.putText(frame_resized, f'People: {len(persons_results)}  {summary}',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame_resized, f'Env:{ambient}/{weather} City:{detected_city}',
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 70, 70), 2)
    cv2.putText(frame_resized, f'Edge AI Privacy', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10,10,10), 2)
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(frame_resized, f'FPS: {fps}', (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2)
    cv2.imshow('AI Smart Streetlight: Multi-person Gait', frame_resized)

    city_data_log.append({
        'timestamp': time.time(),
        'people':len(persons_results),
        'gaits':[p["gait_status"] for p in persons_results],
        'env':(ambient,weather),
        'sound':sound
    })
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\n--- Urban Analytics Log Sample (10 rows) ---")
for record in city_data_log[-10:]:
    print(record)
