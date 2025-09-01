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

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# == SETTINGS ==
ENABLE_YOLO = YOLO_AVAILABLE  # Toggle YOLO multi-person detection if installed
POSE_VIS_THRESH = 0.5         # MediaPipe landmark minimum visibility

# == Initialize models and utilities (do ONCE) ==
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)

if ENABLE_YOLO:
    yolo_model = YOLO("yolov8n.pt")

# == ENVIRONMENT HELPERS ==
load_dotenv()
API_KEY = os.getenv("OWM_API_KEY")

def get_my_location():
    try:
        r = requests.get("http://ip-api.com/json", timeout=3)
        result = r.json()
        return result['lat'], result['lon'], result['city']
    except Exception:
        return 12.9716, 77.5946, "Bangalore"

def get_weather(api_key, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        return data['weather'][0]['main'].lower()
    except Exception:
        return 'clear'

def get_ambient_light_level():
    hour = time.localtime().tm_hour
    if 6 <= hour < 18: return 'day'
    elif 18 <= hour < 20: return 'evening'
    return 'night'

def get_sound_level():
    return random.randint(40, 110)

def is_panic_run_detected(walking_speed):
    return walking_speed > 0.018 and get_sound_level() > 100

def user_assist_button():
    return random.random() < 0.01

# == GAIT ANALYSIS CONFIG ==
WINDOW_SIZE = 21
SPEED_THRESHOLDS = {'slow': 0.003, 'fast': 0.015}
VARIABILITY_THRESHOLD = 0.22
SOME_MIN_SPEED = 0.01

# == FILTERING ==
def remove_spikes(diff_list, k=5):
    arr = np.array(diff_list)
    if len(arr) < k: return arr
    return medfilt(arr, kernel_size=k)

def analyze_gait(ankle_history):
    result = {"gait_status":"No Gait", "light_mode":"OFF", "cue":"", "mode_color":(0,0,255), "assist_mode":False, "walking_speed":0, "stride_var":0}
    if len(ankle_history) >= 2:
        diffs = [abs(a-b) for a,b in zip(list(ankle_history)[1:], list(ankle_history)[:-1])]
        smoothed_diffs = remove_spikes(diffs, k=5)
        walking_speed = float(np.mean(smoothed_diffs))
        stride_var = float(np.std(smoothed_diffs))
        # -- Classify --
        if stride_var > VARIABILITY_THRESHOLD and walking_speed > SOME_MIN_SPEED:
            result.update(gait_status="UNSTABLE", light_mode="Assist Mode: Wide+Guidance", cue="Project stripes", mode_color=(0,255,255), assist_mode=True)
        elif walking_speed > SPEED_THRESHOLDS['fast']:
            result.update(gait_status="FAST/RUN", light_mode="Bright+Forward", cue="Illuminate ahead", mode_color=(255,255,0))
        elif walking_speed < SPEED_THRESHOLDS['slow']:
            result.update(gait_status="SLOW", light_mode="Warm, Diffuse", cue="Soft guidance", mode_color=(0,128,255))
        else:
            result.update(gait_status="NORMAL", light_mode="Adaptive", mode_color=(0,255,0))
        if is_panic_run_detected(walking_speed):
            result.update(light_mode="Panic/Evacuation", cue="Flash Red, Notify!", mode_color=(0,0,255))
        result.update(walking_speed=walking_speed, stride_var=stride_var)
    return result

lat, lon, detected_city = get_my_location()
print(f"Detected location: {detected_city} ({lat}, {lon})")
weather = get_weather(API_KEY, lat, lon)
weather_last_update = time.time()
cap = cv2.VideoCapture(0)
city_data_log = []
prev_time = 0

# == MAIN LOOP ==
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_resized = cv2.resize(frame, (640, 480))
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # -- MULTI-PERSON DETECTION MODE (YOLO + Pose) --
    persons_results = []
    box_count = 0
    if ENABLE_YOLO:
        preds = yolo_model.predict(frame_resized, classes=[0], verbose=False)[0]
        for det in preds.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
            box_h = y2 - y1
            box_w = x2 - x1
            # Skip too small crops (partial, sitting, etc)
            if box_h < 60 or box_w < 45: continue
            person_crop = img_rgb[y1:y2, x1:x2] if y1<y2 and x1<x2 else img_rgb
            result = pose.process(person_crop)
            ankle_history = deque(maxlen=WINDOW_SIZE)
            if result.pose_landmarks:
                # Check good ankle visibility (more robust for partial pose)
                la, ra = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE], result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                if la.visibility > POSE_VIS_THRESH and ra.visibility > POSE_VIS_THRESH:
                    avg_ankle_y = (la.y + ra.y) / 2
                    ankle_history.append(avg_ankle_y)
                    gait = analyze_gait(ankle_history)
                    persons_results.append({"box":(x1,y1,x2,y2), **gait})
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), gait["mode_color"], 2)
                    cv2.putText(frame_resized, f"{gait['gait_status']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gait["mode_color"], 2)
                    box_count += 1
                else:
                    # Low confidence/partial person - grey box
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (128,128,128), 1)
            else:
                # No pose - but person still detected
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame_resized, "NO GAIT", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    else:
        # SINGLE PERSON MODE (Pose only)
        ankle_history = deque(maxlen=WINDOW_SIZE)
        result = pose.process(img_rgb)
        if result.pose_landmarks:
            la, ra = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE], result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            # Accept valid ankles only if standing/walking
            if la.visibility > POSE_VIS_THRESH and ra.visibility > POSE_VIS_THRESH:
                avg_ankle_y = (la.y + ra.y)/2
                ankle_history.append(avg_ankle_y)
                gait = analyze_gait(ankle_history)
                persons_results.append({"box":None, **gait})
                mp_draw.draw_landmarks(frame_resized, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if not persons_results:
            # Detect if any keypoint is visible (sitting/partial), fallback display
            if result.pose_landmarks and any(lm.visibility > 0.3 for lm in result.pose_landmarks.landmark):
                cv2.putText(frame_resized, "PERSON DETECTED (SITTING/PARTIAL)", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
            else:
                cv2.putText(frame_resized, "NO PERSON", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # -- AGGREGATE OVERLAY --
    ambient = get_ambient_light_level()
    if time.time() - weather_last_update > 60:
        weather = get_weather(API_KEY, lat, lon)
        weather_last_update = time.time()
    sound = get_sound_level()
    accessibility_mode = user_assist_button()

    summary = "; ".join([
        f"[{i+1}] {p['gait_status']} (Speed: {p.get('walking_speed',0):.4f})"
        for i, p in enumerate(persons_results)
    ])
    cv2.rectangle(frame_resized, (10,10), (630,120), (100,200,200), -1)
    cv2.putText(frame_resized, f'People: {len(persons_results)} {summary}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(frame_resized, f'Env:{ambient}, {weather} City:{detected_city}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,70,70), 2)
    cv2.putText(frame_resized, f'Edge AI Privacy', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10,10,10), 2)
    curr_time = time.time()
    fps = int(1/(curr_time-prev_time)) if prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(frame_resized, f'FPS: {fps}', (550,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,255), 2)
    cv2.imshow('AI Smart Streetlight: Multi-person Gait', frame_resized)

    city_data_log.append({
        'timestamp': time.time(),
        'people': len(persons_results),
        'gaits': [p["gait_status"] for p in persons_results] if persons_results else ["None"],
        'env': (ambient,weather),
        'sound': sound
    })
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\n--- Urban Analytics Log Sample (10 rows) ---")
for record in city_data_log[-10:]:
    print(record)
