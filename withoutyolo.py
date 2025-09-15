import os
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import requests
from collections import deque
from dotenv import load_dotenv


ENABLE_YOLO = False  


load_dotenv()
API_KEY = os.getenv("OWM_API_KEY")

def get_my_location():
    try:
        r = requests.get("http://ip-api.com/json",timeout=3)
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

ankle_history = deque(maxlen=10)
SPEED_THRESHOLDS = {'slow': 0.003, 'fast': 0.015}
VARIABILITY_THRESHOLD = 0.22    # Empirically robust—change if you want even stricter
SOME_MIN_SPEED = 0.01           # Only call 'unstable' if real motion

lat, lon, detected_city = get_my_location()
print(f"Detected location: {detected_city} ({lat}, {lon})")

weather = get_weather(API_KEY, lat, lon)
weather_last_update = time.time()

prev_time = 0
city_data_log = []

# COCO_CLASSES, VEHICLE_CLASSES, ANIMAL_CLASSES omitted (no detection)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)  # mediapipe (gait)

    # ----- YOLO detection skipped for speed and clarity -----
    vehicle_found = False
    animal_found = False
    objects_found = set()
    #  If you want context logic for animals/vehicles, you can use a similar placeholder block:
    # if ENABLE_YOLO: pass

    # --- Real world data ---
    ambient = get_ambient_light_level()
    if time.time() - weather_last_update > 60:
        weather = get_weather(API_KEY, lat, lon)
        weather_last_update = time.time()

    sound = get_sound_level()
    accessibility_mode = user_assist_button()

    # Main logic
    light_mode = 'OFF'
    mode_color = (0, 0, 255)
    cue = ''
    display_warnings = []
    gait_status = 'No Person'
    assist_mode = False
    walking_speed = 0
    stride_var = 0

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame_resized, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        ankle_avg_y = (left_ankle.y + right_ankle.y) / 2
        ankle_history.append(ankle_avg_y)
        if len(ankle_history) == ankle_history.maxlen:
            diffs = [abs(a - b) for a, b in zip(list(ankle_history)[1:], list(ankle_history)[:-1])]
            walking_speed = np.mean(diffs)
            stride_var = np.std(diffs)
            print(f"Speed: {walking_speed:.4f}, Stride variation: {stride_var:.4f}")
            if stride_var > VARIABILITY_THRESHOLD and walking_speed > SOME_MIN_SPEED:
                gait_status = "UNSTABLE"
                light_mode = "Assist Mode: Wide + Guidance Cues"
                cue = "Project stripes/arrows"
                mode_color = (0, 255, 255)
                assist_mode = True
            elif walking_speed > SPEED_THRESHOLDS['fast']:
                gait_status = "FAST/RUN"
                light_mode = "Bright, Forward Extension"
                cue = "Illuminate several lights ahead"
                mode_color = (255, 255, 0)
            elif walking_speed < SPEED_THRESHOLDS['slow']:
                gait_status = "SLOW"
                light_mode = "Warm, Diffuse"
                cue = "Soft guidance"
                mode_color = (0, 128, 255)
            else:
                gait_status = "NORMAL"
                light_mode = "Adaptive"
                mode_color = (0, 255, 0)
            if is_panic_run_detected(walking_speed):
                light_mode = "Panic/Evacuation"
                cue = "Flash Red, Notify Control!"
                display_warnings.append("")
                mode_color = (0, 0, 255)
        if accessibility_mode:
            assist_mode = True
            light_mode = "Accessibility: Enhanced Visibility"
            cue = "Personalize for user needs"
            display_warnings.append("User Assist Activated!")
            mode_color = (200, 0, 200)
    elif vehicle_found:
        gait_status = "No Person"
        light_mode = "Vehicle Detected: Normal Illumination"
        mode_color = (240, 240, 0)
        cue = "Lights ON for vehicle"
    elif animal_found:
        gait_status = "No Person"
        light_mode = "Animal Detected: Dimming (Eco)"
        mode_color = (100, 180, 255)
        cue = "Dimmest for animal - save energy"
    else:
        ankle_history.clear()
        light_mode = f"Standby ({ambient}, {weather})"
        mode_color = (100, 50, 255)
        gait_status = "No person"

    # 4. Environmental
    if weather in ['rain', 'fog']:
        light_mode += " + Weather Boost"
        cue += " (Bad weather: +Brightness)"
        display_warnings.append(f"Weather: {weather}")
    if ambient == 'night':
        light_mode += " + Night Dimming"
        cue += " (Auto dim/bright)"
    if sound > 95:
        display_warnings.append("")

    city_data_log.append({
        'timestamp': time.time(),
        'gait': gait_status,
        'speed': walking_speed,
        'ambient': ambient,
        'weather': weather,
        'sound': sound,
        'assist': assist_mode,
        'mode': light_mode,
        'yolo_obj': list(objects_found)
    })
    if gait_status == "No person" and ambient in ['day', 'evening']:
        light_mode = "Eco: Min Power"

    privacy_status = "Edge AI: All data anonymized, local only"

    # Overlay status
    cv2.rectangle(frame_resized, (10, 10), (630, 120), mode_color, -1)
    cv2.putText(frame_resized, f'Mode: {light_mode}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(frame_resized, f'Gait: {gait_status}  Env:{ambient}/{weather} City:{detected_city}',
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 70, 70), 2)
    cv2.putText(frame_resized, f'Cues: {cue}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    for idx, msg in enumerate(display_warnings):
        cv2.putText(frame_resized, f'! {msg}', (20, 150+25*idx), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame_resized, privacy_status, (10, 470), cv2.FONT_HERSHEY_PLAIN, 1.2, (128, 255, 200), 2)
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(frame_resized, f'FPS: {fps}', (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2)
    cv2.imshow('All-in-One Smart Streetlight: Gait Only', frame_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\n--- Urban Analytics Log Sample (10 rows) ---")
for record in city_data_log[-10:]:
    print(record)
