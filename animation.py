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
import csv
from datetime import datetime

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# == SETTINGS ==
ENABLE_YOLO = YOLO_AVAILABLE   # Toggle YOLO multi-person detection if installed
POSE_VIS_THRESH = 0.5          # MediaPipe landmark minimum visibility

FRAME_W, FRAME_H = 1280, 720  # Reduced resolution for better performance

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

# == SIMULATION FUNCTIONS ==
def create_street_simulation(width, height, persons_results, ambient, weather):
    # Create a blank simulation canvas
    simulation = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw sky based on ambient light
    if ambient == 'day':
        sky_color = (220, 220, 180)  # Light blue
    elif ambient == 'evening':
        sky_color = (120, 100, 180)  # Orange/purple
    else:
        sky_color = (30, 30, 60)     # Dark blue
    
    # Draw sky
    cv2.rectangle(simulation, (0, 0), (width, height//2), sky_color, -1)
    
    # Draw ground
    ground_color = (50, 50, 50) if ambient == 'night' else (80, 80, 80)
    cv2.rectangle(simulation, (0, height//2), (width, height), ground_color, -1)
    
    # Draw road markings
    road_marking_color = (200, 200, 200) if ambient != 'night' else (150, 150, 150)
    for i in range(0, width, 40):
        cv2.rectangle(simulation, (i, height//2 + 20), (i+20, height//2 + 30), road_marking_color, -1)
    
    # Draw streetlight pole
    pole_x = width // 2
    pole_height = height // 3
    cv2.rectangle(simulation, (pole_x, height//2 - pole_height), (pole_x+10, height//2), (100, 100, 100), -1)
    
    # Draw streetlight based on detected persons
    light_color = (30, 30, 30)  # Default off (dark)
    light_radius = 0
    
    if persons_results:
        # Get the highest priority person
        priority_order = {'Panic/Evacuation':5, 'UNSTABLE':4, 'FAST/RUN':3, 'SLOW':2, 'NORMAL':1, 'OFF':0, 'Assist Mode: Wide+Guidance':4}
        best = None
        best_score = -1
        for p in persons_results:
            lm = p.get('light_mode','OFF')
            score = priority_order.get(lm, 0)
            if score > best_score:
                best_score = score
                best = p
        
        if best:
            light_color = best.get('mode_color', (30, 30, 30))
            # Adjust light intensity based on walking speed
            sp = best.get('walking_speed', 0)
            light_radius = min(int(100 + sp * 5000), 200)
    
    # Draw streetlight bulb
    bulb_center = (pole_x + 5, height//2 - pole_height)
    cv2.circle(simulation, bulb_center, 15, light_color, -1)
    
    # Draw light cone if light is on
    if light_radius > 0:
        # Create a mask for the light cone
        light_mask = np.zeros((height, width, 3), dtype=np.uint8)
        pts = np.array([[pole_x, height//2 - pole_height], 
                        [pole_x - light_radius, height//2 + 50], 
                        [pole_x + light_radius, height//2 + 50]], np.int32)
        cv2.fillPoly(light_mask, [pts], light_color)
        
        # Blend the light cone with the simulation
        simulation = cv2.addWeighted(simulation, 1.0, light_mask, 0.3, 0)
    
    # Draw persons in the simulation
    person_positions = []
    if persons_results:
        for i, person in enumerate(persons_results):
            # Calculate position based on person index
            x_pos = width // (len(persons_results) + 1) * (i + 1)
            y_pos = height//2 + 20  # On the ground
            
            # Draw person (simple stick figure)
            color = person.get('mode_color', (0, 255, 0))
            cv2.circle(simulation, (x_pos, y_pos), 10, color, -1)  # Head
            cv2.line(simulation, (x_pos, y_pos+10), (x_pos, y_pos+40), color, 2)  # Body
            cv2.line(simulation, (x_pos, y_pos+20), (x_pos-15, y_pos+10), color, 2)  # Left arm
            cv2.line(simulation, (x_pos, y_pos+20), (x_pos+15, y_pos+10), color, 2)  # Right arm
            cv2.line(simulation, (x_pos, y_pos+40), (x_pos-10, y_pos+60), color, 2)  # Left leg
            cv2.line(simulation, (x_pos, y_pos+40), (x_pos+10, y_pos+60), color, 2)  # Right leg
            
            person_positions.append((x_pos, y_pos))
    
    # Add weather effects
    if weather == 'rain':
        # Draw rain drops
        for _ in range(50):
            x = random.randint(0, width)
            y = random.randint(0, height//2)
            cv2.line(simulation, (x, y), (x+2, y+8), (200, 200, 200), 1)
    elif weather == 'snow':
        # Draw snow flakes
        for _ in range(30):
            x = random.randint(0, width)
            y = random.randint(0, height//2)
            cv2.circle(simulation, (x, y), 2, (255, 255, 255), -1)
    
    return simulation, person_positions

# == CSV LOGGING INIT ==
def save_city_data_log(log, filename=None):
    filename = filename or f"analytics_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'people', 'gaits', 'env', 'sound']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in log:
            # Convert tuple/list to string for CSV
            entry_copy = entry.copy()
            entry_copy['gaits'] = '|'.join(entry.get('gaits', []))
            entry_copy['env']   = '|'.join([str(x) for x in entry.get('env',())])
            writer.writerow(entry_copy)
    print(f"\n[INFO] Analytics log saved to: {filename}")

# Initialize location and camera
lat, lon, detected_city = get_my_location()
print(f"Detected location: {detected_city} ({lat}, {lon})")
weather = get_weather(API_KEY, lat, lon)
weather_last_update = time.time()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# Fullscreen window setup
cv2.namedWindow('AI Smart Streetlight: Multi-person Gait', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('AI Smart Streetlight: Multi-person Gait', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

city_data_log = []
prev_time = 0

# Persistent tracking structures
next_track_id = 0
tracks = {}  # track_id -> { 'center':(x,y), 'box':(x1,y1,x2,y2), 'ankle_hist': deque, 'last_seen': timestamp }
MAX_MATCH_DIST = 150  # pixels (tune for your camera)
TRACK_TTL = 0.5       # seconds to keep a track not seen before deleting
single_ankle_history = deque(maxlen=WINDOW_SIZE)  # used when YOLO is disabled

print("\n--- Press ESC in the camera window to save analytics log and exit ---")

# == MAIN LOOP ==
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H))
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    persons_results = []
    detected_centers = []
    detections_for_matching = []  # list of (box, center, ankle_y_norm or None)

    # ---------- DETECTION & POSE (YOLO multi) ----------
    if ENABLE_YOLO:
        preds = yolo_model.predict(frame_resized, classes=[0], verbose=False)[0]
        for det in preds.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
            box_h = y2 - y1
            box_w = x2 - x1
            if box_h < 100 or box_w < 60:  # Reduced threshold for better detection
                continue
            # crop and run pose on crop
            crop_x1, crop_y1, crop_x2, crop_y2 = max(x1,0), max(y1,0), min(x2, FRAME_W), min(y2, FRAME_H)
            person_crop = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2] if crop_y2>crop_y1 and crop_x2>crop_x1 else img_rgb
            result = pose.process(person_crop)
            ankle_y_norm = None
            if result.pose_landmarks:
                la = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                ra = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                if la.visibility > POSE_VIS_THRESH and ra.visibility > POSE_VIS_THRESH:
                    # la.y and ra.y are relative to the crop; average them
                    avg_ankle_y = (la.y + ra.y) / 2.0
                    # convert to absolute pixel y in original frame for matching stability if desired:
                    abs_ankle_pixel_y = crop_y1 + avg_ankle_y * (crop_y2 - crop_y1)
                    # normalize to 0..1 relative to full frame height to match existing analyze_gait expectation
                    ankle_y_norm = (abs_ankle_pixel_y / FRAME_H)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            detections_for_matching.append(((x1,y1,x2,y2), center, ankle_y_norm))
            detected_centers.append(center)
    else:
        # single person pose on full frame
        result = pose.process(img_rgb)
        ankle_y_norm = None
        if result.pose_landmarks:
            la = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            ra = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            if la.visibility > POSE_VIS_THRESH and ra.visibility > POSE_VIS_THRESH:
                avg_ankle_y = (la.y + ra.y) / 2.0
                ankle_y_norm = avg_ankle_y  # normalized to full frame since pose was run on full
                single_ankle_history.append(ankle_y_norm)
                gait = analyze_gait(single_ankle_history)
                persons_results.append({"box":None, **gait})
                mp_draw.draw_landmarks(frame_resized, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if not persons_results:
            if result.pose_landmarks and any(lm.visibility > 0.3 for lm in result.pose_landmarks.landmark):
                cv2.putText(frame_resized, "PERSON DETECTED (SITTING/PARTIAL)", (60,160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,0), 2)
            else:
                cv2.putText(frame_resized, "NO PERSON", (60,160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # ---------- SIMPLE TRACKING (centroid distance) ----------
    now_ts = time.time()
    if ENABLE_YOLO:
        # match detections to existing tracks by center distance
        used_track_ids = set()
        new_tracks = {}

        for det_box, det_center, det_ankle_norm in detections_for_matching:
            best_id = None
            best_dist = None
            for tid, t in tracks.items():
                if tid in used_track_ids:
                    continue
                tx, ty = t['center']
                dist = np.hypot(det_center[0]-tx, det_center[1]-ty)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_dist is not None and best_dist < MAX_MATCH_DIST:
                # match found
                tid = best_id
                used_track_ids.add(tid)
                t = tracks[tid]
                t['center'] = det_center
                t['box'] = det_box
                t['last_seen'] = now_ts
                # ensure ankle_history exists
                if 'ankle_hist' not in t:
                    t['ankle_hist'] = deque(maxlen=WINDOW_SIZE)
                if det_ankle_norm is not None:
                    t['ankle_hist'].append(det_ankle_norm)
                    gait = analyze_gait(t['ankle_hist'])
                else:
                    gait = {"gait_status":"NO GAIT", "light_mode":"OFF", "cue":"", "mode_color":(128,128,128), "assist_mode":False, "walking_speed":0, "stride_var":0}
                persons_results.append({"track_id":tid, "box":det_box, **gait})
                new_tracks[tid] = t
            else:
                # create new track
                tid = next_track_id
                next_track_id += 1
                t = {
                    'center': det_center,
                    'box': det_box,
                    'last_seen': now_ts,
                    'ankle_hist': deque(maxlen=WINDOW_SIZE)
                }
                if det_ankle_norm is not None:
                    t['ankle_hist'].append(det_ankle_norm)
                gait = analyze_gait(t['ankle_hist']) if det_ankle_norm is not None else {"gait_status":"NO GAIT", "light_mode":"OFF", "cue":"", "mode_color":(128,128,128), "assist_mode":False, "walking_speed":0, "stride_var":0}
                persons_results.append({"track_id":tid, "box":det_box, **gait})
                new_tracks[tid] = t

        # keep tracks that were recently seen (to maintain histories briefly)
        for tid, t in tracks.items():
            if tid not in new_tracks:
                if now_ts - t.get('last_seen',0) < TRACK_TTL:
                    new_tracks[tid] = t  # keep it alive briefly
        tracks = new_tracks

        # draw boxes and labels
        for p in persons_results:
            box = p.get('box')
            color = p.get('mode_color', (0,255,0))
            label = p.get('gait_status', 'No Gait')
            if box:
                x1,y1,x2,y2 = box
                cv2.rectangle(frame_resized, (x1,y1), (x2,y2), color, 3)
                cv2.putText(frame_resized, f"{label}", (x1, max(y1-30, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                # small track id
                if 'track_id' in p:
                    cv2.putText(frame_resized, f"ID:{p['track_id']}", (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # ---------- ENV & UI OVERLAYS ----------
    ambient = get_ambient_light_level()
    if time.time() - weather_last_update > 60:
        weather = get_weather(API_KEY, lat, lon)
        weather_last_update = time.time()
    sound = get_sound_level()
    accessibility_mode = user_assist_button()

    # info box (top-left)
    summary = "; ".join([
        f"[{i+1}] {p['gait_status']} (Sp:{p.get('walking_speed',0):.4f})"
        for i, p in enumerate(persons_results)
    ]) or "No detections"
    cv2.rectangle(frame_resized, (40,40), (1100,300), (100,200,200), -1)
    cv2.putText(frame_resized, f'People: {len(persons_results)}  {summary}', (60,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(frame_resized, f'Env:{ambient}, {weather}  City:{detected_city}', (60,180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10,70,70), 2)
    cv2.putText(frame_resized, f'Edge AI Privacy', (60,240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10,10,10), 2)

    # FPS
    curr_time = time.time()
    fps = int(1/(curr_time-prev_time)) if prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(frame_resized, f'FPS: {fps}', (1650,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,255), 2)

    # ---------- CREATE SIMULATION VIEW ----------
    simulation_width = FRAME_W // 2
    simulation_height = FRAME_H
    simulation_view, person_positions = create_street_simulation(
        simulation_width, simulation_height, persons_results, ambient, weather
    )
    
    # Combine camera view and simulation view
    combined_view = np.hstack((frame_resized, simulation_view))
    
    # Add separator line
    cv2.line(combined_view, (FRAME_W, 0), (FRAME_W, FRAME_H), (255, 255, 255), 2)
    
    # Add labels
    cv2.putText(combined_view, "CAMERA VIEW", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined_view, "STREET SIMULATION", (FRAME_W + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # show camera + simulation
    cv2.imshow('AI Smart Streetlight: Multi-person Gait', combined_view)

    # logging
    city_data_log.append({
        'timestamp': time.time(),
        'people': len(persons_results),
        'gaits': [p["gait_status"] for p in persons_results] if persons_results else ["None"],
        'env': (ambient,weather),
        'sound': sound
    })

    k = cv2.waitKey(1)
    if k & 0xFF == 27: # ESC -> save and exit
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
save_city_data_log(city_data_log)
print("\n--- Urban Analytics Log Sample (10 rows) ---")
for record in city_data_log[-10:]:
    print(record)