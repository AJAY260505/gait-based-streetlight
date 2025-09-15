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
import json
import socket
import threading

# Import PyGame only if available
try:
    import pygame
    from pygame.locals import *
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Pygame not available, using simplified visualization")

# == SETTINGS ==
ENABLE_YOLO = False  # Disable YOLO for better performance on laptop
POSE_VIS_THRESH = 0.5
FRAME_W, FRAME_H = 640, 480  # Lower resolution for better performance
SIMULATION_ENABLED = False  # Disable 3D simulation for better performance

# == Initialize models and utilities ==
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5)  # Lower complexity

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
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
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

def get_temperature(lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=3)
        data = response.json()
        return data['main']['temp']
    except Exception:
        return random.randint(15, 30)

def user_assist_button():
    return random.random() < 0.01

# == GAIT ANALYSIS CONFIG ==
WINDOW_SIZE = 15  # Reduced window size for faster response
SPEED_THRESHOLDS = {'slow': 0.003, 'fast': 0.015}
VARIABILITY_THRESHOLD = 0.22
SOME_MIN_SPEED = 0.01

# == FILTERING ==
def remove_spikes(diff_list, k=3):  # Smaller kernel for performance
    arr = np.array(diff_list)
    if len(arr) < k: return arr
    return medfilt(arr, kernel_size=k)

def analyze_gait(ankle_history):
    result = {"gait_status":"No Gait", "light_mode":"OFF", "cue":"", "mode_color":(0,0,255), 
              "assist_mode":False, "walking_speed":0, "stride_var":0}
    
    if len(ankle_history) >= 2:
        diffs = [abs(a-b) for a,b in zip(list(ankle_history)[1:], list(ankle_history)[:-1])]
        if not diffs:
            return result
            
        smoothed_diffs = remove_spikes(diffs, k=3)
        walking_speed = float(np.mean(smoothed_diffs))
        stride_var = float(np.std(smoothed_diffs)) if len(smoothed_diffs) > 1 else 0
        
        # -- Classify --
        if stride_var > VARIABILITY_THRESHOLD and walking_speed > SOME_MIN_SPEED:
            result.update(gait_status="UNSTABLE", light_mode="Assist Mode: Wide+Guidance", 
                         cue="Project stripes", mode_color=(0,255,255), assist_mode=True)
        elif walking_speed > SPEED_THRESHOLDS['fast']:
            result.update(gait_status="FAST/RUN", light_mode="Bright+Forward", 
                         cue="Illuminate ahead", mode_color=(255,255,0))
        elif walking_speed < SPEED_THRESHOLDS['slow']:
            result.update(gait_status="SLOW", light_mode="Warm, Diffuse", 
                         cue="Soft guidance", mode_color=(0,128,255))
        else:
            result.update(gait_status="NORMAL", light_mode="Adaptive", mode_color=(0,255,0))
        
        result.update(walking_speed=walking_speed, stride_var=stride_var)
    
    return result

# == SIMPLE 2D VISUALIZATION ==
class SimpleStreetlightVisualization:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.persons = []
        self.ambient = 'day'
        self.weather = 'clear'
        self.temperature = 20
        
    def update(self, persons_results, ambient, weather, temperature):
        self.persons = persons_results
        self.ambient = ambient
        self.weather = weather
        self.temperature = temperature
        
        # Clear canvas
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Set background based on ambient light
        if ambient == 'night':
            self.canvas[:] = (20, 20, 40)
        elif ambient == 'evening':
            self.canvas[:] = (50, 40, 30)
        else:  # day
            self.canvas[:] = (120, 150, 200)
            
        # Draw road
        road_y = self.height * 3 // 4
        road_width = self.width // 2
        cv2.rectangle(self.canvas, (self.width//2 - road_width//2, road_y), 
                     (self.width//2 + road_width//2, self.height), (50, 50, 50), -1)
        
        # Draw streetlight
        pole_x = self.width // 2
        pole_height = self.height // 3
        cv2.rectangle(self.canvas, (pole_x-5, road_y - pole_height), 
                     (pole_x+5, road_y), (100, 100, 100), -1)
        
        # Draw light housing
        housing_size = 20
        cv2.rectangle(self.canvas, (pole_x - housing_size//2, road_y - pole_height - 5), 
                     (pole_x + housing_size//2, road_y - pole_height + 15), (80, 80, 80), -1)
        
        # Draw light based on detection
        if persons_results:
            best = max(persons_results, key=lambda x: {'UNSTABLE':4, 'FAST/RUN':3, 'SLOW':2, 'NORMAL':1}.get(x['gait_status'], 0))
            color = best.get('mode_color', (0, 255, 0))
            intensity = min(0.3 + best.get('walking_speed', 0) * 20, 1.0)
            
            # Draw light cone
            cone_points = np.array([
                [pole_x, road_y - pole_height + 15],
                [pole_x - road_width//4, road_y],
                [pole_x + road_width//4, road_y]
            ], np.int32)
            
            light_color = (int(color[0] * intensity), int(color[1] * intensity), int(color[2] * intensity))
            overlay = self.canvas.copy()
            cv2.fillPoly(overlay, [cone_points], light_color)
            cv2.addWeighted(overlay, 0.3, self.canvas, 0.7, 0, self.canvas)
            
            # Draw light source
            cv2.circle(self.canvas, (pole_x, road_y - pole_height + 10), 8, color, -1)
        
        # Draw persons
        for i, person in enumerate(persons_results):
            x_pos = self.width // 2 - 100 + (i * 50)
            y_pos = road_y - 30
            color = person.get('mode_color', (0, 255, 0))
            
            # Draw person (simple stick figure)
            cv2.circle(self.canvas, (x_pos, y_pos), 10, color, -1)  # Head
            cv2.line(self.canvas, (x_pos, y_pos+10), (x_pos, y_pos+30), color, 2)  # Body
            cv2.line(self.canvas, (x_pos, y_pos+15), (x_pos-10, y_pos+5), color, 2)  # Left arm
            cv2.line(self.canvas, (x_pos, y_pos+15), (x_pos+10, y_pos+5), color, 2)  # Right arm
            cv2.line(self.canvas, (x_pos, y_pos+30), (x_pos-10, y_pos+45), color, 2)  # Left leg
            cv2.line(self.canvas, (x_pos, y_pos+30), (x_pos+10, y_pos+45), color, 2)  # Right leg
            
            # Draw label
            cv2.putText(self.canvas, person['gait_status'], (x_pos-30, y_pos-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw weather effects
        if weather == 'rain':
            for _ in range(20):
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)
                cv2.line(self.canvas, (x, y), (x+3, y+10), (200, 200, 255), 1)
        elif weather == 'snow':
            for _ in range(30):
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)
                cv2.circle(self.canvas, (x, y), 2, (255, 255, 255), -1)
                
        # Draw info text
        cv2.putText(self.canvas, f"People: {len(persons_results)}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.canvas, f"Ambient: {ambient}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.canvas, f"Weather: {weather}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.canvas, f"Temp: {temperature}C", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def get_surface(self):
        return self.canvas

# == DATA LOGGING AND REMOTE MONITORING ==
class DataLogger:
    def __init__(self):
        self.log = []
        self.filename = f"gait_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
    def add_entry(self, persons_results, ambient, weather, temperature):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'people': len(persons_results),
            'gaits': [p.get("gait_status", "Unknown") for p in persons_results],
            'env': f"{ambient},{weather}",
            'temperature': temperature
        }
        self.log.append(entry)
        
        # Save to CSV periodically
        if len(self.log) % 10 == 0:
            self.save_to_csv()
            
        return entry
    
    def save_to_csv(self):
        with open(self.filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'people', 'gaits', 'env', 'temperature']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.log:
                entry_copy = entry.copy()
                entry_copy['gaits'] = '|'.join(entry['gaits'])
                writer.writerow(entry_copy)
    
    def save_to_json(self):
        with open(self.filename.replace('.csv', '.json'), 'w') as f:
            json.dump(self.log, f, indent=2)

# == REMOTE STREAMING SERVER (For Raspberry Pi) ==
class StreamingServer:
    def __init__(self, host='0.0.0.0', port=8000):
        self.host = host
        self.port = port
        self.clients = []
        self.running = False
        
    def start_server(self):
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"Streaming server started on {self.host}:{self.port}")
        
    def _run_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        while self.running:
            try:
                client_socket, addr = server_socket.accept()
                print(f"New connection from {addr}")
                self.clients.append(client_socket)
            except:
                break
                
    def broadcast_frame(self, frame):
        if not self.clients:
            return
            
        # Encode frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_data = jpeg.tobytes()
        
        # Header with frame length
        header = len(frame_data).to_bytes(4, byteorder='big')
        
        # Send to all clients
        disconnected_clients = []
        for client in self.clients:
            try:
                client.send(header + frame_data)
            except:
                disconnected_clients.append(client)
                
        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.remove(client)
            try:
                client.close()
            except:
                pass
                
    def stop_server(self):
        self.running = False
        for client in self.clients:
            try:
                client.close()
            except:
                pass

# == MAIN APPLICATION ==
def main():
    # Initialize location and camera
    lat, lon, detected_city = get_my_location()
    print(f"Detected location: {detected_city} ({lat}, {lon})")
    weather = get_weather(API_KEY, lat, lon)
    weather_last_update = time.time()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize visualization
    simulation_width = FRAME_W
    simulation_height = FRAME_H
    simulation = SimpleStreetlightVisualization(simulation_width, simulation_height)
    
    # Initialize data logger
    logger = DataLogger()
    
    # Initialize streaming server (disabled by default for laptop)
    stream_server = StreamingServer()
    # stream_server.start_server()  # Uncomment for Raspberry Pi
    
    # Gait analysis variables
    city_data_log = []
    prev_time = 0
    single_ankle_history = deque(maxlen=WINDOW_SIZE)
    
    print("\n--- Press ESC to exit, 's' to save data ---")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("Failed to grab frame")
                break
                
            frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H))
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            persons_results = []
            
            # Process pose on full frame (single person)
            result = pose.process(img_rgb)
            ankle_y_norm = None
            
            if result.pose_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame_resized, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Extract ankle data
                la = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                ra = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                
                if la.visibility > POSE_VIS_THRESH and ra.visibility > POSE_VIS_THRESH:
                    avg_ankle_y = (la.y + ra.y) / 2.0
                    ankle_y_norm = avg_ankle_y
                    single_ankle_history.append(ankle_y_norm)
                    gait = analyze_gait(single_ankle_history)
                    persons_results.append({"box": None, **gait})
            
            # Update environment data
            ambient = get_ambient_light_level()
            if time.time() - weather_last_update > 60:
                weather = get_weather(API_KEY, lat, lon)
                weather_last_update = time.time()
            temperature = get_temperature(lat, lon)
            
            # Update visualization
            simulation.update(persons_results, ambient, weather, temperature)
            sim_img = simulation.get_surface()
            
            # Log data
            log_entry = logger.add_entry(persons_results, ambient, weather, temperature)
            
            # Calculate FPS
            curr_time = time.time()
            fps = int(1/(curr_time-prev_time)) if prev_time > 0 else 0
            prev_time = curr_time
            
            # Display info on camera frame
            cv2.putText(frame_resized, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if persons_results:
                gait_status = persons_results[0]['gait_status']
                cv2.putText(frame_resized, f'Gait: {gait_status}', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, persons_results[0]['mode_color'], 2)
            
            # Combine camera view and simulation view
            combined_view = np.hstack((frame_resized, sim_img))
            
            # Display
            cv2.imshow('AI Smart Streetlight: Gait Analysis', combined_view)
            
            # Stream to remote clients (if enabled)
            # stream_server.broadcast_frame(combined_view)  # Uncomment for Raspberry Pi
            
            # Handle key presses
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k == ord('s'):  # Save data
                logger.save_to_csv()
                logger.save_to_json()
                print("Data saved!")
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        logger.save_to_csv()
        logger.save_to_json()
        # stream_server.stop_server()  # Uncomment for Raspberry Pi
        print("Exited cleanly. Data saved to", logger.filename)

if __name__ == "__main__":
    main()