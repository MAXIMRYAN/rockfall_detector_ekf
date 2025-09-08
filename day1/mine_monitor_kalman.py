# mine_monitor_enhanced.py
import cv2
import numpy as np
from ultralytics import YOLO
import time
import pygame
import threading

# Initialize pygame for audio alerts
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound("alarm.wav")  # Add an alarm sound file
except:
    print("Warning: Could not load alarm sound file. Using default beep.")
    # Create a simple beep sound
    sample_rate = 44100
    duration = 0.5  # seconds
    frequency = 880  # Hz
    frames = int(duration * sample_rate)
    arr = np.zeros((frames, 2), dtype=np.int16)
    for i in range(frames):
        arr[i, 0] = int(32767.0 * np.sin(2 * np.pi * frequency * i / sample_rate))
        arr[i, 1] = arr[i, 0]
    alarm_sound = pygame.mixer.Sound(buffer=arr)

# Kalman Filter class for object tracking
class KalmanFilter:
    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=10.0):
        # State transition matrix (constant velocity model)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        # Measurement matrix (we only measure position)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        # State covariance
        self.P = np.eye(4, dtype=np.float32)
        
        # State estimate
        self.x = np.zeros((4, 1), dtype=np.float32)
        
        # Initialized flag
        self.initialized = False
    
    def init(self, x, y):
        self.x = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000  # Large initial uncertainty
        self.initialized = True
    
    def predict(self):
        if not self.initialized:
            return None, None
            
        # Predict state
        self.x = self.F @ self.x
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Return predicted position
        return self.x[0, 0], self.x[1, 0]
    
    def update(self, z_x, z_y):
        if not self.initialized:
            self.init(z_x, z_y)
            return z_x, z_y
            
        # Measurement vector
        z = np.array([[z_x], [z_y]], dtype=np.float32)
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + K @ y
        
        # Update covariance estimate
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        # Return updated position
        return self.x[0, 0], self.x[1, 0]
    
    def get_velocity(self):
        if not self.initialized:
            return 0, 0
        return self.x[2, 0], self.x[3, 0]

# Global variables
points = []
alarm_active = False
alarm_start_time = 0
alert_cooldown = 2  # seconds between alerts

# Mouse callback function
def draw_danger_zone(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        points.pop()  # Remove last point on right click
        print("Last point removed")

# Function to play alarm sound in a separate thread
def play_alarm_sound():
    try:
        alarm_sound.play()
    except:
        print("Error playing alarm sound")

# Initialize webcam and get frame for zone setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from webcam")
    cap.release()
    exit()

# Create a window and set the mouse callback
cv2.namedWindow('Define Danger Zone')
cv2.setMouseCallback('Define Danger Zone', draw_danger_zone)

print("Left-click to define the danger zone polygon.")
print("Right-click to remove the last point.")
print("Press 'ENTER' when done, 'q' to quit.")
print("Define at least 3 points to create a polygon.")

while True:
    img = frame.copy()
    
    # Draw instructions on the image
    cv2.putText(img, "Left-click: Add point", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Right-click: Remove point", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "ENTER: Confirm, Q: Quit", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw all points and connect them with lines
    for i, point in enumerate(points):
        cv2.circle(img, point, 5, (0, 0, 255), -1)
        if i > 0:
            cv2.line(img, points[i-1], point, (0, 0, 255), 2)
    
    # Connect the first and last point to close the polygon
    if len(points) > 2:
        cv2.line(img, points[-1], points[0], (0, 0, 255), 2)
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array(points)], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    cv2.imshow('Define Danger Zone', img)
    key = cv2.waitKey(1) & 0xFF

    if key == 13 and len(points) > 2:  # ENTER key
        print("Danger zone defined. Points:", points)
        break
    elif key == ord('q'):
        print("Setup cancelled")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# --- MAIN MONITORING LOOP ---
model = YOLO('yolov8n.pt')
track_history = {}
object_counter = 0
alert_count = 0

# Persistent sensor states
soil_moisture = 20.0
soil_type = "Loam"
vibration = 2.0
rain_level = 0.0
rain_active = False
ground_tilt = 0.2
tilt_active = False
temperature = 25.0

# For debug printing
last_landslide_risk = -1
last_rockfall_risk = -1

print("Starting monitoring. Press 'q' to quit.")
print("Press 'b' for blast, 'r' for rain, 't' for freeze-thaw, 's' for sandy soil")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # 🔥 1. READ KEY
    key = cv2.waitKey(1) & 0xFF

    # --- 2. UPDATE PERSISTENT SENSORS ---
    # Vibration
    if key == ord('b'):
        vibration = 18.0
    else:
        vibration = max(2.0, vibration - 0.3)

    # Rain
    if key == ord('r'):
        rain_active = True
        rain_level = 0.9
    else:
        rain_active = False
    if not rain_active:
        rain_level = max(0.0, rain_level - 0.05)

    # Ground tilt
    if key == ord('c'):
        tilt_active = True
        ground_tilt = 0.7
    else:
        tilt_active = False
    if not tilt_active:
        ground_tilt = max(0.2, ground_tilt - 0.05)

    # Temperature
    if key == ord('t'):
        temperature = -5
    else:
        temperature = min(25.0, temperature + 0.5)

    # Soil type
    if key == ord('s'):
        soil_type = "Sand"

    # Soil moisture accumulation
    if rain_level > 0.7:
        soil_moisture = min(70.0, soil_moisture + 0.8)
    else:
        soil_moisture = max(20.0, soil_moisture - 0.2)

    # Soil cohesion and effective moisture
    if soil_type == "Sand":
        soil_cohesion = 30
        moisture_multiplier = 1.5
    elif soil_type == "Clay":
        soil_cohesion = 70
        moisture_multiplier = 0.8
    else:
        soil_cohesion = 50
        moisture_multiplier = 1.0

    effective_moisture = soil_moisture * moisture_multiplier

    # --- 3. DETECT OBJECTS (Rockfall) ---
    results = model.predict(frame, verbose=False, conf=0.4)
    annotated_frame = results[0].plot()

    # --- 3.1 DRAW DANGER ZONE ON LIVE FEED ---
    if len(points) < 3:
        cv2.putText(annotated_frame, "⚠️ Define danger zone!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow('Mine Safety Monitor', annotated_frame)
        if key == ord('q'):
            break
        continue

    danger_zone = np.array(points, np.int32)
    cv2.polylines(annotated_frame, [danger_zone], True, (0, 0, 255), 2)
    overlay = annotated_frame.copy()
    cv2.fillPoly(overlay, [danger_zone], (0, 0, 255))
    cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)

    # --- 4. PROCESS DETECTIONS WITH KALMAN FILTER ---
    boxes = results[0].boxes
    danger_detected = False
    current_time = time.time()

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # Track objects
            object_id = None
            min_dist = 25
            for oid, history in track_history.items():
                if history['positions']:
                    last_pos = history['positions'][-1]
                    dist = np.sqrt((x_center - last_pos[0])**2 + (y_center - last_pos[1])**2)
                    if dist < min_dist:
                        object_id = oid
                        break

            if object_id is None:
                object_id = object_counter
                object_counter += 1
                track_history[object_id] = {
                    'positions': [],
                    'first_seen': current_time,
                    'color': np.random.randint(0, 255, 3).tolist(),
                    'kf': KalmanFilter(dt=0.1, process_noise=0.1, measurement_noise=5.0)
                }
                # Initialize Kalman filter with first measurement
                track_history[object_id]['kf'].init(x_center, y_center)
            else:
                # Update Kalman filter with new measurement
                filtered_x, filtered_y = track_history[object_id]['kf'].update(x_center, y_center)
                x_center, y_center = int(filtered_x), int(filtered_y)

            track_history[object_id]['positions'].append((x_center, y_center))
            if len(track_history[object_id]['positions']) > 20:
                track_history[object_id]['positions'].pop(0)

            # Draw trajectory
            positions = track_history[object_id]['positions']
            color = track_history[object_id]['color']
            if len(positions) > 1:
                for j in range(1, len(positions)):
                    cv2.line(annotated_frame, positions[j-1], positions[j], color, 2)

            # Draw ID
            cv2.putText(annotated_frame, f"ID:{object_id}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Predict future path using Kalman filter
            if len(positions) >= 3:  # Need enough points for prediction
                # Get velocity from Kalman filter
                vx, vy = track_history[object_id]['kf'].get_velocity()
                
                # Predict multiple steps ahead based on velocity
                prediction_steps = 5
                future_x = x_center + vx * prediction_steps
                future_y = y_center + vy * prediction_steps
                
                # Draw prediction
                cv2.circle(annotated_frame, (int(future_x), int(future_y)), 8, (0, 255, 255), -1)
                cv2.line(annotated_frame, (x_center, y_center), (int(future_x), int(future_y)), (0, 255, 255), 2)
                
                # Draw velocity vector
                cv2.putText(annotated_frame, f"v:({vx:.1f},{vy:.1f})", 
                           (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                # Check if predicted point is in danger zone
                predicted_point = (int(future_x), int(future_y))
                if cv2.pointPolygonTest(danger_zone, predicted_point, False) >= 0:
                    danger_detected = True

            # Check current position
            if cv2.pointPolygonTest(danger_zone, (x_center, y_center), False) >= 0:
                danger_detected = True

    # --- 5. COMPUTE RISKS ---
    # Rockfall Risk (more sensitive)
    rockfall_risk = 0
    if danger_detected: 
        rockfall_risk += 80  # Increased from 50 to trigger faster
    if vibration > 10: 
        rockfall_risk += 15
    if rain_level > 0.5: 
        rockfall_risk += 15
    if len(track_history) > 5: 
        rockfall_risk += 10
    rockfall_risk = max(0, min(100, rockfall_risk))

    # Landslide Risk
    landslide_risk = 0
    if effective_moisture > 40:     landslide_risk += 30
    if ground_tilt > 0.5:           landslide_risk += 25
    if rain_level > 0.7:            landslide_risk += 20
    if soil_cohesion < 50:          landslide_risk += 15
    if temperature < 0:             landslide_risk += 10
    landslide_risk = min(100, landslide_risk)

    # --- 6. DEBUG: Print only when risk changes ---
    if (round(landslide_risk) != round(last_landslide_risk) or 
        round(rockfall_risk) != round(last_rockfall_risk)):

        print(f"📊 RISK UPDATE | Rockfall: {rockfall_risk:.0f}% | Landslide: {landslide_risk:.0f}%")
        if landslide_risk > 0:
            print(f"   → Moist: {effective_moisture:.1f}% | Tilt: {ground_tilt:.2f}° | Temp: {temperature:.0f}°C | Soil: {soil_type}")
        print("-" * 50)

        last_landslide_risk = landslide_risk
        last_rockfall_risk = rockfall_risk

    # --- 7. DRAW UI ---
    UI_WIDTH = 280
    UI_HEIGHT = 240
    x0, y0 = 10, 10

    # Background panel
    cv2.rectangle(annotated_frame, (x0, y0), (x0 + UI_WIDTH, y0 + UI_HEIGHT), (30, 30, 30), -1)
    cv2.rectangle(annotated_frame, (x0, y0), (x0 + UI_WIDTH, y0 + UI_HEIGHT), (100, 100, 100), 1)

    # Title
    cv2.putText(annotated_frame, "Mine Safety Monitor", (x0 + 10, y0 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Status
    any_alert = rockfall_risk > 70 or landslide_risk > 70
    status_color = (0, 0, 255) if any_alert else (0, 255, 0)
    cv2.putText(annotated_frame, f"Status: {'ALERT!' if any_alert else 'Normal'}",
                (x0 + 10, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

    # Alerts & Objects
    cv2.putText(annotated_frame, f"Alerts: {alert_count} | Objects: {len(track_history)}",
                (x0 + 10, y0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Risk Bars
    def draw_risk_bar(img, label, risk, y_pos):
        cv2.putText(img, label, (x0 + 10, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(img, (x0 + 80, y_pos), (x0 + 250, y_pos + 12), (50, 50, 50), -1)
        bar_width = int(170 * (risk / 100))
        color = (0, 255, 0) if risk < 30 else (0, 165, 255) if risk < 70 else (0, 0, 255)
        cv2.rectangle(img, (x0 + 80, y_pos), (x0 + 80 + bar_width, y_pos + 12), color, -1)
        cv2.putText(img, f"{int(risk)}%", (x0 + 255, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    draw_risk_bar(annotated_frame, "Rockfall:", rockfall_risk, y0 + 90)
    draw_risk_bar(annotated_frame, "Landslide:", landslide_risk, y0 + 115)

    # Sensor Readouts
    cv2.putText(annotated_frame, f"Vib: {vibration:.1f} | Rain: {rain_level:.1f}",
                (x0 + 10, y0 + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(annotated_frame, f"Moist: {effective_moisture:.1f}% | Tilt: {ground_tilt:.2f}°",
                (x0 + 10, y0 + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    cv2.putText(annotated_frame, f"Soil: {soil_type} | Temp: {temperature:.0f}°C",
                (x0 + 10, y0 + 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Key hints
    cv2.putText(annotated_frame, "Keys: B/R/T/C/S/Q", (x0 + 10, y0 + 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # On-screen warnings
    if danger_detected:
        cv2.putText(annotated_frame, "ROCKFALL DETECTED!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if landslide_risk > 70:
        cv2.putText(annotated_frame, "LANDSLIDE WARNING!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # --- 8. TRIGGER ALARM ---
    if any_alert:
        if not alarm_active or (current_time - alarm_start_time > alert_cooldown):
            alarm_active = True
            alarm_start_time = current_time
            alert_count += 1
            print(f"🚨 DANGER! Rockfall: {rockfall_risk:.1f}% | Landslide: {landslide_risk:.1f}%")
            threading.Thread(target=play_alarm_sound).start()
    else:
        alarm_active = False

    # --- 9. DISPLAY ---
    cv2.imshow('Mine Safety Monitor', annotated_frame)

    # --- 10. CONTROL KEYS ---
    if key == ord('q'):
        break
    elif key == ord('z'):
        alert_count = 0
        print("🔁 Alert counter reset")
    elif key == ord('c'):
        track_history = {}
        print("🧹 All tracks cleared")

# CLEANUP
cap.release()
cv2.destroyAllWindows()
print("✅ Monitoring stopped. Great job!")