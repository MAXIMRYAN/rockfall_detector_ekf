from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import time
import pygame
import threading

# Initialize pygame for audio alerts
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound("alarm.wav")
except:
    print("Warning: Could not load alarm sound file. Using default beep.")
    sample_rate = 44100
    duration = 0.5
    frequency = 880
    frames = int(duration * sample_rate)
    arr = np.zeros((frames, 2), dtype=np.int16)
    for i in range(frames):
        arr[i, 0] = int(32767.0 * np.sin(2 * np.pi * frequency * i / sample_rate))
        arr[i, 1] = arr[i, 0]
    alarm_sound = pygame.mixer.Sound(buffer=arr)

# Global variables
safety_zones = []  # List of safety zone polygons
current_zone_points = []  # Points for the current zone being drawn
drawing_mode = False  # Whether we're in drawing mode
alarm_active = False
alarm_start_time = 0
alert_cooldown = 2  # seconds between alerts
alert_count = 0

# Load model
model = YOLO('yolov8n.pt')

# Dictionary to store track history and Kalman filters
track_history = defaultdict(lambda: [])
kalman_filters = {}
object_counter = 0

# Physics parameters
GRAVITY = 0.5  # pixels/frame² (needs calibration)
PREDICTION_HORIZON = 15  # frames to predict ahead

# Persistent sensor states with simulation modes
soil_moisture = 20.0
soil_type = "Loam"
vibration = 2.0
rain_level = 0.0
ground_tilt = 0.2
temperature = 25.0

# Simulation mode flags (persistent until turned off)
blast_mode = False
rain_mode = False
freeze_thaw_mode = False
sandy_soil_mode = False
clear_zones_mode = False
clear_tracks_mode = False
reset_alerts_mode = False

# For debug printing
last_landslide_risk = -1
last_rockfall_risk = -1

def create_kalman_filter(x, y, w, h):
    """Initialize a Kalman Filter for a new track"""
    kf = KalmanFilter(dim_x=8, dim_z=4)
    
    # State vector: [x, y, w, h, vx, vy, vw, vh]
    dt = 1  # Time step (1 frame)
    kf.F = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                     [0, 1, 0, 0, 0, dt, 0, 0],
                     [0, 0, 1, 0, 0, 0, dt, 0],
                     [0, 0, 0, 1, 0, 0, 0, dt],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1]])
    
    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0]])
    
    kf.P *= 1000
    kf.R = np.diag([5, 5, 3, 3])  # Measurement noise
    
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05)
    kf.Q = np.diag([q[0,0], q[0,0], q[0,0], q[0,0], q[1,1], q[1,1], q[1,1], q[1,1]])
    
    kf.x = np.array([x, y, w, h, 0, 0, 0, 0]).T
    
    return kf

def update_kalman_with_physics(kf):
    """Update Kalman Filter with physics constraints"""
    # Apply gravity to y-velocity
    kf.x[5] += GRAVITY
    
    # Dampen velocities for more realistic motion
    kf.x[4] *= 0.95  # x-velocity
    kf.x[6] *= 0.95  # width velocity
    kf.x[7] *= 0.95  # height velocity

def predict_future_positions(kf, steps=10):
    """Predict future positions using the Kalman Filter without copying"""
    future_positions = []
    
    # Save current state to restore later
    original_x = kf.x.copy()
    original_P = kf.P.copy()
    
    for step in range(steps):
        # Predict next state
        kf.predict()
        
        # Apply physics constraints
        update_kalman_with_physics(kf)
        
        # Get predicted position
        x, y, w, h = kf.x[0], kf.x[1], kf.x[2], kf.x[3]
        future_positions.append((x, y, w, h))
    
    # Restore original state to avoid affecting the main filter
    kf.x = original_x
    kf.P = original_P
    
    return future_positions

def play_alarm_sound():
    """Play alarm sound in a separate thread"""
    try:
        alarm_sound.play()
    except:
        print("Error playing alarm sound")

def draw_ui_panel(frame, rockfall_risk, landslide_risk, object_count):
    """Draw the UI panel with risk information"""
    UI_WIDTH = 280
    UI_HEIGHT = 240
    x0, y0 = 10, 10

    # Background panel
    cv2.rectangle(frame, (x0, y0), (x0 + UI_WIDTH, y0 + UI_HEIGHT), (30, 30, 30), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + UI_WIDTH, y0 + UI_HEIGHT), (100, 100, 100), 1)

    # Title
    cv2.putText(frame, "Rockfall Safety Monitor", (x0 + 10, y0 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Status
    any_alert = rockfall_risk > 70 or landslide_risk > 70
    status_color = (0, 0, 255) if any_alert else (0, 255, 0)
    cv2.putText(frame, f"Status: {'ALERT!' if any_alert else 'Normal'}",
                (x0 + 10, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

    # Alerts & Objects
    cv2.putText(frame, f"Alerts: {alert_count} | Objects: {object_count}",
                (x0 + 10, y0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Risk Bars
    def draw_risk_bar(img, label, risk, y_pos):
        cv2.putText(img, label, (x0 + 10, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(img, (x0 + 80, y_pos), (x0 + 250, y_pos + 12), (50, 50, 50), -1)
        bar_width = int(170 * (risk / 100))
        color = (0, 255, 0) if risk < 30 else (0, 165, 255) if risk < 70 else (0, 0, 255)
        cv2.rectangle(img, (x0 + 80, y_pos), (x0 + 80 + bar_width, y_pos + 12), color, -1)
        cv2.putText(img, f"{int(risk)}%", (x0 + 255, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    draw_risk_bar(frame, "Rockfall:", rockfall_risk, y0 + 90)
    draw_risk_bar(frame, "Landslide:", landslide_risk, y0 + 115)

    # Sensor Readouts with active mode indicators
    vib_color = (200, 200, 255) if not blast_mode else (0, 255, 255)
    rain_color = (200, 200, 255) if not rain_mode else (0, 255, 255)
    soil_color = (200, 200, 200) if not sandy_soil_mode else (0, 255, 255)
    temp_color = (200, 200, 200) if not freeze_thaw_mode else (0, 255, 255)
    
    cv2.putText(frame, f"Vib: {vibration:.1f} | Rain: {rain_level:.1f}",
                (x0 + 10, y0 + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vib_color, 1)
    cv2.putText(frame, f"Moist: {soil_moisture:.1f}% | Tilt: {ground_tilt:.2f}°",
                (x0 + 10, y0 + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    cv2.putText(frame, f"Soil: {soil_type} | Temp: {temperature:.0f}°C",
                (x0 + 10, y0 + 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, soil_color, 1)

    # Key hints with active mode indicators
    key_text = "Keys: D=Draw Zone"
    if blast_mode: key_text += " [BLAST]"
    if rain_mode: key_text += " [RAIN]"
    if freeze_thaw_mode: key_text += " [FREEZE]"
    if sandy_soil_mode: key_text += " [SAND]"
    key_text += " Q=Quit"
    
    cv2.putText(frame, key_text, (x0 + 10, y0 + 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

def mouse_callback(event, x, y, flags, param):
    """Mouse callback for drawing safety zones"""
    global current_zone_points, drawing_mode
    
    if drawing_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            current_zone_points.append((x, y))
            print(f"Point added: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN and current_zone_points:
            current_zone_points.pop()  # Remove last point on right click
            print("Last point removed")

# For real-time monitoring, use camera source
cap = cv2.VideoCapture(0)  # 0 for default camera

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Create window and set mouse callback
cv2.namedWindow('Rockfall Safety Monitor')
cv2.setMouseCallback('Rockfall Safety Monitor', mouse_callback)

print("Real-time rockfall monitoring started.")
print("Press 'd' to draw a safety zone, 'c' to clear all zones, 'q' to quit.")
print("Press 'b' for blast, 'r' for rain, 't' for freeze-thaw, 's' for sandy soil")
print("Press 'x' to clear safety zones, 'z' to reset alert counter")

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Failed to capture frame")
        break

    # Process key presses
    key = cv2.waitKey(1) & 0xFF
    
    # --- TOGGLE SIMULATION MODES ---
    if key == ord('b'):
        blast_mode = not blast_mode
        print(f"Blast mode {'ENABLED' if blast_mode else 'DISABLED'}")
    
    if key == ord('r'):
        rain_mode = not rain_mode
        print(f"Rain mode {'ENABLED' if rain_mode else 'DISABLED'}")
    
    if key == ord('t'):
        freeze_thaw_mode = not freeze_thaw_mode
        print(f"Freeze-thaw mode {'ENABLED' if freeze_thaw_mode else 'DISABLED'}")
    
    if key == ord('s'):
        sandy_soil_mode = not sandy_soil_mode
        soil_type = "Sand" if sandy_soil_mode else "Loam"
        print(f"Sandy soil mode {'ENABLED' if sandy_soil_mode else 'DISABLED'}")
    
    # Toggle drawing mode
    if key == ord('d'):
        drawing_mode = not drawing_mode
        if drawing_mode:
            print("Drawing mode enabled. Left-click to add points, right-click to remove.")
            current_zone_points = []
        else:
            if len(current_zone_points) > 2:
                safety_zones.append(current_zone_points.copy())
                print(f"Safety zone added with {len(current_zone_points)} points")
            current_zone_points = []
    
    # Clear all safety zones (one-time action)
    if key == ord('x'):
        safety_zones = []
        print("All safety zones cleared")
    
    # Clear all tracks (one-time action)
    if key == ord('c'):
        track_history.clear()
        kalman_filters.clear()
        print("All tracks cleared")
    
    # Reset alert counter (one-time action)
    if key == ord('z'):
        alert_count = 0
        print("Alert counter reset")

    # --- UPDATE PERSISTENT SENSORS BASED ON ACTIVE MODES ---
    # Vibration (blast mode)
    if blast_mode:
        vibration = min(20.0, vibration + 0.5)  # Increase vibration
    else:
        vibration = max(2.0, vibration - 0.3)   # Gradually decrease

    # Rain (rain mode)
    if rain_mode:
        rain_level = min(1.0, rain_level + 0.05)  # Increase rain
    else:
        rain_level = max(0.0, rain_level - 0.02)  # Gradually decrease

    # Temperature (freeze-thaw mode)
    if freeze_thaw_mode:
        temperature = max(-10.0, temperature - 0.2)  # Decrease temperature
    else:
        temperature = min(25.0, temperature + 0.1)   # Gradually increase

    # Soil moisture (affected by rain)
    if rain_level > 0.7:
        soil_moisture = min(70.0, soil_moisture + 0.8)
    else:
        soil_moisture = max(20.0, soil_moisture - 0.2)

    # Ground tilt (random slight variations)
    ground_tilt = max(0.1, min(1.0, ground_tilt + np.random.uniform(-0.02, 0.02)))

    # Run YOLOv8 tracking with ByteTrack
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5)
    annotated_frame = results[0].plot() if results else frame.copy()

    # Draw current safety zone being defined
    if drawing_mode:
        for i, point in enumerate(current_zone_points):
            cv2.circle(annotated_frame, point, 5, (0, 255, 255), -1)
            if i > 0:
                cv2.line(annotated_frame, current_zone_points[i-1], point, (0, 255, 255), 2)
        
        if len(current_zone_points) > 2:
            cv2.line(annotated_frame, current_zone_points[-1], current_zone_points[0], (0, 255, 255), 2)
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [np.array(current_zone_points)], (0, 255, 255))
            cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
        
        cv2.putText(annotated_frame, "Drawing Safety Zone", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Draw all defined safety zones
    for zone in safety_zones:
        if len(zone) > 2:
            zone_array = np.array(zone, np.int32)
            cv2.polylines(annotated_frame, [zone_array], True, (0, 255, 0), 2)
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [zone_array], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)

    # Process detections and predictions
    danger_detected = False
    current_time = time.time()
    
    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            if conf < 0.5:  # Confidence threshold
                continue
                
            x, y, w, h = box
            x, y, w, h = float(x), float(y), float(w), float(h)

            # Initialize Kalman Filter for new tracks
            if track_id not in kalman_filters:
                kalman_filters[track_id] = create_kalman_filter(x, y, w, h)
            else:
                # Update Kalman Filter with new measurement
                measurement = np.array([x, y, w, h])
                kalman_filters[track_id].update(measurement)
            
            # Predict next state
            kalman_filters[track_id].predict()
            update_kalman_with_physics(kalman_filters[track_id])
            
            # Get smoothed position from Kalman Filter
            kf_x, kf_y, kf_w, kf_h = kalman_filters[track_id].x[0:4]
            
            # Add to track history (using smoothed position)
            track_history[track_id].append((kf_x, kf_y))
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            # Predict future positions
            future_positions = predict_future_positions(
                kalman_filters[track_id], 
                PREDICTION_HORIZON
            )

            # Draw current detection (smoothed)
            cv2.rectangle(
                annotated_frame, 
                (int(kf_x - kf_w/2), int(kf_y - kf_h/2)),
                (int(kf_x + kf_w/2), int(kf_y + kf_h/2)),
                (0, 255, 0), 2  # Green for current position
            )

            # Draw track history
            tracks = track_history[track_id]
            for i in range(1, len(tracks)):
                if i >= len(tracks):
                    continue
                cv2.line(
                    annotated_frame, 
                    (int(tracks[i-1][0]), int(tracks[i-1][1])),
                    (int(tracks[i][0]), int(tracks[i][1])),
                    (0, 255, 255), 2  # Yellow for path
                )

            # Draw future predictions
            for i, (fx, fy, fw, fh) in enumerate(future_positions):
                # Fade from blue to red based on prediction distance
                color_intensity = int(255 * (1 - i/len(future_positions)))
                cv2.circle(
                    annotated_frame, 
                    (int(fx), int(fy)), 
                    3, 
                    (color_intensity, 0, 255 - color_intensity), 
                    -1
                )
                
                # Draw bounding box for the final prediction
                if i == len(future_positions) - 1:
                    cv2.rectangle(
                        annotated_frame, 
                        (int(fx - fw/2), int(fy - fh/2)),
                        (int(fx + fw/2), int(fy + fh/2)),
                        (0, 0, 255), 2  # Red for future prediction
                    )
                    
                # Check if predicted point is in any safety zone
                predicted_point = (int(fx), int(fy))
                for zone in safety_zones:
                    if len(zone) > 2 and cv2.pointPolygonTest(np.array(zone, np.int32), predicted_point, False) >= 0:
                        danger_detected = True
                        cv2.putText(annotated_frame, "SAFETY ZONE BREACH!", (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Check current position against safety zones
            current_point = (int(kf_x), int(kf_y))
            for zone in safety_zones:
                if len(zone) > 2 and cv2.pointPolygonTest(np.array(zone, np.int32), current_point, False) >= 0:
                    danger_detected = True

            # Add ID and confidence label
            label = f"ID: {track_id}, Conf: {conf:.2f}"
            cv2.putText(
                annotated_frame, 
                label, 
                (int(x - w/2), int(y - h/2 - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )

    # Compute risks
    rockfall_risk = 0
    if danger_detected: 
        rockfall_risk += 80
    if vibration > 10: 
        rockfall_risk += 15
    if rain_level > 0.5: 
        rockfall_risk += 15
    if len(track_history) > 5: 
        rockfall_risk += 10
    rockfall_risk = max(0, min(100, rockfall_risk))

    # Landslide Risk
    landslide_risk = 0
    soil_cohesion = 30 if sandy_soil_mode else 70 if soil_type == "Clay" else 50
    moisture_multiplier = 1.5 if sandy_soil_mode else 0.8 if soil_type == "Clay" else 1.0
    effective_moisture = soil_moisture * moisture_multiplier
    
    if effective_moisture > 40:     landslide_risk += 30
    if ground_tilt > 0.5:           landslide_risk += 25
    if rain_level > 0.7:            landslide_risk += 20
    if soil_cohesion < 50:          landslide_risk += 15
    if temperature < 0:             landslide_risk += 10
    landslide_risk = min(100, landslide_risk)

    # Debug output when risk changes
    if (round(landslide_risk) != round(last_landslide_risk) or 
        round(rockfall_risk) != round(last_rockfall_risk)):

        print(f"📊 RISK UPDATE | Rockfall: {rockfall_risk:.0f}% | Landslide: {landslide_risk:.0f}%")
        if landslide_risk > 0:
            print(f"   → Moist: {effective_moisture:.1f}% | Tilt: {ground_tilt:.2f}° | Temp: {temperature:.0f}°C | Soil: {soil_type}")
        print("-" * 50)

        last_landslide_risk = landslide_risk
        last_rockfall_risk = rockfall_risk

    # Draw UI panel
    draw_ui_panel(annotated_frame, rockfall_risk, landslide_risk, len(track_history))

    # On-screen warnings
    if danger_detected:
        cv2.putText(annotated_frame, "SAFETY ZONE BREACH!", (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if landslide_risk > 70:
        cv2.putText(annotated_frame, "LANDSLIDE WARNING!", (50, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show active simulation modes
    mode_y = 170
    if blast_mode:
        cv2.putText(annotated_frame, "BLAST SIMULATION ACTIVE", (50, mode_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        mode_y += 30
    if rain_mode:
        cv2.putText(annotated_frame, "RAIN SIMULATION ACTIVE", (50, mode_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        mode_y += 30
    if freeze_thaw_mode:
        cv2.putText(annotated_frame, "FREEZE-THAW SIMULATION ACTIVE", (50, mode_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        mode_y += 30
    if sandy_soil_mode:
        cv2.putText(annotated_frame, "SANDY SOIL SIMULATION ACTIVE", (50, mode_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Trigger alarm
    any_alert = rockfall_risk > 70 or landslide_risk > 70
    if any_alert:
        if not alarm_active or (current_time - alarm_start_time > alert_cooldown):
            alarm_active = True
            alarm_start_time = current_time
            alert_count += 1
            print(f"🚨 DANGER! Rockfall: {rockfall_risk:.1f}% | Landslide: {landslide_risk:.1f}%")
            threading.Thread(target=play_alarm_sound).start()
    else:
        alarm_active = False

    # Display the frame
    cv2.imshow('Rockfall Safety Monitor', annotated_frame)
    
    # Quit on 'q'
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Monitoring stopped.")