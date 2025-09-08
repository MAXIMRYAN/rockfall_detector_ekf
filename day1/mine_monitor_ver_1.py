# main.py - Fixed Mine Rockfall Monitor
import cv2
import numpy as np
from ultralytics import YOLO
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from threading import Thread, Lock
import json
import os
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import asyncio

from pyngrok import ngrok

# Set up ngrok tunnel
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")
# -------------------------------
# CONFIGURATION
# -------------------------------
# Default danger zone (can be configured via API)
DEFAULT_POINTS = [
    (100, 100),
    (300, 100),
    (300, 300),
    (100, 300)
]

# Configuration file path
CONFIG_FILE = "config.json"

# Shared globals with thread safety
annotated_frame = None
danger_detected = False
track_history = {}
points = DEFAULT_POINTS.copy()
frame_lock = Lock()
config_lock = Lock()

# -------------------------------
# DATA MODELS
# -------------------------------
class Point(BaseModel):
    x: int
    y: int

class ConfigUpdate(BaseModel):
    points: List[Point]

# -------------------------------
# CONFIGURATION MANAGEMENT
# -------------------------------
def load_config():
    """Load configuration from file if it exists"""
    global points
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                if 'points' in config:
                    with config_lock:
                        points = [(pt[0], pt[1]) for pt in config['points']]
                    print(f"✅ Loaded configuration from {CONFIG_FILE}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")

def save_config():
    """Save current configuration to file"""
    try:
        with config_lock:
            config = {'points': points}
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        print(f"✅ Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"❌ Error saving config: {e}")

# Load configuration at startup
load_config()

# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI(title="Mine Rockfall Monitor", version="2.0")

@app.get("/")
def index():
    return {
        "message": "Mine Rockfall Monitor - Enhanced Version",
        "endpoints": [
            "/video - Live video feed with detection",
            "/status - Current system status",
            "/config - Get current configuration",
            "/config/update - Update configuration (POST)",
            "/config/reset - Reset to default configuration"
        ]
    }

@app.get("/video")
async def video_feed():
    """Video streaming route that returns a multipart response"""
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/status")
def get_status():
    with frame_lock:
        status = danger_detected
        obj_count = len(track_history)
    
    return {
        "danger_detected": bool(status),
        "object_count": obj_count,
        "timestamp": datetime.now().isoformat(),
        "system_time": time.time()
    }

@app.get("/config")
def get_config():
    with config_lock:
        return {
            "points": points,
            "danger_zone_configured": len(points) >= 3
        }

@app.post("/config/update")
def update_config(config_update: ConfigUpdate):
    """Update the danger zone configuration"""
    new_points = [(pt.x, pt.y) for pt in config_update.points]
    
    if len(new_points) < 3:
        raise HTTPException(status_code=400, detail="At least 3 points required to define a polygon")
    
    # Update configuration
    with config_lock:
        points.clear()
        points.extend(new_points)
    
    save_config()
    
    return {
        "message": "Configuration updated successfully",
        "new_points": points
    }

@app.post("/config/reset")
def reset_config():
    """Reset to default configuration"""
    with config_lock:
        points.clear()
        points.extend(DEFAULT_POINTS)
    
    save_config()
    
    return {
        "message": "Configuration reset to defaults",
        "points": points
    }

# -------------------------------
# VIDEO STREAM GENERATOR
# -------------------------------
async def generate_video_stream():
    while True:
        try:
            # Get the current frame
            with frame_lock:
                current_frame = annotated_frame
            
            if current_frame is not None:
                # Encode frame as JPEG
                success, buffer = cv2.imencode('.jpg', current_frame)
                if success:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Send error frame
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "Encoding error", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    success, buffer = cv2.imencode('.jpg', blank)
                    if success:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Send waiting frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for detection...", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                success, buffer = cv2.imencode('.jpg', blank)
                if success:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            await asyncio.sleep(0.05)  # ~20 FPS
        except Exception as e:
            print(f"❌ Stream generator error: {e}")
            # Send error frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"Stream error: {e}", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            success, buffer = cv2.imencode('.jpg', blank)
            if success:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(1)

# -------------------------------
# DETECTION LOOP (Runs in Background)
# -------------------------------
def run_detection():
    global annotated_frame, danger_detected, track_history
    
    # Load model
    try:
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        with frame_lock:
            annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(annotated_frame, "Model Load Error", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        with frame_lock:
            annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(annotated_frame, "Webcam Error", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Detection started")

    # Initialize FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            with frame_lock:
                annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(annotated_frame, "Frame Read Failed", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            time.sleep(1)
            continue

        try:
            # Calculate FPS
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_frame_count
                fps_frame_count = 0
                fps_start_time = time.time()

            # Get current configuration
            with config_lock:
                current_points = points.copy()

            # Detect objects
            results = model.predict(frame, conf=0.5, verbose=False)
            annotated = results[0].plot()

            # Reset danger flag
            current_danger = False

            # Process detection boxes
            boxes = results[0].boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Initialize tracking for new objects
                    if i not in track_history:
                        track_history[i] = []
                    
                    # Update tracking history
                    track_history[i].append((cx, cy))
                    if len(track_history[i]) > 20:  # Keep last 20 positions
                        track_history[i].pop(0)

                    # Draw trajectory
                    if len(track_history[i]) > 1:
                        for j in range(1, len(track_history[i])):
                            cv2.line(annotated, track_history[i][j-1], track_history[i][j], (255, 0, 0), 2)

                    # Predict future position
                    if len(track_history[i]) > 2:
                        pts = track_history[i][-3:]
                        dx = pts[-1][0] - pts[0][0]
                        dy = pts[-1][1] - pts[0][1]
                        speed = np.sqrt(dx**2 + dy**2)
                        
                        # More sophisticated prediction
                        pred_x = pts[-1][0] + int(dx * min(2.0, 1.0 + speed / 30))
                        pred_y = pts[-1][1] + int(dy * min(2.0, 1.0 + speed / 30))
                        
                        # Draw prediction point
                        cv2.circle(annotated, (pred_x, pred_y), 8, (0, 255, 255), -1)
                        cv2.putText(annotated, f"Pred", (pred_x+10, pred_y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        # Check if prediction is in danger zone
                        if len(current_points) > 2:
                            poly = np.array(current_points, np.int32)
                            if cv2.pointPolygonTest(poly, (pred_x, pred_y), False) >= 0:
                                current_danger = True
                                cv2.putText(annotated, "DANGER PREDICTED!", (pred_x+10, pred_y+10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Check current position in danger zone
                    if len(current_points) > 2:
                        poly = np.array(current_points, np.int32)
                        if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                            current_danger = True
                            cv2.putText(annotated, "IN DANGER ZONE!", (cx+10, cy+10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw danger zone
            if len(current_points) > 2:
                poly = np.array(current_points, np.int32)
                cv2.polylines(annotated, [poly], True, (0, 0, 255), 2)
                
                # Create semi-transparent overlay
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [poly], (0, 0, 255))
                cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
                
                # Label the danger zone
                centroid = np.mean(poly, axis=0).astype(int)
                cv2.putText(annotated, "DANGER ZONE", (centroid[0]-80, centroid[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Update global danger status
            with frame_lock:
                danger_detected = current_danger

            # Add status information to frame
            status_color = (0, 0, 255) if current_danger else (0, 255, 0)
            status_text = "ALERT: ROCKFALL DETECTED!" if current_danger else "Status: Normal"
            
            cv2.putText(annotated, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Add FPS and object count
            cv2.putText(annotated, f"FPS: {fps}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, f"Objects: {len(track_history)}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated, timestamp, (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Update global frame
            with frame_lock:
                annotated_frame = annotated

        except Exception as e:
            print(f"❌ Detection loop error: {e}")
            # Create error frame
            error_frame = frame.copy() if 'frame' in locals() else np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Processing Error: {str(e)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            with frame_lock:
                annotated_frame = error_frame

        # Control processing rate
        time.sleep(0.05)  # ~20 FPS

    cap.release()

# -------------------------------
# START BACKGROUND THREAD
# -------------------------------
detection_thread = Thread(target=run_detection, daemon=True)
detection_thread.start()

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    print("\n🚀 Starting Mine Rockfall Monitor Server")
    print("📍 Available at: http://localhost:8000")
    print("👉 Endpoints:")
    print("   - http://localhost:8000/video (Live video feed)")
    print("   - http://localhost:8000/status (System status)")
    print("   - http://localhost:8000/config (Configuration)")
    print("\n📝 Use POST requests to /config/update to change the danger zone")
    print("   Example: {\"points\": [{\"x\": 100, \"y\": 100}, {\"x\": 300, \"y\": 100}, {\"x\": 300, \"y\": 300}, {\"x\": 100, \"y\": 300}]}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)