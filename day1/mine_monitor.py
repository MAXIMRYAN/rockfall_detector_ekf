# main.py - Mine Safety Monitor with Full Dashboard API
import cv2
import numpy as np
from ultralytics import YOLO
import time
import psutil
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from threading import Thread

# -------------------------------
# CONFIGURATION
# -------------------------------
config = {
    "danger_zone": [
        {"x": 100, "y": 100},
        {"x": 300, "y": 100},
        {"x": 300, "y": 300},
        {"x": 100, "y": 300}
    ],
    "confidence_threshold": 0.5,
    "prediction_sensitivity": 1.5,
    "alert_enabled": True,
    "model_type": "yolov8n.pt",
    "stream_active": True
}

# Persistent sensor states
soil_moisture = 20.0      # %
rain_level = 0.0          # 0.0 to 1.0
ground_tilt = 0.2         # degrees
temperature = 25.0        # °C
soil_type = "Loam"        # "Sand", "Clay", "Loam"

# Shared globals
annotated_frame = None
danger_detected = False
track_history = {}
alert_count = 0
last_alert_time = None
model_loaded = False
webcam_active = False

# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI()

@app.get("/")
def index():
    return {
        "message": "Mine Safety Monitor API",
        "endpoints": [
            "/video - Live MJPEG stream",
            "/status - Basic status",
            "/dashboard - Full dashboard data",
            "/system - System health",
            "/config - Configuration"
        ]
    }

# -------------------------------
# VIDEO STREAM
# -------------------------------
def generate_video_stream():
    global annotated_frame
    while config["stream_active"]:
        if annotated_frame is not None:
            try:
                success, buffer = cv2.imencode('.jpg', annotated_frame)
                if success:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Frame encoding error: {e}")
        else:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Loading...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
        time.sleep(0.1)

@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# -------------------------------
# BASIC STATUS
# -------------------------------
@app.get("/status")
def get_status():
    return {
        "danger_detected": danger_detected,
        "object_count": len(track_history),
        "alert_count": alert_count,
        "timestamp": time.time()
    }

# -------------------------------
# DASHBOARD DATA (For frontend)
# -------------------------------
@app.get("/dashboard")
def dashboard_data():
    # Compute rockfall risk
    rockfall_risk = 50 if danger_detected else 0
    rockfall_risk += len(track_history) * 5
    rockfall_risk = min(100, rockfall_risk)

    # Compute landslide risk
    landslide_risk = 0
    if soil_moisture > 40: landslide_risk += 30
    if ground_tilt > 0.5: landslide_risk += 25
    if rain_level > 0.7: landslide_risk += 20
    if soil_type == "Sand": landslide_risk += 15
    if temperature < 0: landslide_risk += 10
    landslide_risk = min(100, landslide_risk)

    return {
        "status": "ALERT" if (danger_detected or landslide_risk > 70) else "NORMAL",
        "metrics": {
            "object_count": len(track_history),
            "danger_detected": danger_detected,
            "alert_count": alert_count,
            "last_alert_time": last_alert_time
        },
        "risks": {
            "rockfall_risk": rockfall_risk,
            "landslide_risk": landslide_risk
        },
        "environment": {
            "soil_moisture": round(soil_moisture, 1),
            "rain_level": round(rain_level, 2),
            "ground_tilt": round(ground_tilt, 2),
            "temperature": round(temperature, 1),
            "soil_type": soil_type
        },
        "system": {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if os.path.exists('/') else 0,
            "process_memory_mb": int(psutil.Process().memory_info().rss / 1_000_000),
            "uptime": time.time() - app.start_time,
            "webcam_status": "connected" if webcam_active else "disconnected",
            "model_status": "loaded" if model_loaded else "error"
        },
        "config": config
    }

# -------------------------------
# SYSTEM HEALTH
# -------------------------------
@app.get("/system")
def system_health():
    return {
        "cpu": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_mb": int(psutil.virtual_memory().used / 1_000_000),
        "disk_percent": psutil.disk_usage('/').percent if os.path.exists('/') else 0,
        "uptime": time.time() - app.start_time,
        "timestamp": time.time()
    }

# -------------------------------
# CONFIG
# -------------------------------
@app.get("/config")
def get_config():
    return {"config": config}

# -------------------------------
# DETECTION LOOP
# -------------------------------
def run_detection():
    global annotated_frame, danger_detected, track_history
    global alert_count, last_alert_time, model_loaded, webcam_active
    global soil_moisture, rain_level, ground_tilt, temperature, soil_type

    # Simulated sensor states
    vibration = 2.0

    try:
        model = YOLO(config["model_type"])
        model_loaded = True
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        model_loaded = False
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not available")
        webcam_active = False
        return
    webcam_active = True

    print("✅ Detection started")

    while True:
        # Simulate sensor updates
        if np.random.random() < 0.02:  # Random rain trigger
            rain_level = 0.9 if rain_level < 0.1 else 0.0

        if rain_level > 0.7:
            soil_moisture = min(70.0, soil_moisture + 0.8)
        else:
            soil_moisture = max(20.0, soil_moisture - 0.2)

        # Read frame
        ret, frame = cap.read()
        if not ret:
            webcam_active = False
            time.sleep(1)
            continue
        else:
            webcam_active = True

        try:
            results = model.predict(frame, conf=config["confidence_threshold"], verbose=False)
            annotated = results[0].plot()
            danger_detected = False

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Track object
                    obj_id = hash(str(box)) % 1000
                    if obj_id not in track_history:
                        track_history[obj_id] = []
                    track_history[obj_id].append((cx, cy))
                    if len(track_history[obj_id]) > 10:
                        track_history[obj_id].pop(0)

                    # Draw trajectory
                    if len(track_history[obj_id]) > 1:
                        for j in range(1, len(track_history[obj_id])):
                            cv2.line(annotated, track_history[obj_id][j-1], track_history[obj_id][j], (255, 0, 0), 2)

                    # Predict path
                    if len(track_history[obj_id]) > 2:
                        pts = track_history[obj_id][-3:]
                        dx = pts[-1][0] - pts[0][0]
                        dy = pts[-1][1] - pts[0][1]
                        pred_x = pts[-1][0] + int(dx * config["prediction_sensitivity"])
                        pred_y = pts[-1][1] + int(dy * config["prediction_sensitivity"])
                        cv2.circle(annotated, (pred_x, pred_y), 8, (0, 255, 255), -1)

                        # Check danger zone
                        poly = np.array([(p["x"], p["y"]) for p in config["danger_zone"]], np.int32)
                        if len(config["danger_zone"]) > 2:
                            if cv2.pointPolygonTest(poly, (pred_x, pred_y), False) >= 0:
                                danger_detected = True
                            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                                danger_detected = True

            # Draw danger zone
            if len(config["danger_zone"]) > 2:
                poly = np.array([(p["x"], p["y"]) for p in config["danger_zone"]], np.int32)
                cv2.polylines(annotated, [poly], True, (0, 0, 255), 2)
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [poly], (0, 0, 255))
                cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)

            # Status text
            if danger_detected:
                cv2.putText(annotated, "ALERT: ROCKFALL DETECTED!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if last_alert_time is None or time.time() - last_alert_time > 2:
                    alert_count += 1
                    last_alert_time = time.time()

            cv2.putText(annotated, f"Status: {'ALERT!' if danger_detected else 'Normal'}",
                       (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255) if danger_detected else (0, 255, 0), 2)

            annotated_frame = annotated

        except Exception as e:
            print(f"❌ Detection error: {e}")

        time.sleep(0.1)

    cap.release()

# -------------------------------
# STARTUP
# -------------------------------
app.start_time = time.time()

detection_thread = Thread(target=run_detection, daemon=True)
detection_thread.start()

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    print("\n🚀 Mine Safety Monitor Started")
    print("🌐 Dashboard: http://localhost:8000/dashboard")
    print("🎥 Video: http://localhost:8000/video")
    print("📊 System: http://localhost:8000/system")
    uvicorn.run(app, host="0.0.0.0", port=8000)