# mine_monitor.py - With System Status & Configuration
import cv2
import numpy as np
from ultralytics import YOLO
import time
import psutil  # For system monitoring
from fastapi import FastAPI
from starlette.responses import StreamingResponse
import uvicorn
from threading import Thread
import os

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

# Shared globals
annotated_frame = None
danger_detected = False
track_history = {}
model_loaded = False
webcam_active = False

# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI()

@app.get("/")
def index():
    return {
        "message": "Mine Rockfall Monitor API",
        "endpoints": [
            "/video - MJPEG stream",
            "/status - Detection & system status",
            "/config - Current configuration",
            "/system - Detailed system health"
        ]
    }

# -------------------------------
# VIDEO STREAM
# -------------------------------
def generate_video_frames():
    global annotated_frame
    while config["stream_active"]:
        if annotated_frame is not None:
            try:
                success, buffer = cv2.imencode('.jpg', annotated_frame)
                if success:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')
            except:
                pass
        else:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No Frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            success, buffer = cv2.imencode('.jpg', blank)
            if success:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
        time.sleep(0.1)

@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# -------------------------------
# STATUS ENDPOINT
# -------------------------------
@app.get("/status")
def get_status():
    return {
        "detection": {
            "danger_detected": bool(danger_detected),
            "object_count": len(track_history),
            "track_history_size": sum(len(track) for track in track_history.values()),
            "model_loaded": model_loaded,
            "webcam_active": webcam_active
        },
        "system": {
            "timestamp": time.time(),
            "uptime": time.time() - app.start_time,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if os.path.exists('/') else 0
        }
    }

# -------------------------------
# CONFIGURATION ENDPOINT
# -------------------------------
@app.get("/config")
def get_config():
    return {
        "config": config,
        "message": "Use PUT /config to update (not implemented in this version)"
    }

# -------------------------------
# DETAILED SYSTEM HEALTH
# -------------------------------
@app.get("/system")
def system_health():
    process = psutil.Process(os.getpid())
    return {
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total_mb": int(psutil.virtual_memory().total / 1_000_000),
                "available_mb": int(psutil.virtual_memory().available / 1_000_000),
                "used_percent": psutil.virtual_memory().percent
            },
            "disk": {
                "root_usage_percent": psutil.disk_usage('/').percent if os.path.exists('/') else "N/A"
            },
            "process": {
                "memory_mb": int(process.memory_info().rss / 1_000_000),
                "cpu_percent": process.cpu_percent(interval=1)
            },
            "environment": {
                "platform": os.name,
                "python_version": os.sys.version.split()[0],
                "working_directory": os.getcwd()
            },
            "app": {
                "uptime_seconds": time.time() - app.start_time,
                "model_loaded": model_loaded,
                "webcam_active": webcam_active,
                "detected_objects": len(track_history),
                "alerts_enabled": config["alert_enabled"]
            }
        }
    }

# -------------------------------
# DETECTION LOOP
# -------------------------------
def run_detection():
    global annotated_frame, danger_detected, track_history, model_loaded, webcam_active

    # Load model
    try:
        model = YOLO(config["model_type"])
        model_loaded = True
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        model_loaded = False
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not available")
        webcam_active = False
        annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(annotated_frame, "Webcam Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return
    webcam_active = True

    print("✅ Detection started")

    while True:
        ret, frame = cap.read()
        if not ret:
            webcam_active = False
            annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(annotated_frame, "Frame Read Failed", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            time.sleep(1)
            continue
        else:
            webcam_active = True

        try:
            results = model.predict(frame, conf=config["confidence_threshold"], verbose=False)
            annotated = results[0].plot()
            danger_detected = False

            if results[0].boxes is not None:
                for i, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Track
                    if i not in track_history:
                        track_history[i] = []
                    track_history[i].append((cx, cy))
                    if len(track_history[i]) > 10:
                        track_history[i].pop(0)

                    # Draw trajectory
                    if len(track_history[i]) > 1:
                        for j in range(1, len(track_history[i])):
                            cv2.line(annotated, track_history[i][j-1], track_history[i][j], (255, 0, 0), 2)

                    # Predict
                    if len(track_history[i]) > 2:
                        pts = track_history[i][-3:]
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

# Start detection in background
detection_thread = Thread(target=run_detection, daemon=True)
detection_thread.start()

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    print("\n🚀 Starting Mine Safety Monitor")
    print("🌐 Endpoints:")
    print("   - http://localhost:8000/video     (Live stream)")
    print("   - http://localhost:8000/status    (Detection + system)")
    print("   - http://localhost:8000/config    (Current config)")
    print("   - http://localhost:8000/system    (Detailed health)")
    uvicorn.run(app, host="0.0.0.0", port=8000)