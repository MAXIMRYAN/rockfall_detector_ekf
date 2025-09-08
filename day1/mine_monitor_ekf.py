from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# Load model
model = YOLO('yolov8n.pt')

# Dictionary to store track history and Kalman filters
track_history = defaultdict(lambda: [])
kalman_filters = {}

# Physics parameters
GRAVITY = 0.5  # pixels/frame² (needs calibration)
PREDICTION_HORIZON = 15  # frames to predict ahead

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

print("Real-time rockfall monitoring started. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Failed to capture frame")
        break

    # Run YOLOv8 tracking with ByteTrack
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5)

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
                frame, 
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
                    frame, 
                    (int(tracks[i-1][0]), int(tracks[i-1][1])),
                    (int(tracks[i][0]), int(tracks[i][1])),
                    (0, 255, 255), 2  # Yellow for path
                )

            # Draw future predictions
            for i, (fx, fy, fw, fh) in enumerate(future_positions):
                # Fade from blue to red based on prediction distance
                color_intensity = int(255 * (1 - i/len(future_positions)))
                cv2.circle(
                    frame, 
                    (int(fx), int(fy)), 
                    3, 
                    (color_intensity, 0, 255 - color_intensity), 
                    -1
                )
                
                # Draw bounding box for the final prediction
                if i == len(future_positions) - 1:
                    cv2.rectangle(
                        frame, 
                        (int(fx - fw/2), int(fy - fh/2)),
                        (int(fx + fw/2), int(fy + fh/2)),
                        (0, 0, 255), 2  # Red for future prediction
                    )

            # Add ID and confidence label
            label = f"ID: {track_id}, Conf: {conf:.2f}"
            cv2.putText(
                frame, 
                label, 
                (int(x - w/2), int(y - h/2 - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )

    # Display the frame
    cv2.imshow('Real-time Rockfall Monitoring', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Monitoring stopped.")