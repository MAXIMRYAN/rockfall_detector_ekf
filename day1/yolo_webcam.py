# yolo_webcam_demo.py
from ultralytics import YOLO
import cv2

# Load the COCO pre-trained model
model = YOLO('yolov8n.pt') 

# Initialize the webcam
cap = cv2.VideoCapture(0)

print("Running YOLO Live Detection on GPU. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference. 'stream=True' returns a generator.
    results = model.predict(frame, stream=True, verbose=False)
    
    # Loop through the generator (usually just one result)
    for result in results:
        # Draw boxes on the frame
        annotated_frame = result.plot()
        # Display the frame
        cv2.imshow('YOLO Live Detection - Rockfall AI', annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()