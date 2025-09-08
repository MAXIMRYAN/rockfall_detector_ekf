# webcam_test.py
import cv2  # This is the OpenCV library you just installed

# Initialize the webcam. '0' means the first (default) camera.
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam. Is it being used by another program?")
    exit()

print("Webcam is live! Press 'q' to quit.")

# This loop runs forever, reading frames from the webcam
while True:
    # Read a frame from the webcam
    # 'ret' is True if the frame was read correctly, False otherwise.
    # 'frame' is the image data.
    ret, frame = cap.read()

    # If the frame wasn't read correctly, break the loop.
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # Display the frame in a window named 'Rockfall AI Feed'
    cv2.imshow('Rockfall AI Feed', frame)

    # Wait for 1 millisecond, and check if the pressed key was 'q'
    if cv2.waitKey(1) == ord('q'):
        print("Quitting...")
        break

# When everything is done, release the camera and close the window
cap.release()
cv2.destroyAllWindows()