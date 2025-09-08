# motion_detector.py
import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a background subtractor object.
# This object will learn what the static background looks like.
# Any new object that appears and moves is considered "foreground."
# history=500: It remembers the last 500 frames to build its background model.
# varThreshold=16: How sensitive it is to change. Lower = more sensitive.
# detectShadows=True: It will detect shadows but mark them as a different color (good for ignoring them).
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

print("Simple Motion Detector is active! Move around.")
print("Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the background subtractor to the current frame.
    # This creates a 'mask' – a black and white image.
    # White pixels = moving objects (foreground).
    # Black pixels = static background.
    # Gray pixels = (if shadows are detected) usually ignored.
    fgmask = fgbg.apply(frame)

    # (Optional but useful) Apply some image processing to clean up the mask.
    # This removes small noise (like tiny white dots).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Now, find the contours (outlines) of the white blobs in the mask.
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour we found.
    for contour in contours:
        # Ignore very small contours (likely noise)
        if cv2.contourArea(contour) < 1000:  # Adjust this number to change sensitivity
            continue

        # Get the coordinates of a bounding box around the contour.
        x, y, w, h = cv2.boundingRect(contour)

        # Draw a green rectangle around the moving object on the original frame.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add a label
        cv2.putText(frame, 'Motion Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the original frame with green boxes drawn on it.
    cv2.imshow('Rockfall Detector (Motion)', frame)
    # Display the black-and-white mask (useful for debugging).
    cv2.imshow('Foreground Mask', fgmask)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release everything and close windows
cap.release()
cv2.destroyAllWindows()