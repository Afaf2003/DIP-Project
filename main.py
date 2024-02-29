import cv2

# Create tracker object
tracker = cv2.TrackerCSRT_create()

# Load video
video = cv2.VideoCapture('los_angeles.mp4')

# Read first frame
success, frame = video.read()

# Select ROI (Region of Interest)
bbox = cv2.selectROI('Tracking', frame, False)
tracker.init(frame, bbox)

while True:
    # Read a new frame
    success, frame = video.read()
    if not success:
        break

    # Update tracker
    success, bbox = tracker.update(frame)

    # Draw bounding box
    if success:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display result
    cv2.imshow('Tracking', frame)

    # Exit if ESC pressed
    if cv2.waitKey(1) == 27:
        break

# Release resources
video.release()
cv2.destroyAllWindows()
