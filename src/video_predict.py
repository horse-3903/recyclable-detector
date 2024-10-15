import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
print("Loading Model...", end="")
model = YOLO("./src/model.pt")
print("Done")

# Open the webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam")
        break

    # Run the YOLO model on the frame
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # Draw the bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Display the frame with detections
    cv2.imshow('YOLOv8 Live Detection', annotated_frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()