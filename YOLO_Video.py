from ultralytics import YOLO
import cv2
import math

# Initialize the YOLO model outside this function if you're calling it multiple times.
# For the sake of this example, I'm including it here.
model = YOLO("../YOLO-Weights/most_updated.pt")  # Load the model once, outside your video processing loop
classNames = ["drowning", "person-out-of-water", "swimming"]


def video_detection(path_x):
    cap = cv2.VideoCapture(path_x)

    while True:
        success, img = cap.read()
        if not success:
            break  # Exit the loop if the video ends or there is a read failure

        # Optional: Resize image for faster processing, e.g., img = cv2.resize(img, (640, 360))

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
                conf = round(box.conf[0].item(), 2)  # Confidence score
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf}'

                # Determine color based on class
                color = (0, 255, 255)  # Default to Yellow
                if class_name == "drowning":
                    color = (0, 0, 255)  # Red
                elif class_name == "person-out-of-water":
                    color = (0, 255, 0)  # Green

                if conf > 0.5:  # Draw only for high confidence detections
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1)

        yield img  # Use yield to return the processed frame

    cap.release()
    cv2.destroyAllWindows()

