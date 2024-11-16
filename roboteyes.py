import cv2
from cvzone.FaceDetectionModule import FaceDetector
import time
from gpiozero import AngularServo

def map_value(value, in_min, in_max, out_min, out_max):
    """Map a value from one range to another."""
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def display_image_with_text(img, text):
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Image", img)

# Initialize camera and face detector
cap = cv2.VideoCapture(0)  # Use the correct index if your camera is different

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

detector = FaceDetector(minDetectionCon=0.5, modelSelection=1)

# Initialize servos
servo_x = AngularServo(5, min_angle=0, max_angle=180)  # Pin 5 for X-axis
servo_y = AngularServo(6, min_angle=0, max_angle=180)  # Pin 6 for Y-axis
servo_boundary = AngularServo(13, min_angle=0, max_angle=180)  # Pin 13 for boundary servo

# Initialize variables
locked_face_id = None
last_detected_time = time.time()
timeout_duration = 2  # Timeout in seconds

# Screen resolution (assuming 640x480)
frame_width = 640
frame_height = 480
boundary_padding = 40  # Padding zone in pixels

while True:
    success, img = cap.read()

    # Check if the frame was successfully captured
    if not success or img is None:
        print("Warning: Failed to capture image. Retrying...")
        time.sleep(0.1)  # Wait briefly before retrying
        continue

    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        # Attempt to find the locked face
        locked_face = next((bbox for bbox in bboxs if bbox["id"] == locked_face_id), None)

        if locked_face:
            last_detected_time = time.time()
            center = locked_face["center"]
            x, y = center

            # Map x and y coordinates to servo angles
            servo_x_angle = map_value(x, 0, frame_width, 0, 180)
            servo_y_angle = map_value(y, 0, frame_height, 0, 180)

            # Move the x and y servos
            servo_x.angle = servo_x_angle
            servo_y.angle = servo_y_angle

            # Control the boundary servo
            if x < boundary_padding:  # Left boundary
                boundary_angle = map_value(x, 0, boundary_padding, 0, 90)
                servo_boundary.angle = boundary_angle
            elif x > frame_width - boundary_padding:  # Right boundary
                boundary_angle = map_value(x, frame_width - boundary_padding, frame_width, 90, 180)
                servo_boundary.angle = boundary_angle
            else:  # Neutral zone
                servo_boundary.angle = None  # Keep it still

            # Display text
            display_text = f"X: {x} -> Angle: {servo_x_angle}, Y: {y} -> Angle: {servo_y_angle}, Boundary Servo: {servo_boundary.angle}"
            print(display_text)

            cv2.circle(img, center, 2, (255, 0, 255), cv2.FILLED)
            cv2.putText(img, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        else:
            # Switch to a new face if the timeout has passed
            if time.time() - last_detected_time > timeout_duration:
                locked_face_id = bboxs[0]["id"]  # Lock onto a new face
                last_detected_time = time.time()

    display_image_with_text(img, "")

    key = cv2.waitKey(1)
    if key == 13:  # Enter key to exit
        break

cap.release()
cv2.destroyAllWindows()
