import cv2
import torch
from experimental import attempt_load
from general import non_max_suppression, scale_boxes
from plots import Annotator
from torch_utils import select_device
import time  # For cooldown tracking
import pyttsx3  # For text-to-speech
import threading  # For running TTS in a separate thread
import queue  # For managing TTS requests

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Queue to hold TTS requests
tts_queue = queue.Queue()

# Function to handle TTS in a dedicated thread
def tts_worker():
    while True:
        # Get the next TTS request from the queue
        sign_name = tts_queue.get()
        if sign_name is None:  # Sentinel value to stop the thread
            break
        # Announce the detected sign using text-to-speech
        engine.say(f"Detected {sign_name}")
        engine.runAndWait()
        # Mark the task as done
        tts_queue.task_done()

# Start the TTS worker thread
tts_thread = threading.Thread(target=tts_worker)
tts_thread.start()

# Load the model
weights = "modelpath"  # Path to the pre-trained weights
device = select_device("")  # Use GPU if available, otherwise CPU
model = attempt_load(weights, device)  # Load the model
stride = int(model.stride.max())  # Model stride
names = model.module.names if hasattr(model, "module") else model.names  # Class names

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the input size for the model
imgsz = 640  # YOLOv5 input size

# Create a resizable window
cv2.namedWindow("Traffic Sign Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Sign Detection", 800, 600)

# Dictionary to store the last detection time for each sign
last_detection_time = {}

# Cooldown period in seconds
cooldown = 20

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess the frame
    img = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)  # Resize with aspect ratio
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, HWC to CHW, and make a copy
    img = torch.from_numpy(img).to(device)  # Convert to tensor
    img = img.float() / 255.0  # Normalize to [0, 1]
    img = img.unsqueeze(0)  # Add batch dimension

    # Run inference
    pred = model(img)[0]  # Get predictions
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)  # Apply NMS

    # Process detections
    for det in pred:  # Iterate over detections
        if len(det):
            # Rescale bounding box coordinates to the original frame size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            # Initialize Annotator
            annotator = Annotator(frame, line_width=2, font_size=0.75, example=str(names))

            # Draw bounding boxes and labels
            for *xyxy, conf, cls in det:
                sign_name = names[int(cls)]
                label = f"{sign_name} {conf:.2f}"
                annotator.box_label(xyxy, label, color=(0, 255, 0))

                # Check if the sign has been detected recently
                current_time = time.time()
                if sign_name not in last_detection_time or (current_time - last_detection_time[sign_name]) >= cooldown:
                    # Print the detected sign's name in the terminal
                    print(f"Detected: {sign_name}")
                    # Add the TTS request to the queue
                    tts_queue.put(sign_name)
                    # Update the last detection time for this sign
                    last_detection_time[sign_name] = current_time

    # Display the frame
    cv2.imshow("Traffic Sign Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# Stop the TTS worker thread
tts_queue.put(None)  # Sentinel value to stop the thread
tts_thread.join()  # Wait for the thread to finish
