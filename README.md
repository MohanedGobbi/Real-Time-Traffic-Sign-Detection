Traffic Sign Detection using YOLOv5
This project is a real-time traffic sign detection system that uses YOLOv5 and OpenCV to identify and announce traffic signs from a live camera feed. The system utilizes a pre-trained model to detect traffic signs and employs text-to-speech (TTS) to verbally announce the detected signs.

Features
✅ Real-time traffic sign detection with bounding boxes
✅ Uses YOLOv5 for accurate object detection
✅ Text-to-speech (TTS) to announce detected signs
✅ Cooldown mechanism to prevent repeated detections

Installation
Clone this repository:

git clone https://github.com/yourusername/traffic-sign-detection.git  
cd traffic-sign-detection

Install dependencies:

pip install -r requirements.txt

Run the program:

python traffic_sign_detection.py

Usage
The script will access the camera feed, detect traffic signs, and draw bounding boxes around them.
Detected signs are announced using text-to-speech.
Press 'q' to exit the program.

License
This project is licensed under the MIT License.
