import cv2
import mediapipe as mp
import time
import math
import keyboard
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. Hand Tracker Setup ---
model_path = 'hand_landmarker.task' 
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1 
)

# --- 2. Persistence Variables ---
smooth_ratio = 0.5
alpha = 0.3  
cooldown = 0
status = "Initializing..." # Moved outside the loop to prevent flickering

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        # Reset status if no hand is in frame
        if not detection_result.hand_landmarks:
            status = "Hand Not Detected"
        else:
            for hand_landmarks in detection_result.hand_landmarks:
                thumb = hand_landmarks[4]
                index = hand_landmarks[8]
                wrist = hand_landmarks[0]
                m_mcp = hand_landmarks[9]

                # Depth-aware scaling
                ref_dist = math.hypot(m_mcp.x - wrist.x, m_mcp.y - wrist.y)
                pinch_dist = math.hypot(index.x - thumb.x, index.y - thumb.y)
                current_ratio = pinch_dist / ref_dist if ref_dist != 0 else 0

                # Smoothing
                smooth_ratio = (alpha * current_ratio) + ((1 - alpha) * smooth_ratio)

                # --- TRIGGER LOGIC ---
                cooldown += 1
                if cooldown % 4 == 0:
                    if smooth_ratio > 1.1:
                        keyboard.press_and_release('volume up')
                        status = "Volume UP"
                    elif smooth_ratio < 0.35:
                        keyboard.press_and_release('volume down')
                        status = "Volume DOWN"
                    else:
                        status = "Monitoring..."

                # Visual Feedback
                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (20, 20, 100), 3) # Darker line

        # --- DARK TEXT OVERLAY ---
        # OpenCV uses BGR: (20, 20, 100) is a dark navy/maroon mix
        dark_color = (60, 20, 20) 
        
        cv2.putText(frame, f"Ratio: {smooth_ratio:.2f}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, dark_color, 2)
        cv2.putText(frame, f"Status: {status}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, dark_color, 2)

        cv2.imshow('Depth-Aware Volume Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()