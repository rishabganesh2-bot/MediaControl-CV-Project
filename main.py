import cv2
import mediapipe as mp
import time
import math
import keyboard

# --- 1. Hand Tracker Setup ---
model_path = 'hand_landmarker.task' 
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1 
)

# --- 2. Helper Functions ---
def is_palm_open(hand_landmarks):
    """Checks if fingers are extended relative to the knuckles."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        if hand_landmarks[tip].y > hand_landmarks[pip].y:
            return False 
    return True

# --- 3. Control Variables ---
smooth_ratio = 0.5
alpha = 0.35          
palm_active = False   
last_pause_time = 0   
vol_cooldown = 0      
status = "Initializing..."

# --- 4. Main Loop ---
with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if not detection_result.hand_landmarks:
            status = "Hand Not Detected"
            palm_active = False 
        else:
            for hand_landmarks in detection_result.hand_landmarks:
                thumb = hand_landmarks[4]
                index = hand_landmarks[8]
                wrist = hand_landmarks[0]
                m_mcp = hand_landmarks[9]

                ref_dist = math.hypot(m_mcp.x - wrist.x, m_mcp.y - wrist.y)
                pinch_dist = math.hypot(index.x - thumb.x, index.y - thumb.y)
                current_ratio = pinch_dist / ref_dist if ref_dist != 0 else 0
                
                smooth_ratio = (alpha * current_ratio) + ((1 - alpha) * smooth_ratio)
                current_time = time.time()

                # --- GESTURE LOGIC WITH NEW RANGES ---
                
                # A. PAUSE/PLAY (Full Palm + High Ratio)
                # Pushed to 1.20 to avoid overlap with new Volume Up range
                if is_palm_open(hand_landmarks) and smooth_ratio > 1.20:
                    if not palm_active and (current_time - last_pause_time > 1.5):
                        keyboard.press_and_release('play/pause')
                        status = "EVENT: PLAY/PAUSE"
                        last_pause_time = current_time
                        palm_active = True 
                    else:
                        status = "Palm Active (Locked)"
                
                # B. VOLUME DOWN (Tight Pinch: < 0.30)
                elif smooth_ratio < 0.30:
                    palm_active = False
                    vol_cooldown += 1
                    if vol_cooldown % 4 == 0:
                        keyboard.press_and_release('volume down')
                        status = "Volume DOWN"

                # C. VOLUME UP (New Range: 0.75 - 1.15)
                elif 0.75 <= smooth_ratio <= 1.15:
                    palm_active = False
                    vol_cooldown += 1
                    if vol_cooldown % 4 == 0:
                        keyboard.press_and_release('volume up')
                        status = "Volume UP"
                
                # D. THE DEADZONE (0.30 to 0.75)
                else:
                    palm_active = False
                    status = "Idle (Deadzone)"

                # Visual Debug
                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- UI Overlay ---
        cv2.rectangle(frame, (0, 0), (320, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Ratio: {smooth_ratio:.2f}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow('Gesture Control v3', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()