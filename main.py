import cv2
import numpy as np # Add this for drawing
from hand_tracker import HandTracker
from gestures import GestureController

def draw_ui(frame, status, state, current_y=None):
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    # 1. State-based colors
    color = (255, 255, 255) # White
    if state == "PINCH": color = (0, 255, 0)   # Green
    if state == "OPEN":  color = (255, 200, 0) # Cyan/Blue

    # 2. Draw a Modern Status Box (Top Left)
    cv2.rectangle(overlay, (20, 20), (300, 80), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"MODE: {state}", (35, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
    cv2.putText(frame, status, (35, 72), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)

    # 3. Vertical Volume Bar (Only when pinching)
    if state == "PINCH" and current_y:
        bar_x = w - 50
        bar_height = int(h * 0.6)
        bar_top = (h - bar_height) // 2
        
        # Background track
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 10, bar_top + bar_height), (50, 50, 50), -1)
        # The "Indicator" knob following your hand
        knob_y = int(current_y * h)
        knob_y = max(bar_top, min(knob_y, bar_top + bar_height)) # Clamp to bar
        cv2.circle(frame, (bar_x + 5, knob_y), 12, (0, 255, 0), -1)
        cv2.putText(frame, "VOL", (bar_x - 15, bar_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    tracker = HandTracker()
    controller = GestureController()
    cap = cv2.VideoCapture(0)
    prev_landmarks = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        
        results = tracker.find_hands(frame)
        status = "No Hand Detected"
        hand_y = None

        if results.hand_landmarks:
            raw_landmarks = results.hand_landmarks[0]
            # Smooth the jitter!
            smoothed = tracker.get_smoothed_landmarks(raw_landmarks, prev_landmarks)
            prev_landmarks = smoothed
            
            is_open = tracker.is_palm_open(smoothed)
            status = controller.process_gestures(smoothed, is_open)
            hand_y = (smoothed[4].y + smoothed[8].y) / 2 # Thumb + Index avg

            # Draw sophisticated skeleton
            for connection in [(4,3,2,1,0), (8,7,6,5,0)]: # Simplified lines
                points = [ (int(smoothed[i].x * frame.shape[1]), int(smoothed[i].y * frame.shape[0])) for i in connection ]
                for i in range(len(points)-1):
                    cv2.line(frame, points[i], points[i+1], (200, 200, 200), 1)

        draw_ui(frame, status, controller.state, hand_y)
        
        cv2.imshow('Pro Gesture HUD', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()