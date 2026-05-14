import cv2
import numpy as np
from hand_tracker import HandTracker
from gestures import GestureController

# Global variable for the fade-in/out effect
toast_alpha = 0.0 

def draw_modern_hud(frame, status, state, ratio, controller, hand_y=None):
    global toast_alpha
    h, w, _ = frame.shape
    
    # 1. Top HUD Bar (Translucent Glass)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Theme Colors
    colors = {"IDLE": (255, 255, 255), "PINCH": (0, 255, 127), "OPEN": (255, 50, 50)}
    theme_color = colors.get(state, (255, 255, 255))

    # 2. System Info (Top Left & Right)
    cv2.putText(frame, f"SYS: {state}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"RATIO: {ratio:.2f}", (w - 110, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    # 3. ACTION TOAST (Center Top with Fade Effect)
    # Target alpha is 1.0 if controller has a message, 0.0 if empty
    target_alpha = 1.0 if controller.toast_msg != "" else 0.0
    
    # Interpolate alpha (0.1 = speed of the fade)
    toast_alpha += (target_alpha - toast_alpha) * 0.1 

    if toast_alpha > 0.01:
        # Create a separate layer to apply alpha blending to the toast
        toast_layer = frame.copy()
        
        # Determine the message to show (either current status or the stored toast)
        display_msg = controller.toast_msg if controller.toast_msg != "" else status
        
        text_size = cv2.getTextSize(display_msg, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
        tx, ty = (w - text_size[0]) // 2, 85
        
        # Draw box and text on the temporary layer
        cv2.rectangle(toast_layer, (tx - 15, ty - 25), (tx + text_size[0] + 15, ty + 10), (20, 20, 20), -1)
        cv2.putText(toast_layer, display_msg, (tx, ty - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Blend the layer into the main frame
        cv2.addWeighted(toast_layer, toast_alpha, frame, 1 - toast_alpha, 0, frame)

    # 4. VOL SLIDER (Right Edge)
    if state == "PINCH" and hand_y:
        bar_x = w - 40
        start_y, end_y = 120, h - 120
        cv2.line(frame, (bar_x, start_y), (bar_x, end_y), (40, 40, 40), 2, cv2.LINE_AA)
        
        # Map hand Y coordinate to the visual slider
        current_pos = int(np.interp(hand_y, [0.2, 0.8], [start_y, end_y]))
        current_pos = np.clip(current_pos, start_y, end_y)
        
        # Glowing Indicator
        cv2.circle(frame, (bar_x, current_pos), 6, theme_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (bar_x, current_pos), 10, theme_color, 1, cv2.LINE_AA)

def draw_tech_skeleton(frame, landmarks, state):
    h, w, _ = frame.shape
    color = (0, 255, 127) if state == "PINCH" else (255, 255, 255)
    
    connections = [(0,1,2,3,4), (0,5,6,7,8), (5,9,13,17), (0,17,18,19,20)]
    for path in connections:
        for i in range(len(path)-1):
            p1 = (int(landmarks[path[i]].x * w), int(landmarks[path[i]].y * h))
            p2 = (int(landmarks[path[i+1]].x * w), int(landmarks[path[i+1]].y * h))
            cv2.line(frame, p1, p2, (100, 100, 100), 1, cv2.LINE_AA)

    for i in [0, 4, 8]:
        cx, cy = int(landmarks[i].x * w), int(landmarks[i].y * h)
        cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

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
        status = "READY"
        hand_y = None

        if results.hand_landmarks:
            raw_landmarks = results.hand_landmarks[0]
            smoothed = tracker.get_smoothed_landmarks(raw_landmarks, prev_landmarks, beta=0.15)
            prev_landmarks = smoothed
            
            is_open = tracker.is_palm_open(smoothed)
            status = controller.process_gestures(smoothed, is_open)
            hand_y = (smoothed[4].y + smoothed[8].y) / 2
            
            draw_tech_skeleton(frame, smoothed, controller.state)

        # CRITICAL: Pass the 'controller' object here for toast/fade management
        draw_modern_hud(frame, status, controller.state, controller.smooth_ratio, controller, hand_y)
        
        cv2.imshow('Cyber Gesture Interface', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()