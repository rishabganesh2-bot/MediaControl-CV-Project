import mediapipe as mp
import cv2
import time

class HandTracker:
    def __init__(self, model_path='hand_landmarker.task'):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1 
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def find_hands(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    def is_palm_open(self, hand_landmarks):
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            if hand_landmarks[tip].y > hand_landmarks[pip].y:
                return False 
        return True
    
    def get_smoothed_landmarks(self, hand_landmarks, prev_landmarks, beta=0.2):
        """Applies a simple Low-Pass Filter to the coordinates."""
        if prev_landmarks is None:
            return hand_landmarks
        
        smoothed = []
        for curr, prev in zip(hand_landmarks, prev_landmarks):
            # Move only a fraction of the way to the new point
            s_x = prev.x + beta * (curr.x - prev.x)
            s_y = prev.y + beta * (curr.y - prev.y)
            # Create a mock object or dict to mimic the landmark structure
            smoothed.append(type('obj', (object,), {'x': s_x, 'y': s_y, 'z': curr.z}))
        return smoothed