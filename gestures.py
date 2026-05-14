import math
import keyboard
import time

class GestureController:
    def __init__(self):
        self.smooth_ratio = 0.5
        self.alpha = 0.3
        self.is_dragging = False
        self.drag_anchor_y = 0
        self.drag_threshold = 0.04
        self.last_vol_time = 0
        self.vol_cooldown_ms = 0.1
        self.last_pause_time = 0
        self.palm_active = False
        self.state = "IDLE" 

    def process_gestures(self, hand_landmarks, is_open):
        thumb, index, wrist, m_mcp = hand_landmarks[4], hand_landmarks[8], hand_landmarks[0], hand_landmarks[9]
        
        # 1. Distance Normalization
        ref_dist = math.hypot(m_mcp.x - wrist.x, m_mcp.y - wrist.y)
        pinch_dist = math.hypot(index.x - thumb.x, index.y - thumb.y)
        current_ratio = pinch_dist / ref_dist if ref_dist != 0 else 0
        
        # 2. EMA Smoothing
        self.smooth_ratio = (self.alpha * current_ratio) + ((1 - self.alpha) * self.smooth_ratio)
        
        current_time = time.time()
        current_y = (thumb.y + index.y) / 2
        
        # 3. Logic & State Management
        # We process the logic and return the string IMMEDIATELY 
        # while also updating self.state for the UI.
        
        if self.smooth_ratio < 0.35:
            self.state = "PINCH"
            return self._handle_volume(current_y, current_time)
            
        elif is_open and self.smooth_ratio > 1.10:
            self.state = "OPEN"
            return self._handle_play_pause(current_time)
            
        else:
            self.state = "IDLE"
            self.is_dragging = False
            self.palm_active = False
            return "Searching..."

    def _handle_volume(self, current_y, current_time):
        self.palm_active = False
        if not self.is_dragging:
            self.is_dragging = True
            self.drag_anchor_y = current_y
            return "Pinching..."
        
        diff = self.drag_anchor_y - current_y
        
        if abs(diff) > self.drag_threshold and (current_time - self.last_vol_time > self.vol_cooldown_ms):
            cmd = 'volume up' if diff > 0 else 'volume down'
            keyboard.press_and_release(cmd)
            self.last_vol_time = current_time
            self.drag_anchor_y = current_y
            return f"VOL {'UP' if diff > 0 else 'DOWN'}"
            
        return "Adjusting Volume"

    def _handle_play_pause(self, current_time):
        self.is_dragging = False
        # 1.5 second cooldown prevents "machine-gun" pausing
        if not self.palm_active and (current_time - self.last_pause_time > 1.5):
            keyboard.press_and_release('play/pause')
            self.last_pause_time = current_time
            self.palm_active = True
            return "Action: PLAY/PAUSE"
        return "Palm Detected"