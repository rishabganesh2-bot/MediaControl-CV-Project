import math
import time
from pynput.keyboard import Controller as KeyboardController, Key

_kb = KeyboardController()

def _press_media(key: Key):
    _kb.press(key)
    _kb.release(key)


_IDLE     = 0
_ARMED    = 1
_TRACKING = 2


class GestureController:
    def __init__(self):
        #Pinch / volume
        self.smooth_ratio   = 0.5
        self.alpha          = 0.3
        self.is_dragging    = False
        self.drag_anchor_y  = 0.0
        self.drag_threshold = 0.04
        self.last_vol_time  = 0.0
        self.vol_cooldown   = 0.15

        #Play / pause
        self.last_pause_time = 0.0
        self.palm_active     = False

        # UI
        self.state       = "IDLE"
        self.toast_msg   = ""
        self.toast_timer = 0

        #Swipe FSM
        self._fsm         = _IDLE
        self._arm_count   = 0
        self._arm_frames  = 4        

        self._anchor_x    = 0.0
        self._anchor_y    = 0.0
        self._smooth_x    = 0.0
        self._sx_alpha    = 0.40     

        self._threshold_ratio = 0.9
        self._vert_limit  = 0.75     
        self._cooldown    = 1.2      
        self._last_fire   = 0.0

        self._grace_max   = 6
        self._grace_count = 0
        self._last_palm_size = 0.05  

        # HUD data
        self.swipe_trail : list  = []
        self.arrow_row_y : float = 0.5  

    def process_gestures(self, hand_landmarks, is_open, palm_size: float,
                         two_finger: bool) -> str:
        thumb = hand_landmarks[4]
        index = hand_landmarks[8]
        wrist = hand_landmarks[0]
        m_mcp = hand_landmarks[9]

        if palm_size > 0.01:
            self._last_palm_size = palm_size

        ref_dist      = math.hypot(m_mcp.x - wrist.x, m_mcp.y - wrist.y)
        pinch_dist    = math.hypot(index.x - thumb.x, index.y - thumb.y)
        current_ratio = pinch_dist / ref_dist if ref_dist > 0 else 0
        self.smooth_ratio = self.alpha * current_ratio + (1 - self.alpha) * self.smooth_ratio

        now      = time.time()
        center_y = (thumb.y + index.y) / 2
        self._grace_count = 0

        #Gesture priority
        if self.smooth_ratio < 0.35:
            self.state = "PINCH"
            self._reset_fsm()
            res = self._handle_volume(center_y, now)

        elif two_finger:
            self.state       = "SWIPE"
            self.is_dragging = False
            self.palm_active = False
            res = self._run_fsm(index.x, index.y, now)

        elif is_open and self.smooth_ratio > 1.10:
            self.state = "OPEN"
            self._reset_fsm()
            res = self._handle_play_pause(now)

        else:
            self.state       = "IDLE"
            self.is_dragging = False
            self.palm_active = False
            self._reset_fsm()
            res = "READY"

        # Toast countdown
        action_triggered = any(k in res for k in ("VOL UP", "VOL DOWN", "PLAY", "PAUSE", "NEXT", "PREV"))

        if action_triggered:
            self.toast_msg   = res
            self.toast_timer = 22
        elif self.toast_timer > 0:
            self.toast_timer -= 1
        else:
            self.toast_msg = ""

        return res

    def tick_no_hand(self):
        self._grace_count += 1
        if self._grace_count > self._grace_max:
            self.state = "IDLE"
            self._reset_fsm()

    def _run_fsm(self, raw_x: float, raw_y: float, now: float) -> str:
        x, y = raw_x, raw_y

        if now - self._last_fire < self._cooldown:
            self._arm_count = 0
            return "SWIPE COOLDOWN"

        if self._fsm == _IDLE:
            self._arm_count += 1
            if self._arm_count >= self._arm_frames:
                self._smooth_x = x
                self._fsm      = _ARMED
            return "POSE HELD"

        if self._fsm == _ARMED:
            self._smooth_x   = self._sx_alpha * x + (1 - self._sx_alpha) * self._smooth_x
            self._anchor_x   = self._smooth_x
            self._anchor_y   = y
            self.arrow_row_y = y          
            self.swipe_trail = [self._smooth_x]
            self._fsm        = _TRACKING
            return "TRACKING"

        if self._fsm == _TRACKING:
            self._smooth_x = self._sx_alpha * x + (1 - self._sx_alpha) * self._smooth_x
            self.swipe_trail.append(self._smooth_x)
            if len(self.swipe_trail) > 24:
                self.swipe_trail.pop(0)

            raw_dx = self._smooth_x - self._anchor_x
            dy     = y - self._anchor_y
            scaled_dx = raw_dx / max(self._last_palm_size, 0.01)

            if abs(raw_dx) > 0.005 and (abs(dy) / abs(raw_dx)) > self._vert_limit:
                return "SWIPING"

            if scaled_dx > self._threshold_ratio:
                _press_media(Key.media_next)
                self._last_fire = now
                self._reset_fsm()
                return "NEXT >>"

            if scaled_dx < -self._threshold_ratio:
                _press_media(Key.media_previous)
                self._last_fire = now
                self._reset_fsm()
                return "PREV <<"

            return "SWIPING"

        return "SWIPE READY"

    def _reset_fsm(self):
        self._fsm         = _IDLE
        self._arm_count   = 0
        self._anchor_x    = 0.0
        self._anchor_y    = 0.0
        self._smooth_x    = 0.0
        self._grace_count = 0
        self.swipe_trail  = []
        self.arrow_row_y  = 0.5

    def _handle_volume(self, current_y: float, now: float) -> str:
        self.palm_active = False
        if not self.is_dragging:
            self.is_dragging   = True
            self.drag_anchor_y = current_y
            return "PINCH ACTIVE"
        
        diff = self.drag_anchor_y - current_y
        if abs(diff) > self.drag_threshold and (now - self.last_vol_time > self.vol_cooldown):
            if diff > 0:
                _press_media(Key.media_volume_up)
                res = "VOL UP"
            else:
                _press_media(Key.media_volume_down)
                res = "VOL DOWN"
            self.last_vol_time  = now
            self.drag_anchor_y  = current_y
            return res
        return "ADJUSTING"

    def _handle_play_pause(self, now: float) -> str:
        self.is_dragging = False
        if not self.palm_active and (now - self.last_pause_time > 1.5):
            _press_media(Key.media_play_pause)
            self.last_pause_time = now
            self.palm_active     = True
            return "PLAY / PAUSE"
        return "PALM OPEN"