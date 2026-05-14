import mediapipe as mp
import cv2
import math
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
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        return self.landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    # ─────────────────────────────────────────────────────────
    #  Palm size (ref_dist) — used by GestureController for
    #  dynamic threshold scaling. Exposed here so there is one
    #  canonical calculation rather than two diverging copies.
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def palm_size(lm) -> float:
        """Normalised wrist→middle-MCP distance (landmark 0 → 9)."""
        return math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)

    # ─────────────────────────────────────────────────────────
    #  Finger-extension test — orientation-aware
    #
    #  The y-axis test (tip.y < pip.y) only works when the hand
    #  is roughly upright.  If the hand is tilted/sideways the
    #  wrist→MCP vector is no longer aligned with gravity, so we
    #  project each fingertip vector onto that reference axis
    #  instead of comparing raw y values.
    #
    #  For each finger we check:
    #    dot(tip - mcp, palm_axis) > 0   →  finger is "away from palm"
    #  where palm_axis = normalised (middle_mcp - wrist).
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _finger_extended(lm, tip_id: int, mcp_id: int, ax: float, ay: float) -> bool:
        """True when the tip is further along the palm axis than the MCP."""
        dx = lm[tip_id].x - lm[mcp_id].x
        dy = lm[tip_id].y - lm[mcp_id].y
        return (dx * ax + dy * ay) > 0.02          # small dead-zone

    @staticmethod
    def _finger_curled(lm, tip_id: int, pip_id: int, ax: float, ay: float) -> bool:
        """True when the tip is NOT extended past the PIP joint."""
        dx = lm[tip_id].x - lm[pip_id].x
        dy = lm[tip_id].y - lm[pip_id].y
        return (dx * ax + dy * ay) <= 0.02

    @staticmethod
    def _palm_axis(lm):
        """Unit vector pointing from wrist (0) toward middle MCP (9)."""
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            return 0.0, -1.0           # fallback: straight up
        return dx / mag, dy / mag

    # ── Public pose detectors (single source of truth) ───────

    def is_palm_open(self, lm) -> bool:
        """All four fingers extended in the direction of the palm axis."""
        ax, ay = self._palm_axis(lm)
        return all(
            self._finger_extended(lm, tip, mcp, ax, ay)
            for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]
        )

    def is_two_finger_pose(self, lm) -> bool:
        """
        Index and middle extended; ring and pinky curled.
        Orientation-aware: uses palm-axis projection so the pose
        is detected correctly even when the hand is tilted sideways.
        """
        ax, ay = self._palm_axis(lm)
        index_up  = self._finger_extended(lm,  8,  5, ax, ay)
        middle_up = self._finger_extended(lm, 12,  9, ax, ay)
        ring_down = self._finger_curled  (lm, 16, 14, ax, ay)
        pinky_down= self._finger_curled  (lm, 20, 18, ax, ay)
        return index_up and middle_up and ring_down and pinky_down

    # ── Smoothing ─────────────────────────────────────────────

    def get_smoothed_landmarks(self, hand_landmarks, prev_landmarks, beta=0.2):
        """Low-pass filter: moves beta of the way to the new position each frame."""
        if prev_landmarks is None:
            return hand_landmarks
        smoothed = []
        for curr, prev in zip(hand_landmarks, prev_landmarks):
            sx = prev.x + beta * (curr.x - prev.x)
            sy = prev.y + beta * (curr.y - prev.y)
            smoothed.append(
                type('LM', (), {'x': sx, 'y': sy, 'z': curr.z})()
            )
        return smoothed