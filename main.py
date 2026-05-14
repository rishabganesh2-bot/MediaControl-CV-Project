import cv2
import numpy as np
from hand_tracker import HandTracker
from gestures import GestureController

_toast_alpha = 0.0


# ═══════════════════════════════════════════════════════════
#  HUD
# ═══════════════════════════════════════════════════════════

def draw_hud(frame, state, ratio, controller, hand_y, swipe_trail):
    global _toast_alpha
    h, w = frame.shape[:2]

    THEME = {
        "IDLE":  (200, 200, 200),
        "PINCH": (0,   210,  90),
        "OPEN":  (80,  80,  255),
        "SWIPE": (50,  200, 255),
    }
    col = THEME.get(state, (200, 200, 200))

    # ── 1. Top bar ──────────────────────────────────────────
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 52), (10, 10, 10), -1)
    cv2.addWeighted(bar, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, 52), (w, 52), col, 1, cv2.LINE_AA)

    cv2.putText(frame, state, (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{ratio:.2f}", (w - 68, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # ── 2. Toast ────────────────────────────────────────────
    target = 1.0 if controller.toast_msg else 0.0
    _toast_alpha += (target - _toast_alpha) * 0.12

    if _toast_alpha > 0.02:
        msg  = controller.toast_msg
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = 0.72
        (tw, th), _ = cv2.getTextSize(msg, font, fs, 2)
        tx = (w - tw) // 2
        ty = 100
        pad = 14

        layer = frame.copy()
        cv2.rectangle(layer,
                      (tx - pad,      ty - th - pad // 2),
                      (tx + tw + pad, ty + pad // 2),
                      (18, 18, 18), -1)
        cv2.rectangle(layer,
                      (tx - pad,     ty - th - pad // 2),
                      (tx - pad + 4, ty + pad // 2),
                      col, -1)
        cv2.putText(layer, msg, (tx, ty), font, fs, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.addWeighted(layer, _toast_alpha, frame, 1 - _toast_alpha, 0, frame)

    # ── 3. Volume bar (PINCH) ───────────────────────────────
    if state == "PINCH" and hand_y is not None:
        bx      = w - 38
        bar_top = 80
        bar_bot = h - 80
        cv2.line(frame, (bx, bar_top), (bx, bar_bot), (45, 45, 45), 3, cv2.LINE_AA)
        ky = int(np.clip(np.interp(hand_y, [0.15, 0.85], [bar_top, bar_bot]),
                         bar_top, bar_bot))
        cv2.circle(frame, (bx, ky),  7, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, ky), 12, col,  1, cv2.LINE_AA)
        cv2.putText(frame, "VOL", (bx - 14, bar_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (75, 75, 75), 1, cv2.LINE_AA)

    # ── 4. Swipe arrow (SWIPE, actively tracking) ──────────
    # Arrow row follows the hand's actual height (arrow_row_y locked at arm time),
    # clipped so it never lands in the HUD bar or bottom label area.
    # Start: fixed left or right edge.  End: fixed screen centre (w//2, row_y).
    if state == "SWIPE" and swipe_trail and len(swipe_trail) >= 2:
        CYAN = (50, 200, 255)
        DIM  = (25, 75, 95)

        dx = swipe_trail[-1] - controller._anchor_x
        if abs(dx) > 0.02:
            going_right = dx > 0

            # Row is anchored to the hand's height at tracking start, not live hand_y
            row_y = int(np.clip(controller.arrow_row_y * h, 80, h - 60))

            mid_x = w // 2
            if going_right:
                start_pt = (30,      row_y)
                end_pt   = (mid_x,   row_y)
                label    = "SWIPE RIGHT >>"
            else:
                start_pt = (w - 30,  row_y)
                end_pt   = (mid_x,   row_y)
                label    = "<< SWIPE LEFT"

            # Full-width dim guide
            cv2.line(frame, (30, row_y), (w - 30, row_y), DIM, 1, cv2.LINE_AA)

            # Bright arrow: edge → centre
            cv2.arrowedLine(frame, start_pt, end_pt,
                            CYAN, 3, cv2.LINE_AA, tipLength=0.06)

            # Pulsing dot at centre tip
            cv2.circle(frame, end_pt, 6,  CYAN, -1, cv2.LINE_AA)
            cv2.circle(frame, end_pt, 11, CYAN,  1, cv2.LINE_AA)

            cv2.putText(frame, label, (20, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, CYAN, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════
#  Skeleton
# ═══════════════════════════════════════════════════════════

def draw_skeleton(frame, landmarks, state):
    h, w = frame.shape[:2]

    BONE  = (65, 65, 65)
    TIPS  = {
        "PINCH": ([0, 4, 8],             (0,   210,  90)),
        "SWIPE": ([0, 8, 12],            (50,  200, 255)),
        "OPEN":  ([0, 4, 8, 12, 16, 20], (80,  80,  255)),
    }
    dot_ids, dot_col = TIPS.get(state, ([0, 4, 8], (170, 170, 170)))

    PATHS = [
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (5, 9, 10, 11, 12),
        (9, 13, 14, 15, 16),
        (13, 17, 18, 19, 20),
        (0, 17),
        (5, 9), (9, 13), (13, 17),   # knuckle bar
    ]
    for path in PATHS:
        for i in range(len(path) - 1):
            p1 = (int(landmarks[path[i]].x * w),   int(landmarks[path[i]].y * h))
            p2 = (int(landmarks[path[i+1]].x * w), int(landmarks[path[i+1]].y * h))
            cv2.line(frame, p1, p2, BONE, 1, cv2.LINE_AA)

    for i in dot_ids:
        cx, cy = int(landmarks[i].x * w), int(landmarks[i].y * h)
        cv2.circle(frame, (cx, cy), 5, dot_col, -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════

def main():
    tracker        = HandTracker()
    controller     = GestureController()
    cap            = cv2.VideoCapture(0)
    prev_landmarks = None

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        results = tracker.find_hands(frame)
        hand_y  = None

        if results.hand_landmarks:
            raw = results.hand_landmarks[0]
            lm  = tracker.get_smoothed_landmarks(raw, prev_landmarks, beta=0.15)
            prev_landmarks = lm

            # Compute all pose facts once in main — passed to controller,
            # not re-evaluated inside gestures.py
            is_open    = tracker.is_palm_open(lm)
            two_finger = tracker.is_two_finger_pose(lm)
            p_size     = tracker.palm_size(lm)

            controller.process_gestures(lm, is_open, p_size, two_finger)
            hand_y = (lm[4].y + lm[8].y) / 2
            draw_skeleton(frame, lm, controller.state)
        else:
            # Grace period: don't reset FSM immediately on a missed frame
            controller.tick_no_hand()
            prev_landmarks = None

        swipe_trail = controller.swipe_trail or None

        draw_hud(frame,
                 controller.state,
                 controller.smooth_ratio,
                 controller,
                 hand_y,
                 swipe_trail)

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()