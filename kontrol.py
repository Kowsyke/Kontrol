#!/usr/bin/env python3
"""
Kontrol v0.2-dev — Hand gesture mouse control
MediaPipe Tasks API (0.10+) → ydotool (no pynput, Wayland-safe)

Controls:
  Index fingertip (LM 8)                        → cursor position
  Pinch index+thumb  (LM 8+4)                   → left click
  Pinch ring+thumb   (LM 16+4)                  → right click
  Index+middle extended, hand moves up/down     → scroll
  Wrist flick (LM 0 velocity > threshold)       → KDE window tiling  ← NEW
    Flick right  → Meta+Right  (tile right)
    Flick left   → Meta+Left   (tile left)
    Flick up     → Meta+Up     (tile up / maximise)
    Flick down   → Meta+Down   (tile down / restore)
  Q in preview                                  → quit

Tuning knobs at top of file.

NOTE: ydotool key uses raw Linux keycodes (input-event-codes.h), NOT X11
keysym names. Key sequences are sent as "<code>:1 <code>:0" pairs.
"""

import cv2
import mediapipe as mp
import subprocess
import time
import os
import math
from collections import deque
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SCREEN_W         = 1920
SCREEN_H         = 1080
SMOOTH           = 0.2
PINCH_THRESHOLD  = 0.05
PINCH_COOLDOWN   = 0.4
SCROLL_DEADZONE  = 0.012
SCROLL_SPEED     = 8.0
FLICK_MIN_VEL    = 450     # px/s wrist velocity to register as flick
FLICK_WINDOW_MS  = 120     # ms window over which velocity is measured
FLICK_COOLDOWN   = 0.8     # seconds between consecutive tiling gestures
CAM_ID           = 0
FLIP             = True
ABS_SCALE_X      = 1.0
ABS_SCALE_Y      = 1.0
YDOTOOL_SOCKET   = "/run/user/1000/.ydotool_socket"
MODEL_PATH       = Path(__file__).parent / "hand_landmarker.task"
# ─────────────────────────────────────────────────────────────────────────────

# Raw Linux keycodes for tiling shortcuts (from input-event-codes.h).
# Format fed to ydotool key: "code:1 code:0" (down then up).
# KEY_LEFTMETA=125, KEY_LEFT=105, KEY_RIGHT=106, KEY_UP=103, KEY_DOWN=108
_TILING_KEYS = {
    "right": "125:1 106:1 106:0 125:0",   # Super + Right
    "left":  "125:1 105:1 105:0 125:0",   # Super + Left
    "up":    "125:1 103:1 103:0 125:0",   # Super + Up
    "down":  "125:1 108:1 108:0 125:0",   # Super + Down
}

os.environ["YDOTOOL_SOCKET"] = YDOTOOL_SOCKET

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections


# ── ydotool helpers ───────────────────────────────────────────────────────────
def ydocall(*args):
    subprocess.run(
        ["ydotool", *[str(a) for a in args]],
        env=os.environ,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def move_cursor(x: float, y: float):
    ydocall("mousemove", "-a", "-x", int(x * ABS_SCALE_X), "-y", int(y * ABS_SCALE_Y))

def left_click():
    ydocall("click", "0xC0")

def right_click():
    ydocall("click", "0xC1")   # BTN_RIGHT = evdev id 1 → 0x01|0xC0

def scroll_up(ticks: int = 1):
    ydocall("mousemove", "--wheel", "-y", str(-ticks))

def scroll_down(ticks: int = 1):
    ydocall("mousemove", "--wheel", "-y", str(ticks))

def tiling_key(direction: str):
    """
    Fire KDE Plasma 6 window-tiling shortcut (Meta+Arrow).
    ydotool key takes raw Linux keycodes, NOT X11 keysym names.
    Codes are separated by spaces; each "code:1" = key-down, "code:0" = key-up.
    """
    seq = _TILING_KEYS.get(direction)
    if seq:
        # Split into individual tokens and pass as separate args to ydotool key
        ydocall("key", *seq.split())


# ── Landmark math ─────────────────────────────────────────────────────────────
def pinch_dist(lm, a: int, b: int) -> float:
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

def is_finger_extended(lm, tip: int, pip: int) -> bool:
    """Fingertip above PIP joint = extended (lower y = higher on screen)."""
    return lm[tip].y < lm[pip].y

def is_scroll_pose(lm) -> bool:
    """Index + middle up; ring + pinky curled."""
    return (is_finger_extended(lm, 8,  6)  and
            is_finger_extended(lm, 12, 10) and
            not is_finger_extended(lm, 16, 14) and
            not is_finger_extended(lm, 20, 18))

def check_flick(history: deque) -> str | None:
    """
    Compute wrist velocity over the last FLICK_WINDOW_MS milliseconds.
    Returns 'left'|'right'|'up'|'down' if speed > FLICK_MIN_VEL px/s,
    based on the dominant axis.  Returns None otherwise.

    history entries: (timestamp_s, x_px, y_px)
    """
    if len(history) < 2:
        return None

    t_now, x_now, y_now = history[-1]
    t_cutoff = t_now - FLICK_WINDOW_MS / 1000.0

    # Oldest sample still within the measurement window
    oldest = None
    for entry in history:
        if entry[0] >= t_cutoff:
            oldest = entry
            break
    if oldest is None or oldest is history[-1]:
        return None

    t0, x0, y0 = oldest
    dt = t_now - t0
    if dt < 0.001:
        return None

    vx = (x_now - x0) / dt   # px/s
    vy = (y_now - y0) / dt

    if math.hypot(vx, vy) < FLICK_MIN_VEL:
        return None

    # Primary axis decides direction
    if abs(vx) >= abs(vy):
        return "right" if vx > 0 else "left"
    else:
        return "down" if vy > 0 else "up"


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    cx, cy        = float(SCREEN_W) / 2, float(SCREEN_H) / 2
    pinch_held_L  = False;  last_click_L = 0.0
    pinch_held_R  = False;  last_click_R = 0.0
    scroll_ref_y  = None
    wrist_history = deque(maxlen=30)   # (t, x_px, y_px)
    last_flick_t  = 0.0
    fps = 0.0;  frame_count = 0;  fps_timer = time.time()

    print("Kontrol running — press Q in preview to quit")

    with HandLandmarker.create_from_options(opts) as detector:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.time()
            frame_count += 1
            if now - fps_timer >= 1.0:
                fps = frame_count / (now - fps_timer)
                frame_count = 0;  fps_timer = now

            if FLIP:
                frame = cv2.flip(frame, 1)

            fh_px, fw_px = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)

            gesture_label = "NONE"

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]

                # Always track wrist position for flick detection
                wrist_history.append((now,
                                      lm[0].x * SCREEN_W,
                                      lm[0].y * SCREEN_H))

                pd_L = pinch_dist(lm, 4, 8)
                pd_R = pinch_dist(lm, 4, 16)

                if is_scroll_pose(lm):
                    # ── Scroll — suppress flick detection while scrolling ──
                    cur_y = lm[8].y
                    if scroll_ref_y is None:
                        scroll_ref_y = cur_y
                    else:
                        dy = cur_y - scroll_ref_y
                        scroll_ref_y = cur_y
                        if abs(dy) > SCROLL_DEADZONE:
                            ticks = max(1, int(abs(dy) * SCROLL_SPEED))
                            if dy < 0:
                                scroll_up(ticks);   gesture_label = f"SCROLL UP x{ticks}"
                            else:
                                scroll_down(ticks); gesture_label = f"SCROLL DN x{ticks}"
                        else:
                            gesture_label = "SCROLL"

                    for lm_idx in [8, 12]:
                        px = (int(lm[lm_idx].x * fw_px), int(lm[lm_idx].y * fh_px))
                        cv2.circle(frame, px, 14, (0, 200, 255), -1)
                        cv2.circle(frame, px, 14, (255, 255, 255), 2)

                else:
                    scroll_ref_y = None

                    # ── Wrist flick → KDE window tiling ───────────────────
                    flick = check_flick(wrist_history)
                    if flick and (now - last_flick_t) > FLICK_COOLDOWN:
                        tiling_key(flick)
                        last_flick_t = now
                        gesture_label = f"FLICK {flick.upper()}"
                        wrist_history.clear()  # prevent re-triggering on same motion
                    else:
                        # ── Cursor + clicks ────────────────────────────────
                        tx = lm[8].x * SCREEN_W
                        ty = lm[8].y * SCREEN_H
                        cx = cx * (1.0 - SMOOTH) + tx * SMOOTH
                        cy = cy * (1.0 - SMOOTH) + ty * SMOOTH
                        move_cursor(cx, cy)

                        if pd_R < PINCH_THRESHOLD:
                            if not pinch_held_R and (now - last_click_R) > PINCH_COOLDOWN:
                                right_click(); last_click_R = now
                            pinch_held_R = True
                            gesture_label = f"RIGHT CLICK d={pd_R:.3f}"
                        else:
                            pinch_held_R = False

                        if pd_L < PINCH_THRESHOLD and not (pd_R < PINCH_THRESHOLD):
                            if not pinch_held_L and (now - last_click_L) > PINCH_COOLDOWN:
                                left_click(); last_click_L = now
                            pinch_held_L = True
                            gesture_label = f"LEFT CLICK  d={pd_L:.3f}"
                        else:
                            pinch_held_L = False

                        if not (pd_L < PINCH_THRESHOLD) and not (pd_R < PINCH_THRESHOLD):
                            gesture_label = "CURSOR"

                    # Skeleton
                    for c in HandLandmarksConnections.HAND_CONNECTIONS:
                        ax, ay = int(lm[c.start].x*fw_px), int(lm[c.start].y*fh_px)
                        bx, by = int(lm[c.end].x  *fw_px), int(lm[c.end].y  *fh_px)
                        cv2.line(frame, (ax,ay), (bx,by), (60,160,60), 1)
                    for lmk in lm:
                        cv2.circle(frame, (int(lmk.x*fw_px), int(lmk.y*fh_px)),
                                   3, (100,200,100), -1)
            else:
                scroll_ref_y = None

            cv2.putText(frame, gesture_label, (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 80), 2)
            cv2.putText(frame, f"FPS {fps:.0f}", (fw_px-90, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.imshow("Kontrol", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit."); break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
