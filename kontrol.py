#!/usr/bin/env python3
"""
Kontrol v0.2-dev — Hand gesture mouse control
MediaPipe Tasks API (0.10+) → ydotool (no pynput, Wayland-safe)

Controls:
  Index fingertip (landmark 8)                        → cursor position
  Pinch index ↔ thumb  (landmarks 8+4)                → left click
  Pinch ring  ↔ thumb  (landmarks 16+4)               → right click
  Index + middle extended, ring + pinky curled        → scroll mode
    Hand moves up                                     → scroll up
    Hand moves down                                   → scroll down
  Q in preview window                                 → quit

Tuning knobs at top of file.
"""

import cv2
import mediapipe as mp
import subprocess
import time
import os
import math
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SCREEN_W        = 1920
SCREEN_H        = 1080
SMOOTH          = 0.2           # EMA smoothing factor
PINCH_THRESHOLD = 0.05          # normalized pinch distance → click
PINCH_COOLDOWN  = 0.4           # seconds between click events (per button)
SCROLL_DEADZONE = 0.012         # min normalized y-delta per frame to scroll
SCROLL_SPEED    = 8.0           # ticks = int(abs(delta_y) * SCROLL_SPEED)
CAM_ID          = 0
FLIP            = True
ABS_SCALE_X     = 1.0
ABS_SCALE_Y     = 1.0
YDOTOOL_SOCKET  = "/run/user/1000/.ydotool_socket"
MODEL_PATH      = Path(__file__).parent / "hand_landmarker.task"
# ─────────────────────────────────────────────────────────────────────────────

os.environ["YDOTOOL_SOCKET"] = YDOTOOL_SOCKET

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections


# ── ydotool helpers ───────────────────────────────────────────────────────────
def ydocall(*args):
    """Fire-and-forget ydotool subprocess."""
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
    # BTN_RIGHT = evdev 0x111 = button id 1 → 0x01 | 0xC0 (down+up flags) = 0xC1
    ydocall("click", "0xC1")

def scroll_up(ticks: int = 1):
    # ydotool mousemove --wheel -y N: negative y = scroll up (content moves up)
    ydocall("mousemove", "--wheel", "-y", str(-ticks))

def scroll_down(ticks: int = 1):
    ydocall("mousemove", "--wheel", "-y", str(ticks))


# ── Landmark math ─────────────────────────────────────────────────────────────
def pinch_dist(lm, a: int, b: int) -> float:
    """Normalized Euclidean distance between two landmarks."""
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

def is_finger_extended(lm, tip: int, pip: int) -> bool:
    """True when fingertip is above its PIP joint (lower y = higher on screen)."""
    return lm[tip].y < lm[pip].y

def is_scroll_pose(lm) -> bool:
    """
    Index (LM 8) and middle (LM 12) extended above their PIP joints (6, 10).
    Ring (LM 16) and pinky (LM 20) must be curled — prevents false positives
    on an open palm that should just move the cursor.
    """
    return (is_finger_extended(lm, 8,  6)  and   # index up
            is_finger_extended(lm, 12, 10) and   # middle up
            not is_finger_extended(lm, 16, 14) and  # ring curled
            not is_finger_extended(lm, 20, 18))     # pinky curled


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Download:\n  curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

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
    scroll_ref_y  = None    # normalized index-tip y when entering scroll mode
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

                pd_L = pinch_dist(lm, 4, 8)    # thumb <-> index
                pd_R = pinch_dist(lm, 4, 16)   # thumb <-> ring

                if is_scroll_pose(lm):
                    # ── Scroll mode ───────────────────────────────────────
                    # Use index-tip normalized y as the reference axis.
                    # Finger moves up (y decreases) → scroll up.
                    cur_y = lm[8].y
                    if scroll_ref_y is None:
                        scroll_ref_y = cur_y          # anchor on first scroll frame
                    else:
                        dy = cur_y - scroll_ref_y     # positive = hand moved down
                        scroll_ref_y = cur_y          # update every frame (delta, not total)
                        if abs(dy) > SCROLL_DEADZONE:
                            ticks = max(1, int(abs(dy) * SCROLL_SPEED))
                            if dy < 0:
                                scroll_up(ticks)
                                gesture_label = f"SCROLL UP x{ticks}"
                            else:
                                scroll_down(ticks)
                                gesture_label = f"SCROLL DN x{ticks}"
                        else:
                            gesture_label = "SCROLL"

                    # Draw two-finger highlight
                    for lm_idx in [8, 12]:
                        px = (int(lm[lm_idx].x * fw_px), int(lm[lm_idx].y * fh_px))
                        cv2.circle(frame, px, 14, (0, 200, 255), -1)
                        cv2.circle(frame, px, 14, (255, 255, 255), 2)

                else:
                    scroll_ref_y = None  # reset when leaving scroll pose

                    # ── Cursor: index fingertip ────────────────────────────
                    tx = lm[8].x * SCREEN_W
                    ty = lm[8].y * SCREEN_H
                    cx = cx * (1.0 - SMOOTH) + tx * SMOOTH
                    cy = cy * (1.0 - SMOOTH) + ty * SMOOTH
                    move_cursor(cx, cy)

                    # ── Right click: ring+thumb (LM 16+4) — priority ───────
                    if pd_R < PINCH_THRESHOLD:
                        if not pinch_held_R and (now - last_click_R) > PINCH_COOLDOWN:
                            right_click()
                            last_click_R = now
                        pinch_held_R = True
                        gesture_label = f"RIGHT CLICK d={pd_R:.3f}"
                    else:
                        pinch_held_R = False

                    # ── Left click: index+thumb (LM 8+4) ──────────────────
                    if pd_L < PINCH_THRESHOLD and not (pd_R < PINCH_THRESHOLD):
                        if not pinch_held_L and (now - last_click_L) > PINCH_COOLDOWN:
                            left_click()
                            last_click_L = now
                        pinch_held_L = True
                        gesture_label = f"LEFT CLICK  d={pd_L:.3f}"
                    else:
                        pinch_held_L = False

                    if not (pd_L < PINCH_THRESHOLD) and not (pd_R < PINCH_THRESHOLD):
                        gesture_label = "CURSOR"

                    # Skeleton
                    for c in HandLandmarksConnections.HAND_CONNECTIONS:
                        ax, ay = int(lm[c.start].x * fw_px), int(lm[c.start].y * fh_px)
                        bx, by = int(lm[c.end].x   * fw_px), int(lm[c.end].y   * fh_px)
                        cv2.line(frame, (ax, ay), (bx, by), (60, 160, 60), 1)
                    for lmk in lm:
                        cv2.circle(frame, (int(lmk.x * fw_px), int(lmk.y * fh_px)),
                                   3, (100, 200, 100), -1)

                    # Pinch indicator lines
                    ring_px  = (int(lm[16].x * fw_px), int(lm[16].y * fh_px))
                    thumb_px = (int(lm[4].x  * fw_px), int(lm[4].y  * fh_px))
                    idx_px   = (int(lm[8].x  * fw_px), int(lm[8].y  * fh_px))
                    is_R = pd_R < PINCH_THRESHOLD
                    is_L = pd_L < PINCH_THRESHOLD
                    cv2.line(frame, ring_px, thumb_px, (0,0,255) if is_R else (80,80,200), 2)
                    cv2.line(frame, idx_px,  thumb_px, (0,0,255) if is_L else (80,200,80), 2)
                    cv2.circle(frame, idx_px,  12, (0,0,255) if is_L else (0,255,80),  -1)
                    cv2.circle(frame, ring_px, 12, (0,0,255) if is_R else (180,80,0),  -1)

            else:
                scroll_ref_y = None

            cv2.putText(frame, gesture_label, (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 80), 2)
            cv2.putText(frame, f"FPS {fps:.0f}", (fw_px - 90, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Kontrol", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
