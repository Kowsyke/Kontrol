#!/usr/bin/env python3
"""
Kontrol v0.2-dev — Hand gesture mouse control
MediaPipe Tasks API (0.10+) → ydotool (no pynput, Wayland-safe)

Controls:
  Index fingertip (landmark 8)             → cursor position
  Pinch index ↔ thumb  (landmarks 8+4)     → left click
  Pinch ring  ↔ thumb  (landmarks 16+4)    → right click  ← NEW
  Q in preview window                      → quit

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
    # BTN_RIGHT = Linux evdev 0x111 = button id 1; 0xC0 flags = full click
    # Encoding: button_id | 0xC0  →  0x01 | 0xC0 = 0xC1
    ydocall("click", "0xC1")


# ── Landmark math ─────────────────────────────────────────────────────────────
def pinch_dist(lm, a: int, b: int) -> float:
    """Normalized Euclidean distance between two landmarks."""
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


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
    fps = 0.0;  frame_count = 0;  fps_timer = time.time()

    print("Kontrol running — press Q in preview to quit")
    print(f"  SCREEN {SCREEN_W}x{SCREEN_H}  SMOOTH {SMOOTH}"
          f"  PINCH_THRESHOLD {PINCH_THRESHOLD}  COOLDOWN {PINCH_COOLDOWN}s")

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

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]

                # ── Cursor: index fingertip (LM 8) → screen position ──────
                tx = lm[8].x * SCREEN_W
                ty = lm[8].y * SCREEN_H
                cx = cx * (1.0 - SMOOTH) + tx * SMOOTH
                cy = cy * (1.0 - SMOOTH) + ty * SMOOTH
                move_cursor(cx, cy)

                pd_L = pinch_dist(lm, 4, 8)    # thumb <-> index
                pd_R = pinch_dist(lm, 4, 16)   # thumb <-> ring

                # ── Right click: ring+thumb pinch (LM 16+4) ───────────────
                # Checked first so right-click takes priority over left-click
                if pd_R < PINCH_THRESHOLD:
                    if not pinch_held_R and (now - last_click_R) > PINCH_COOLDOWN:
                        right_click()
                        last_click_R = now
                    pinch_held_R = True
                else:
                    pinch_held_R = False

                # ── Left click: index+thumb pinch (LM 8+4) ────────────────
                # Only fires when right pinch is NOT active (avoid dual-click)
                if pd_L < PINCH_THRESHOLD and not (pd_R < PINCH_THRESHOLD):
                    if not pinch_held_L and (now - last_click_L) > PINCH_COOLDOWN:
                        left_click()
                        last_click_L = now
                    pinch_held_L = True
                else:
                    pinch_held_L = False

                # ── Skeleton + pinch indicators ───────────────────────────
                for c in HandLandmarksConnections.HAND_CONNECTIONS:
                    ax, ay = int(lm[c.start].x * fw_px), int(lm[c.start].y * fh_px)
                    bx, by = int(lm[c.end].x   * fw_px), int(lm[c.end].y   * fh_px)
                    cv2.line(frame, (ax, ay), (bx, by), (60, 160, 60), 1)
                for lmk in lm:
                    cv2.circle(frame, (int(lmk.x * fw_px), int(lmk.y * fh_px)), 3,
                               (100, 200, 100), -1)

                is_R = pd_R < PINCH_THRESHOLD
                is_L = pd_L < PINCH_THRESHOLD
                ring_px  = (int(lm[16].x * fw_px), int(lm[16].y * fh_px))
                thumb_px = (int(lm[4].x  * fw_px), int(lm[4].y  * fh_px))
                idx_px   = (int(lm[8].x  * fw_px), int(lm[8].y  * fh_px))
                cv2.line(frame, ring_px, thumb_px, (0, 0, 255) if is_R else (80, 80, 200), 2)
                cv2.line(frame, idx_px,  thumb_px, (0, 0, 255) if is_L else (80, 200, 80), 2)
                cv2.circle(frame, idx_px,  12, (0, 0, 255) if is_L else (0, 255, 80),  -1)
                cv2.circle(frame, ring_px, 12, (0, 0, 255) if is_R else (180, 80, 0),  -1)

                if is_R:
                    status, sc = f"RIGHT CLICK  d={pd_R:.3f}", (0, 80, 255)
                elif is_L:
                    status, sc = f"LEFT CLICK   d={pd_L:.3f}", (0, 200, 80)
                else:
                    status, sc = f"L={pd_L:.3f}  R={pd_R:.3f}", (180, 180, 180)
                cv2.putText(frame, status, (10, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)
            else:
                cv2.putText(frame, "no hand detected", (10, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)

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
