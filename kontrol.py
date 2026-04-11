#!/usr/bin/env python3
"""
Kontrol v0.1 — Hand gesture mouse control
MediaPipe Tasks API (0.10+) → ydotool (no pynput, Wayland-safe)

Controls:
  Index fingertip (landmark 8) → cursor position
  Pinch index ↔ thumb (landmarks 8 ↔ 4) → left click
  Q in preview window → quit

Tuning knobs at top of file:
  SMOOTH          — EMA factor: 0.1 = buttery, 0.5 = snappy
  PINCH_THRESHOLD — normalized pinch distance that triggers click
  PINCH_COOLDOWN  — seconds between allowed clicks
  SCREEN_W/H      — match your primary monitor resolution
  ABS_SCALE_X/Y   — set to 32767/1920 etc. if cursor jumps to wrong place
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
SMOOTH          = 0.2          # EMA smoothing factor (0.1–0.5)
PINCH_THRESHOLD = 0.05         # normalized pinch distance → click
PINCH_COOLDOWN  = 0.4          # seconds between click events
CAM_ID          = 0
FLIP            = True         # mirror webcam for natural feel

# If cursor jumps to wrong spot, try ABS_SCALE_X = 32767 / SCREEN_W etc.
ABS_SCALE_X     = 1.0
ABS_SCALE_Y     = 1.0

YDOTOOL_SOCKET  = "/run/user/1000/.ydotool_socket"
MODEL_PATH      = Path(__file__).parent / "hand_landmarker.task"
# ─────────────────────────────────────────────────────────────────────────────

os.environ["YDOTOOL_SOCKET"] = YDOTOOL_SOCKET

BaseOptions      = mp.tasks.BaseOptions
HandLandmarker   = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode      = mp.tasks.vision.RunningMode

# Drawing helpers (still available in 0.10 tasks)
_drawing = mp.tasks.vision.drawing_utils
_styles  = mp.tasks.vision.drawing_styles
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections


def ydocall(*args):
    """Fire-and-forget ydotool subprocess."""
    subprocess.run(
        ["ydotool", *[str(a) for a in args]],
        env=os.environ,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def move_cursor(x: float, y: float):
    ydocall("mousemove", "-a",
            "-x", int(x * ABS_SCALE_X),
            "-y", int(y * ABS_SCALE_Y))


def left_click():
    ydocall("click", "0xC0")


def pinch_dist(lm, a: int, b: int) -> float:
    """Normalized Euclidean distance between two landmarks."""
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def run():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}\n"
                                "Download with:\n"
                                "  curl -L -o hand_landmarker.task "
                                "https://storage.googleapis.com/mediapipe-models/"
                                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

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

    cx, cy      = SCREEN_W / 2.0, SCREEN_H / 2.0
    last_click  = 0.0
    pinch_held  = False
    fps_timer   = time.time()
    fps         = 0.0
    frame_count = 0

    print("Kontrol v0.1 running — press Q in preview window to quit")
    print(f"  SCREEN  {SCREEN_W}×{SCREEN_H}  |  SMOOTH {SMOOTH}"
          f"  |  PINCH_THRESHOLD {PINCH_THRESHOLD}  |  COOLDOWN {PINCH_COOLDOWN}s")
    print(f"  MODEL   {MODEL_PATH}")
    print(f"  SOCKET  {YDOTOOL_SOCKET}")

    with HandLandmarker.create_from_options(opts) as detector:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps         = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer   = now

            if FLIP:
                frame = cv2.flip(frame, 1)

            fh, fw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)

            info_color = (200, 200, 200)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]   # first hand

                # ── Cursor: index fingertip = landmark 8 ─────────────────
                tx = lm[8].x * SCREEN_W
                ty = lm[8].y * SCREEN_H
                cx = cx * (1.0 - SMOOTH) + tx * SMOOTH
                cy = cy * (1.0 - SMOOTH) + ty * SMOOTH
                move_cursor(cx, cy)

                # ── Pinch: thumb tip=4, index tip=8 ──────────────────────
                dist     = pinch_dist(lm, 4, 8)
                is_pinch = dist < PINCH_THRESHOLD

                if is_pinch:
                    if not pinch_held and (now - last_click) > PINCH_COOLDOWN:
                        left_click()
                        last_click = now
                        pinch_held = True
                else:
                    pinch_held = False

                # ── Draw landmarks on preview frame ───────────────────────
                # Convert normalized → pixel for drawing
                tip_px   = (int(lm[8].x * fw), int(lm[8].y * fh))
                thumb_px = (int(lm[4].x * fw), int(lm[4].y * fh))

                # Skeleton (manual, since drawing_utils may need NormalizedLandmarkList)
                for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    a_idx = connection.start
                    b_idx = connection.end
                    ax = int(lm[a_idx].x * fw); ay = int(lm[a_idx].y * fh)
                    bx = int(lm[b_idx].x * fw); by = int(lm[b_idx].y * fh)
                    cv2.line(frame, (ax, ay), (bx, by), (80, 180, 80), 1)

                for i, lmk in enumerate(lm):
                    px = int(lmk.x * fw); py = int(lmk.y * fh)
                    cv2.circle(frame, (px, py), 3, (120, 220, 120), -1)

                # Pinch line
                line_color = (0, 0, 255) if is_pinch else (100, 200, 100)
                cv2.line(frame, tip_px, thumb_px, line_color, 2)

                # Index fingertip big dot
                dot_color = (0, 0, 255) if is_pinch else (0, 255, 80)
                cv2.circle(frame, tip_px,   14, dot_color,       -1)
                cv2.circle(frame, tip_px,   14, (255,255,255),    2)
                cv2.circle(frame, thumb_px, 10, (255, 180,   0), -1)

                # Status
                status       = f"CLICK! d={dist:.3f}" if is_pinch else f"d={dist:.3f}"
                status_color = (0, 80, 255) if is_pinch else (80, 220, 80)
                cv2.putText(frame, status, (10, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(frame, f"cursor ({int(cx)}, {int(cy)})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, info_color, 1)
            else:
                cv2.putText(frame, "no hand detected", (10, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)

            cv2.putText(frame, f"FPS {fps:.0f}", (fw - 90, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 1)
            cv2.imshow("Kontrol v0.1", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
