#!/usr/bin/env python3
"""
test_cursor.py — minimal cursor tracking diagnostic
Bypasses all Kontrol smoothing/gestures to verify raw ydotool→cursor pipeline.

Diagnosis result (2026-04-12):
  ydotoold virtual device (event26) has NO EV_ABS capability — 'ev' sysfs = 0x07.
  Every ydotool mousemove --absolute was silently dropped by the kernel.
  ydotool mousemove (relative, no --absolute) emits EV_REL which event26 supports.
  keyd (ids=*) grabs event26 and re-emits REL events through event20 to KWin.
  Verified: REL_X=300 appeared on event20 immediately after ydotool relative move.

COORD_MODE = 'RELATIVE'  ← only working mode on this system
"""

import cv2
import mediapipe as mp
import subprocess
import os

os.environ['YDOTOOL_SOCKET'] = '/run/user/1000/.ydotool_socket'

SCREEN_W = 4480
SCREEN_H = 1440

prev_x = [None]
prev_y = [None]

def send_cursor(nx, ny):
    """nx, ny are normalized 0.0–1.0 from MediaPipe (already flipped)."""
    tx = nx * SCREEN_W
    ty = ny * SCREEN_H

    if prev_x[0] is None:
        prev_x[0] = tx
        prev_y[0] = ty
        return

    dx = round(tx - prev_x[0])
    dy = round(ty - prev_y[0])
    prev_x[0] += dx
    prev_y[0] += dy

    print(f"  norm=({nx:.3f},{ny:.3f})  px=({int(tx)},{int(ty)})  delta=({dx:+d},{dy:+d})")
    if dx != 0 or dy != 0:
        subprocess.run(
            ['ydotool', 'mousemove', '-x', str(dx), '-y', str(dy)],
            env={**os.environ, 'YDOTOOL_SOCKET': '/run/user/1000/.ydotool_socket'},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[test_cursor] RELATIVE mode  |  screen 4480x1440  |  Q to quit")
print("[test_cursor] Move hand — cursor should follow directly")

with HandLandmarker.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read /dev/video2")
            break

        frame = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        if result.hand_landmarks:
            lm  = result.hand_landmarks[0]
            tip = lm[8]  # index fingertip
            # frame is already flipped so nx = tip.x directly (left = left)
            send_cursor(tip.x, tip.y)
            cv2.putText(frame, f"nx={tip.x:.2f} ny={tip.y:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Cursor Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
