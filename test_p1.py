#!/usr/bin/env python3
"""
Kontrol Phase 1 — Minimal test harness
Tests ONLY: cursor tracking · left pinch click · fist lock/unlock

Everything else stripped. Purpose is to measure and tune:
  - Cursor speed and feel across both monitors
  - Lag (MediaPipe inference time + pipeline overhead)
  - Jitter when hand is still
  - Camera exposure stability

Device: Logitech C920  046d:0892  serial 92E67B2F
Path:   /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_92E67B2F-video-index0
USB:    direct USB 2.0 port (480 Mbps) — supports MJPG 1080p@30, 720p@30

Run:
  cd /home/K/Storage/Projects/Kontrol
  source venv/bin/activate
  DISPLAY=:0 QT_QPA_PLATFORM=xcb \\
    YDOTOOL_SOCKET=/run/user/1000/.ydotool_socket \\
    python test_p1.py

HUD keys:
  Q          quit
  +/-        exposure +/- 25 steps (live tune without restart)
  G          gain +/- 10 (toggle)
  R          print current v4l2 readings to terminal
"""

import cv2
import mediapipe as mp
import subprocess
import time
import os
import math
from collections import deque
from pathlib import Path

# ── Hardware constants ─────────────────────────────────────────────────────────
CAM_PATH   = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_92E67B2F-video-index0"
CAM_W, CAM_H, CAM_FPS = 640, 480, 30   # Phase 1 baseline — test 1280x720 in Phase 2
SCREEN_W   = 4480
SCREEN_H   = 1440
FLIP       = True
TUNE_SH    = Path(__file__).parent / "cam-tune.sh"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
SCREEN_DIAG = math.hypot(SCREEN_W, SCREEN_H)

# ── Smoothing — tune these live via kontrol.conf edits ────────────────────────
LM_SMOOTH    = 0.30    # Stage 1: fixed-alpha pre-filter on raw landmark
VEL_SCALE    = 3.5     # Stage 2: vel_norm × this → EMA factor
SMOOTH_MIN   = 0.05    # EMA floor (heavy smooth at rest)
SMOOTH_MAX   = 0.35    # EMA ceiling (snappy at full speed)
STILL_THRESH = 0.0015  # vel_norm below this → freeze EMA (kills jitter drift)
EDGE_BOOST   = 1.8     # velocity multiplier when mapped cursor is within 10% of edge
DEADZONE_PX  = 2       # suppress moves smaller than this (px)

# ── Zone mapping ──────────────────────────────────────────────────────────────
ZONE_X_MIN, ZONE_X_MAX = 0.15, 0.85   # camera-space active rectangle
ZONE_Y_MIN, ZONE_Y_MAX = 0.10, 0.90   # maps linearly to full screen

# ── Gestures ──────────────────────────────────────────────────────────────────
PINCH_THRESH   = 0.05   # LM 4+8 normalised distance
PINCH_COOLDOWN = 0.4    # seconds between clicks
FIST_LOCK_FRAMES = 12   # frames of fist to toggle lock

# ── System ────────────────────────────────────────────────────────────────────
YDOTOOL_SOCKET = "/run/user/1000/.ydotool_socket"
os.environ["YDOTOOL_SOCKET"] = YDOTOOL_SOCKET

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode
Connections           = mp.tasks.vision.HandLandmarksConnections


# ── Helpers ───────────────────────────────────────────────────────────────────
def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def ydopopen(*args):
    subprocess.Popen(["ydotool", *[str(a) for a in args]],
                     env=os.environ,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def move_cursor(dx, dy):
    ydopopen("mousemove", "-x", dx, "-y", dy)


def left_click():
    ydopopen("click", "0xC0")


def pinch_dist(lm, a, b):
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def is_fist(lm):
    return all(lm[t].y > lm[m].y for t, m in zip([8, 12, 16, 20], [5, 9, 13, 17]))


def apply_cam_tune():
    if TUNE_SH.exists():
        subprocess.run(["bash", str(TUNE_SH)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_cam_controls():
    """Read live v4l2 control values — verifies settings are sticking."""
    try:
        raw = subprocess.check_output(
            ["v4l2-ctl", "-d", CAM_PATH,
             "--get-ctrl=auto_exposure,exposure_time_absolute,gain,power_line_frequency"],
            stderr=subprocess.DEVNULL,
        ).decode()
        vals = {}
        for line in raw.strip().splitlines():
            k, _, v = line.partition(": ")
            vals[k.strip()] = v.strip()
        return vals
    except Exception:
        return {}


def set_exposure(value):
    value = _clamp(value, 3, 2047)
    subprocess.run(["v4l2-ctl", "-d", CAM_PATH,
                    f"--set-ctrl=exposure_time_absolute={value}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return value


def draw_skeleton(frame, lm, fw, fh):
    for c in Connections.HAND_CONNECTIONS:
        ax, ay = int(lm[c.start].x * fw), int(lm[c.start].y * fh)
        bx, by = int(lm[c.end].x   * fw), int(lm[c.end].y   * fh)
        cv2.line(frame, (ax, ay), (bx, by), (60, 160, 60), 1)
    for lmk in lm:
        cv2.circle(frame, (int(lmk.x * fw), int(lmk.y * fh)), 3, (100, 200, 100), -1)


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    # Open camera
    cap = cv2.VideoCapture(CAM_PATH, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)

    # Pass 1: pre-stream tuning
    apply_cam_tune()

    # Drain 5 frames to trigger VIDIOC_STREAMON (C920 resets exposure on stream start)
    for _ in range(5):
        cap.read()

    # Pass 2: re-lock after stream-on reset
    apply_cam_tune()

    # Verify and print
    ctrl = read_cam_controls()
    exp = int(ctrl.get("exposure_time_absolute", "?") or 0)
    print(f"[P1] Camera: {CAM_W}x{CAM_H} MJPG @{CAM_FPS}fps  |  controls: {ctrl}")
    if exp != 250:
        print(f"[P1] WARNING: exposure is {exp}, expected 250 — cam-tune may have failed")
    else:
        print(f"[P1] Exposure locked at {exp} (manual, 50 Hz, gain=80) ✓")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.70,
        min_tracking_confidence=0.60,
    )

    WIN = "Kontrol P1"
    cv2.namedWindow(WIN)

    # ── State ─────────────────────────────────────────────────────────────────
    cx, cy         = float(SCREEN_W) / 2, float(SCREEN_H) / 2
    prev_tx, prev_ty   = cx, cy
    prev_sx, prev_sy   = cx, cy
    raw_x = raw_y  = 0.5
    hand_present   = False

    pinch_held     = False
    last_click_t   = 0.0
    locked         = False
    fist_count     = 0
    fist_toggled   = False

    # ── Perf tracking ─────────────────────────────────────────────────────────
    fps          = 0.0
    fps_alpha    = 0.1
    last_t       = time.time()
    infer_buf    = deque(maxlen=60)   # last 60 inference times (ms)
    last_report  = time.time()
    cur_exposure = exp

    # Live-tuneable values (keys +/-)
    live_exposure = exp

    with HandLandmarker.create_from_options(opts) as det:
        print(f"[P1] Running — Q=quit  +/-=exposure  R=print cam controls")

        while cap.isOpened():
            # ── Capture ───────────────────────────────────────────────────────
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.time()
            dt  = now - last_t;  last_t = now
            if dt > 0:
                fps = fps * (1 - fps_alpha) + (1 / dt) * fps_alpha

            if FLIP:
                frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]

            # ── MediaPipe inference — timed ───────────────────────────────────
            t0  = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            infer_ms = (time.perf_counter() - t0) * 1000
            infer_buf.append(infer_ms)

            # ── Per-frame display state ───────────────────────────────────────
            gesture    = "LOCKED" if locked else "NONE"
            v_disp     = 0.0
            smooth_disp = 0.0
            pd_disp    = 1.0

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                pd_disp = pinch_dist(lm, 4, 8)

                # Re-entry reset — no cursor jump on hand return
                if not hand_present:
                    raw_x, raw_y = lm[8].x, lm[8].y
                    nx0 = _clamp((raw_x - ZONE_X_MIN) / (ZONE_X_MAX - ZONE_X_MIN), 0.0, 1.0)
                    ny0 = _clamp((raw_y - ZONE_Y_MIN) / (ZONE_Y_MAX - ZONE_Y_MIN), 0.0, 1.0)
                    cx, cy           = nx0 * SCREEN_W, ny0 * SCREEN_H
                    prev_tx, prev_ty = cx, cy
                    prev_sx, prev_sy = cx, cy
                hand_present = True

                # ── Fist → lock/unlock (always active, bypasses locked state) ─
                fist_now = is_fist(lm)
                if fist_now:
                    fist_count += 1
                    if fist_count >= FIST_LOCK_FRAMES and not fist_toggled:
                        locked = not locked
                        fist_toggled = True
                else:
                    fist_count   = 0
                    fist_toggled = False

                if locked:
                    gesture = f"LOCKED  fist {fist_count}/{FIST_LOCK_FRAMES}"

                else:
                    # ── Stage 1: fixed-alpha landmark pre-filter ───────────────
                    raw_x = raw_x * (1 - LM_SMOOTH) + lm[8].x * LM_SMOOTH
                    raw_y = raw_y * (1 - LM_SMOOTH) + lm[8].y * LM_SMOOTH

                    # ── Zone map → screen pixels ──────────────────────────────
                    nx = _clamp((raw_x - ZONE_X_MIN) / (ZONE_X_MAX - ZONE_X_MIN), 0.0, 1.0)
                    ny = _clamp((raw_y - ZONE_Y_MIN) / (ZONE_Y_MAX - ZONE_Y_MIN), 0.0, 1.0)
                    tx = nx * SCREEN_W
                    ty = ny * SCREEN_H

                    # ── Stage 2: velocity-adaptive EMA with stillness gate ─────
                    v_norm       = math.hypot(tx - prev_tx, ty - prev_ty) / SCREEN_DIAG
                    prev_tx, prev_ty = tx, ty
                    v_disp       = v_norm

                    if v_norm > STILL_THRESH:
                        near_edge = nx < 0.1 or nx > 0.9 or ny < 0.1 or ny > 0.9
                        boost     = EDGE_BOOST if near_edge else 1.0
                        smooth    = _clamp(v_norm * VEL_SCALE * boost, SMOOTH_MIN, SMOOTH_MAX)
                        smooth_disp = smooth
                        cx = cx * (1 - smooth) + tx * smooth
                        cy = cy * (1 - smooth) + ty * smooth

                    dx = round(cx - prev_sx)
                    dy = round(cy - prev_sy)
                    if max(abs(dx), abs(dy)) >= DEADZONE_PX:
                        move_cursor(dx, dy)
                        prev_sx += dx
                        prev_sy += dy

                    # ── Left pinch → click ────────────────────────────────────
                    if pd_disp < PINCH_THRESH:
                        if not pinch_held and (now - last_click_t) > PINCH_COOLDOWN:
                            left_click()
                            last_click_t = now
                        pinch_held = True
                        gesture    = f"CLICK  d={pd_disp:.3f}"
                    else:
                        pinch_held = False
                        if not fist_now:
                            gesture = "CURSOR"

                draw_skeleton(frame, lm, fw, fh)

            else:
                hand_present = False
                fist_count   = 0
                fist_toggled = False
                gesture      = "NONE"

            # ── HUD ───────────────────────────────────────────────────────────
            avg_infer = sum(infer_buf) / len(infer_buf) if infer_buf else 0.0
            border    = (0, 40, 200) if locked else (30, 190, 50)
            cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), border, 3)

            hud_lines = [
                f"FPS      {fps:5.1f}",
                f"Infer    {infer_ms:5.1f} ms  avg {avg_infer:4.1f}",
                f"Gesture  {gesture}",
                f"Pinch    {pd_disp:.3f}  (thr {PINCH_THRESH})",
                f"v_norm   {v_disp:.4f}  (still<{STILL_THRESH})",
                f"Smooth   {smooth_disp:.3f}  [{SMOOTH_MIN}-{SMOOTH_MAX}]",
                f"Fist     {fist_count}/{FIST_LOCK_FRAMES}",
                f"Exposure {live_exposure}",
            ]
            for i, txt in enumerate(hud_lines):
                cv2.putText(frame, txt, (8, 18 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 215, 215), 1)

            lbl = "LOCKED" if locked else "TRACKING"
            cv2.putText(frame, lbl, (fw // 2 - 55, fh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, border, 2)

            cv2.imshow(WIN, frame)

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("+") or key == ord("="):
                live_exposure = set_exposure(live_exposure + 25)
                print(f"[P1] Exposure → {live_exposure}")
            elif key == ord("-"):
                live_exposure = set_exposure(live_exposure - 25)
                print(f"[P1] Exposure → {live_exposure}")
            elif key == ord("r") or key == ord("R"):
                ctrl = read_cam_controls()
                print(f"[P1] Live cam controls: {ctrl}")

            # ── Periodic terminal report ──────────────────────────────────────
            if now - last_report >= 5.0:
                it = list(infer_buf)
                print(
                    f"[P1 t={now:.0f}]  "
                    f"FPS={fps:.1f}  "
                    f"infer min={min(it):.1f} avg={avg_infer:.1f} max={max(it):.1f} ms  "
                    f"v={v_disp:.4f}  smooth={smooth_disp:.3f}  "
                    f"exposure={live_exposure}"
                )
                last_report = now

    cap.release()
    cv2.destroyAllWindows()
    print("[P1] Done.")


if __name__ == "__main__":
    run()
