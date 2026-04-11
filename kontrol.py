#!/usr/bin/env python3
"""
Kontrol v0.2-dev — Hand gesture mouse control
MediaPipe Tasks API (0.10+) → ydotool (no pynput, Wayland-safe)

Controls:
  Index fingertip (LM 8)                        → cursor position
  Pinch index+thumb  (LM 8+4)                   → left click
  Pinch ring+thumb   (LM 16+4)                  → right click
  Index+middle extended, hand moves up/down     → scroll
  Wrist flick (LM 0 velocity > threshold)       → KDE window tiling
  Fist (all tips below MCP joints)              → toggle tracking lock  ← NEW
    LOCKED   = red border, cursor frozen
    TRACKING = green border, normal operation

Tuning knobs at top of file.
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
FLICK_MIN_VEL    = 450
FLICK_WINDOW_MS  = 120
FLICK_COOLDOWN   = 0.8
FIST_HOLD_FRAMES = 6      # frames fist must be held continuously to toggle lock
CAM_ID           = 0
FLIP             = True
ABS_SCALE_X      = 1.0
ABS_SCALE_Y      = 1.0
YDOTOOL_SOCKET   = "/run/user/1000/.ydotool_socket"
MODEL_PATH       = Path(__file__).parent / "hand_landmarker.task"
# ─────────────────────────────────────────────────────────────────────────────

_TILING_KEYS = {
    "right": ("125:1", "106:1", "106:0", "125:0"),
    "left":  ("125:1", "105:1", "105:0", "125:0"),
    "up":    ("125:1", "103:1", "103:0", "125:0"),
    "down":  ("125:1", "108:1", "108:0", "125:0"),
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

def left_click():    ydocall("click", "0xC0")
def right_click():   ydocall("click", "0xC1")

def scroll_up(ticks: int = 1):
    ydocall("mousemove", "--wheel", "-y", str(-ticks))

def scroll_down(ticks: int = 1):
    ydocall("mousemove", "--wheel", "-y", str(ticks))

def tiling_key(direction: str):
    seq = _TILING_KEYS.get(direction)
    if seq:
        ydocall("key", *seq)


# ── Landmark math ─────────────────────────────────────────────────────────────
def pinch_dist(lm, a: int, b: int) -> float:
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

def is_finger_extended(lm, tip: int, pip: int) -> bool:
    return lm[tip].y < lm[pip].y

def is_scroll_pose(lm) -> bool:
    return (is_finger_extended(lm, 8,  6)  and
            is_finger_extended(lm, 12, 10) and
            not is_finger_extended(lm, 16, 14) and
            not is_finger_extended(lm, 20, 18))

def is_fist(lm) -> bool:
    """
    All 4 fingertips (8,12,16,20) are below their MCP base joints (5,9,13,17).
    In MediaPipe: y increases downward, so tip.y > mcp.y means tip is below mcp
    (finger is curled).  Thumb is excluded — a closed fist often leaves the
    thumb partially extended.
    """
    tips = [8, 12, 16, 20]
    mcps = [5,  9, 13, 17]
    return all(lm[t].y > lm[m].y for t, m in zip(tips, mcps))

def check_flick(history: deque) -> str | None:
    if len(history) < 2:
        return None
    t_now, x_now, y_now = history[-1]
    t_cutoff = t_now - FLICK_WINDOW_MS / 1000.0
    oldest = next((e for e in history if e[0] >= t_cutoff), None)
    if oldest is None or oldest is history[-1]:
        return None
    t0, x0, y0 = oldest
    dt = t_now - t0
    if dt < 0.001:
        return None
    vx, vy = (x_now - x0) / dt, (y_now - y0) / dt
    if math.hypot(vx, vy) < FLICK_MIN_VEL:
        return None
    return ("right" if vx > 0 else "left") if abs(vx) >= abs(vy) else ("down" if vy > 0 else "up")


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_lock_overlay(frame, locked: bool, gesture_label: str):
    """Border + large mode label to make lock state immediately obvious."""
    h, w = frame.shape[:2]
    color = (0, 40, 200) if locked else (30, 190, 50)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)

    label = "LOCKED" if locked else "TRACKING"
    cv2.putText(frame, label, (w // 2 - 58, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, gesture_label, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 200, 80), 2)


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
    wrist_history = deque(maxlen=30)
    last_flick_t  = 0.0

    locked                 = False
    fist_hold_count        = 0
    fist_toggled_this_hold = False  # prevents toggling twice per single held fist

    fps = 0.0;  frame_count = 0;  fps_timer = time.time()

    print("Kontrol running — fist to toggle lock — press Q in preview to quit")

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

                wrist_history.append((now,
                                      lm[0].x * SCREEN_W,
                                      lm[0].y * SCREEN_H))

                # ── Fist detection runs always (even when locked) ─────────
                fist_now = is_fist(lm)
                if fist_now:
                    fist_hold_count += 1
                else:
                    fist_hold_count = 0
                    fist_toggled_this_hold = False   # arm for next fist hold

                # Toggle on the first frame we've held long enough (once per hold)
                if fist_hold_count >= FIST_HOLD_FRAMES and not fist_toggled_this_hold:
                    locked = not locked
                    fist_toggled_this_hold = True

                if locked:
                    gesture_label = "FIST-LOCKED" if fist_now else "LOCKED"
                    # Cursor is intentionally frozen — no move_cursor() call

                else:
                    # ── All gesture branches (only when tracking) ─────────
                    pd_L = pinch_dist(lm, 4, 8)
                    pd_R = pinch_dist(lm, 4, 16)

                    if is_scroll_pose(lm):
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

                        flick = check_flick(wrist_history)
                        if flick and (now - last_flick_t) > FLICK_COOLDOWN:
                            tiling_key(flick)
                            last_flick_t = now
                            gesture_label = f"FLICK {flick.upper()}"
                            wrist_history.clear()
                        else:
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

                        # Skeleton (only in tracking mode)
                        for c in HandLandmarksConnections.HAND_CONNECTIONS:
                            ax,ay = int(lm[c.start].x*fw_px), int(lm[c.start].y*fh_px)
                            bx,by = int(lm[c.end].x  *fw_px), int(lm[c.end].y  *fh_px)
                            cv2.line(frame, (ax,ay), (bx,by), (60,160,60), 1)
                        for lmk in lm:
                            cv2.circle(frame, (int(lmk.x*fw_px), int(lmk.y*fh_px)),
                                       3, (100,200,100), -1)
            else:
                fist_hold_count = 0
                fist_toggled_this_hold = False
                scroll_ref_y = None
                gesture_label = "NONE"

            draw_lock_overlay(frame, locked, gesture_label)
            cv2.putText(frame, f"FPS {fps:.0f}", (fw_px - 90, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Kontrol", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit."); break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
