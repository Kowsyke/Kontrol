#!/usr/bin/env python3
"""
Kontrol v0.2 — Hand gesture mouse control
MediaPipe Tasks API (0.10+) → ydotool (no pynput, Wayland-safe)

Gestures:
  Index fingertip (LM 8)                      → cursor (velocity-adaptive EMA)
  Pinch index+thumb  (LM 8+4)                 → left click
  Pinch ring+thumb   (LM 16+4)                → right click
  Index+middle extended, ring+pinky curled    → scroll (up/down with speed scaling)
  Wrist flick (LM 0 velocity)                 → KDE window tiling (Meta+Arrow)
  Fist (all 4 tips below MCP joints)          → toggle tracking lock

Config is read from kontrol.conf (INI format) in the same directory.
If kontrol.conf is missing it is created with default values.
All tuning knobs live in kontrol.conf — no need to edit this file.

Press Q in preview window to quit.
"""

import cv2
import mediapipe as mp
import subprocess
import time
import os
import math
import wave
import struct
import tempfile
import configparser
from collections import deque
from pathlib import Path

# ── Config file ───────────────────────────────────────────────────────────────
CONF_PATH  = Path(__file__).parent / "kontrol.conf"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

_DEFAULTS = {
    "screen": {
        "width":  "1920",
        "height": "1080",
    },
    "camera": {
        "id":   "0",
        "flip": "true",
    },
    "smoothing": {
        # Velocity-adaptive EMA: factor = clamp(vel_norm * scale, min, max)
        # vel_norm = pixel_delta / screen_diagonal  (0..~0.15 for typical hand motion)
        "min_smooth":     "0.08",
        "max_smooth":     "0.35",
        "velocity_scale": "2.5",
    },
    "gestures": {
        "pinch_threshold":    "0.05",   # normalized landmark distance → click
        "pinch_cooldown":     "0.4",    # seconds between clicks (per button)
        "scroll_deadzone":    "0.012",  # min norm y-delta per frame before scrolling
        "scroll_speed":       "8.0",    # ticks = max(1, int(abs(dy) * scroll_speed))
        "flick_min_velocity": "450",    # px/s wrist velocity to register flick
        "flick_window_ms":    "120",    # ms window for velocity measurement
        "flick_cooldown":     "0.8",    # seconds between consecutive tiling gestures
        "fist_hold_frames":   "6",      # frames fist must be held to toggle lock
    },
    "system": {
        "ydotool_socket": "/run/user/1000/.ydotool_socket",
        "abs_scale_x":    "1.0",   # set to 32767/1920 if cursor maps to wrong spot
        "abs_scale_y":    "1.0",
    },
}


def load_config() -> configparser.ConfigParser:
    """
    Load kontrol.conf, overlaying user values on top of built-in defaults.
    If the file does not exist, write the defaults and return them.
    """
    cfg = configparser.ConfigParser()
    cfg.read_dict(_DEFAULTS)        # populate defaults first
    if not CONF_PATH.exists():
        with open(CONF_PATH, "w") as f:
            cfg.write(f)
        print(f"[kontrol] Created default config: {CONF_PATH}")
    else:
        cfg.read(CONF_PATH)         # overlay user values
    return cfg


_cfg = load_config()

SCREEN_W         = _cfg.getint("screen",    "width")
SCREEN_H         = _cfg.getint("screen",    "height")
CAM_ID           = _cfg.getint("camera",    "id")
FLIP             = _cfg.getboolean("camera","flip")
MIN_SMOOTH       = _cfg.getfloat("smoothing", "min_smooth")
MAX_SMOOTH       = _cfg.getfloat("smoothing", "max_smooth")
VELOCITY_SCALE   = _cfg.getfloat("smoothing", "velocity_scale")
PINCH_THRESHOLD  = _cfg.getfloat("gestures", "pinch_threshold")
PINCH_COOLDOWN   = _cfg.getfloat("gestures", "pinch_cooldown")
SCROLL_DEADZONE  = _cfg.getfloat("gestures", "scroll_deadzone")
SCROLL_SPEED     = _cfg.getfloat("gestures", "scroll_speed")
FLICK_MIN_VEL    = _cfg.getfloat("gestures", "flick_min_velocity")
FLICK_WINDOW_MS  = _cfg.getfloat("gestures", "flick_window_ms")
FLICK_COOLDOWN   = _cfg.getfloat("gestures", "flick_cooldown")
FIST_HOLD_FRAMES = _cfg.getint("gestures",   "fist_hold_frames")
YDOTOOL_SOCKET   = _cfg.get("system",        "ydotool_socket")
ABS_SCALE_X      = _cfg.getfloat("system",   "abs_scale_x")
ABS_SCALE_Y      = _cfg.getfloat("system",   "abs_scale_y")

SCREEN_DIAG = math.hypot(SCREEN_W, SCREEN_H)
# ─────────────────────────────────────────────────────────────────────────────

# Raw Linux keycodes for KDE tiling shortcuts (input-event-codes.h).
# KEY_LEFTMETA=125, KEY_LEFT=105, KEY_RIGHT=106, KEY_UP=103, KEY_DOWN=108
# ydotool key requires raw keycodes — NOT X11 keysym strings.
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
    """Fingertip above PIP joint = extended (y increases downward in MediaPipe)."""
    return lm[tip].y < lm[pip].y

def is_scroll_pose(lm) -> bool:
    """Index + middle extended above PIP; ring + pinky curled."""
    return (is_finger_extended(lm, 8,  6)  and
            is_finger_extended(lm, 12, 10) and
            not is_finger_extended(lm, 16, 14) and
            not is_finger_extended(lm, 20, 18))

def is_fist(lm) -> bool:
    """All 4 fingertip y-coords exceed their MCP joints (fully curled)."""
    tips = [8, 12, 16, 20]
    mcps = [5,  9, 13, 17]
    return all(lm[t].y > lm[m].y for t, m in zip(tips, mcps))

def check_flick(history: deque) -> str | None:
    """
    Measure wrist velocity over the last FLICK_WINDOW_MS milliseconds.
    Returns dominant-axis direction string or None.
    """
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


# ── Startup sound ─────────────────────────────────────────────────────────────
def play_startup_sound():
    """880 Hz double-beep via aplay. stdlib only — no extra deps."""
    rate, freq, amp = 22050, 880, 28000
    dur, gap = 0.12, 0.04

    def burst(duration: float) -> bytes:
        n = int(rate * duration)
        out = bytearray(n * 2)
        for i in range(n):
            t = i / rate
            env = min(t / (duration * 0.1), 1.0, (duration - t) / (duration * 0.1))
            struct.pack_into("<h", out, i * 2,
                             int(amp * env * math.sin(2.0 * math.pi * freq * t)))
        return bytes(out)

    pcm = burst(dur) + bytes(int(rate * gap) * 2) + burst(dur)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1);  wf.setsampwidth(2);  wf.setframerate(rate)
        wf.writeframes(pcm)
    tmp.close()
    subprocess.Popen(["aplay", "-q", tmp.name],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_skeleton(frame, lm, fw: int, fh: int):
    for c in HandLandmarksConnections.HAND_CONNECTIONS:
        ax, ay = int(lm[c.start].x * fw), int(lm[c.start].y * fh)
        bx, by = int(lm[c.end].x   * fw), int(lm[c.end].y   * fh)
        cv2.line(frame, (ax, ay), (bx, by), (60, 160, 60), 1)
    for lmk in lm:
        cv2.circle(frame, (int(lmk.x * fw), int(lmk.y * fh)), 3, (100, 200, 100), -1)


def draw_hud(frame, locked: bool, gesture: str, fps: float,
             pd_L: float, pd_R: float):
    h, w = frame.shape[:2]
    border_color = (0, 40, 200) if locked else (30, 190, 50)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 3)
    mode_txt = "LOCKED" if locked else "TRACKING"
    cv2.putText(frame, mode_txt, (w // 2 - 58, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)
    for i, txt in enumerate([
        f"FPS     {fps:5.1f}",
        f"Mode    {mode_txt}",
        f"Gesture {gesture}",
        f"L-pinch {pd_L:.3f}",
        f"R-pinch {pd_R:.3f}",
    ]):
        cv2.putText(frame, txt, (8, 18 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)


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

    play_startup_sound()

    cx, cy           = float(SCREEN_W) / 2, float(SCREEN_H) / 2
    prev_tx, prev_ty = cx, cy
    pinch_held_L  = False;  last_click_L = 0.0
    pinch_held_R  = False;  last_click_R = 0.0
    scroll_ref_y  = None
    wrist_history = deque(maxlen=30)
    last_flick_t  = 0.0

    locked                 = False
    fist_hold_count        = 0
    fist_toggled_this_hold = False

    active_gesture = "NONE"
    pd_L_hud = pd_R_hud = 1.0
    fps = 0.0;  frame_count = 0;  fps_timer = time.time()

    print(f"Kontrol v0.2  {SCREEN_W}x{SCREEN_H}"
          f"  smooth[{MIN_SMOOTH}-{MAX_SMOOTH}]x{VELOCITY_SCALE}"
          f"  pinch={PINCH_THRESHOLD}  config={CONF_PATH.name}")
    print("  Fist = toggle lock   Q in preview = quit")

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

                wrist_history.append((now,
                                      lm[0].x * SCREEN_W,
                                      lm[0].y * SCREEN_H))

                pd_L_hud = pinch_dist(lm, 4, 8)
                pd_R_hud = pinch_dist(lm, 4, 16)

                fist_now = is_fist(lm)
                if fist_now:
                    fist_hold_count += 1
                else:
                    fist_hold_count = 0
                    fist_toggled_this_hold = False

                if fist_hold_count >= FIST_HOLD_FRAMES and not fist_toggled_this_hold:
                    locked = not locked
                    fist_toggled_this_hold = True

                if locked:
                    active_gesture = "FIST-LOCKED" if fist_now else "LOCKED"

                else:
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
                                    scroll_up(ticks)
                                    active_gesture = f"SCROLL UP x{ticks}"
                                else:
                                    scroll_down(ticks)
                                    active_gesture = f"SCROLL DN x{ticks}"
                            else:
                                active_gesture = "SCROLL"

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
                            active_gesture = f"FLICK {flick.upper()}"
                            wrist_history.clear()
                        else:
                            tx = lm[8].x * SCREEN_W
                            ty = lm[8].y * SCREEN_H

                            vel_norm = math.hypot(tx - prev_tx, ty - prev_ty) / SCREEN_DIAG
                            smooth   = max(MIN_SMOOTH, min(MAX_SMOOTH,
                                                           vel_norm * VELOCITY_SCALE))
                            cx = cx * (1.0 - smooth) + tx * smooth
                            cy = cy * (1.0 - smooth) + ty * smooth
                            prev_tx, prev_ty = tx, ty
                            move_cursor(cx, cy)

                            if pd_R_hud < PINCH_THRESHOLD:
                                if not pinch_held_R and (now - last_click_R) > PINCH_COOLDOWN:
                                    right_click(); last_click_R = now
                                pinch_held_R = True
                                active_gesture = f"RIGHT CLICK d={pd_R_hud:.3f}"
                            else:
                                pinch_held_R = False

                            if pd_L_hud < PINCH_THRESHOLD and not (pd_R_hud < PINCH_THRESHOLD):
                                if not pinch_held_L and (now - last_click_L) > PINCH_COOLDOWN:
                                    left_click(); last_click_L = now
                                pinch_held_L = True
                                active_gesture = f"LEFT CLICK  d={pd_L_hud:.3f}"
                            else:
                                pinch_held_L = False

                            if not (pd_L_hud < PINCH_THRESHOLD) and \
                               not (pd_R_hud < PINCH_THRESHOLD):
                                active_gesture = "CURSOR"

                        draw_skeleton(frame, lm, fw_px, fh_px)
            else:
                fist_hold_count = 0
                fist_toggled_this_hold = False
                scroll_ref_y = None
                active_gesture = "NONE"

            draw_hud(frame, locked, active_gesture, fps, pd_L_hud, pd_R_hud)
            cv2.imshow("Kontrol v0.2", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit."); break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
