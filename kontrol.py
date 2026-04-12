#!/usr/bin/env python3
"""
Kontrol v0.5 — Hand gesture mouse control  ("Stark Pass")
MediaPipe Tasks API (0.10+) → ydotool (no pynput, Wayland-safe)

Gesture priority order:
  1. Fist LONG   (>= fist_lock_frames, held)                      → toggle tracking lock
  2. Fist MEDIUM (>= fist_minimize_frames, released < lock_frames) → minimize window
  3. Wrist rotate (fingers pointing down, LM12 y > LM0 y + 0.10)  → KDE task overview
  4. Index+middle extended, ring+pinky curled                       → scroll (velocity-scaled)
  5. Wrist flick (LM 0 axis-pure velocity)                         → KDE window tiling
  6. Ring+thumb pinch (LM 16+4)                                    → right click
  7. Index+thumb hold+move (LM 8+4)                               → drag (mousedown/mouseup)
  8. Index+thumb quick tap                                         → left click
  9. Index fingertip (LM 8) — zone-mapped, double EMA             → cursor

v0.5 additions:
  - Zone-based hand mapping: [zone_x_min..zone_x_max] → full screen + edge_boost
  - Double EMA: landmark pre-filter (fixed alpha) → velocity-adaptive screen EMA
  - Strict gesture priority chain (clean if/elif)
  - In-frame ✕ / — buttons (setMouseCallback + wmctrl)
  - Single-hand enforcement: detection/presence confidence raised to 0.70/0.70
  - Performance: Popen fire-and-forget, 20 fps cap, rolling FPS EMA
  - Hand visibility warning: directional arrow when index tip enters outer 15% of frame
  - Smooth re-entry: no cursor jump on hand return (state reset on first frame back)

Config is read from kontrol.conf (INI format) in the same directory.
If kontrol.conf is missing it is created with default values.

In-frame controls: click ✕ (top-right) to quit, — to minimize active window.
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

# ── Config ────────────────────────────────────────────────────────────────────
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
    "mapping": {
        # Active zone in camera-normalized [0,1] space.
        # Hand must stay inside this rectangle; it maps to the full screen.
        "zone_x_min": "0.15",
        "zone_x_max": "0.85",
        "zone_y_min": "0.10",
        "zone_y_max": "0.90",
        # Velocity multiplier applied when the mapped cursor is within 10% of a
        # screen edge — lets the user snap to corners without leaving the zone.
        "edge_boost": "1.8",
    },
    "smoothing": {
        # Stage 1 — fixed-alpha pre-filter applied directly to the raw landmark
        # before zone mapping.  Removes per-frame detector jitter.
        "landmark_smooth":    "0.4",
        # Stage 2 — velocity-adaptive screen-space EMA.
        # smooth = clamp(vel_norm * velocity_scale * boost, min_smooth, max_smooth)
        "min_smooth":         "0.05",
        "max_smooth":         "0.18",
        "velocity_scale":     "2.0",
        "cursor_deadzone_px": "2",     # skip moves ≤ this to kill residual jitter
    },
    "gestures": {
        "pinch_threshold":      "0.05",
        "pinch_cooldown":       "0.4",
        "scroll_deadzone":      "0.012",
        "scroll_speed":         "8.0",
        "flick_min_velocity":   "2.0",   # norm/s in [0,1]² space
        "flick_window_ms":      "120",
        "flick_axis_ratio":     "3.0",   # dom/minor — filters diagonal cursor sweeps
        "flick_cooldown":       "0.8",
        "fist_minimize_frames": "6",
        "fist_lock_frames":     "12",
        "taskview_hold_frames": "8",
        "taskview_cooldown":    "1.5",
    },
    "system": {
        "ydotool_socket": "/run/user/1000/.ydotool_socket",
        "abs_scale_x":    "1.0",
        "abs_scale_y":    "1.0",
    },
}


def load_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read_dict(_DEFAULTS)
    if not CONF_PATH.exists():
        with open(CONF_PATH, "w") as f:
            cfg.write(f)
        print(f"[kontrol] Created default config: {CONF_PATH}")
    else:
        cfg.read(CONF_PATH)
    return cfg


_cfg = load_config()

SCREEN_W           = _cfg.getint("screen",    "width")
SCREEN_H           = _cfg.getint("screen",    "height")
CAM_ID             = _cfg.getint("camera",    "id")
FLIP               = _cfg.getboolean("camera","flip")
ZONE_X_MIN         = _cfg.getfloat("mapping", "zone_x_min")
ZONE_X_MAX         = _cfg.getfloat("mapping", "zone_x_max")
ZONE_Y_MIN         = _cfg.getfloat("mapping", "zone_y_min")
ZONE_Y_MAX         = _cfg.getfloat("mapping", "zone_y_max")
EDGE_BOOST         = _cfg.getfloat("mapping", "edge_boost")
LANDMARK_SMOOTH    = _cfg.getfloat("smoothing", "landmark_smooth")
MIN_SMOOTH         = _cfg.getfloat("smoothing", "min_smooth")
MAX_SMOOTH         = _cfg.getfloat("smoothing", "max_smooth")
VELOCITY_SCALE     = _cfg.getfloat("smoothing", "velocity_scale")
CURSOR_DEADZONE_PX = _cfg.getint("smoothing",  "cursor_deadzone_px")
PINCH_THRESHOLD    = _cfg.getfloat("gestures", "pinch_threshold")
PINCH_COOLDOWN     = _cfg.getfloat("gestures", "pinch_cooldown")
SCROLL_DEADZONE    = _cfg.getfloat("gestures", "scroll_deadzone")
SCROLL_SPEED       = _cfg.getfloat("gestures", "scroll_speed")
FLICK_MIN_VEL      = _cfg.getfloat("gestures", "flick_min_velocity")
FLICK_WINDOW_MS    = _cfg.getfloat("gestures", "flick_window_ms")
FLICK_AXIS_RATIO   = _cfg.getfloat("gestures", "flick_axis_ratio")
FLICK_COOLDOWN     = _cfg.getfloat("gestures", "flick_cooldown")
FIST_MINIMIZE_FRAMES = _cfg.getint("gestures",  "fist_minimize_frames")
FIST_LOCK_FRAMES     = _cfg.getint("gestures",  "fist_lock_frames")
TASKVIEW_HOLD_FRAMES = _cfg.getint("gestures",  "taskview_hold_frames")
TASKVIEW_COOLDOWN    = _cfg.getfloat("gestures","taskview_cooldown")
YDOTOOL_SOCKET       = _cfg.get("system",       "ydotool_socket")
ABS_SCALE_X        = _cfg.getfloat("system",   "abs_scale_x")
ABS_SCALE_Y        = _cfg.getfloat("system",   "abs_scale_y")

SCREEN_DIAG = math.hypot(SCREEN_W, SCREEN_H)

# Raw Linux keycodes (input-event-codes.h).
# KEY_LEFTMETA=125, KEY_LEFT=105, KEY_RIGHT=106, KEY_UP=103, KEY_DOWN=108
_TILING_KEYS = {
    "right": ("125:1", "106:1", "106:0", "125:0"),
    "left":  ("125:1", "105:1", "105:0", "125:0"),
    "up":    ("125:1", "103:1", "103:0", "125:0"),
    "down":  ("125:1", "108:1", "108:0", "125:0"),
}

os.environ["YDOTOOL_SOCKET"] = YDOTOOL_SOCKET

BaseOptions              = mp.tasks.BaseOptions
HandLandmarker           = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions    = mp.tasks.vision.HandLandmarkerOptions
RunningMode              = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ── ydotool helpers ───────────────────────────────────────────────────────────
def ydocall(*args, blocking: bool = False):
    """
    Issue a ydotool command.
    blocking=False (default) — Popen, fire-and-forget (cursor/click calls).
    blocking=True            — subprocess.run, wait for completion (key sequences).
    """
    cmd = ["ydotool", *[str(a) for a in args]]
    if blocking:
        subprocess.run(cmd, env=os.environ,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(cmd, env=os.environ,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def move_cursor(dx: int, dy: int):
    """Send relative mouse delta. ydotoold has no EV_ABS — only REL works."""
    ydocall("mousemove", "-x", dx, "-y", dy)

def mouse_down():   ydocall("click", "0x40")
def mouse_up():     ydocall("click", "0x80")
def left_click():   ydocall("click", "0xC0")
def right_click():  ydocall("click", "0xC1")

def scroll_up(ticks: int = 1):
    ydocall("mousemove", "--wheel", "-x", "0", "-y", str(-ticks))

def scroll_down(ticks: int = 1):
    ydocall("mousemove", "--wheel", "-x", "0", "-y", str(ticks))

def tiling_key(direction: str):
    seq = _TILING_KEYS.get(direction)
    if seq:
        ydocall("key", *seq, blocking=True)


# ── Landmark math ─────────────────────────────────────────────────────────────
def pinch_dist(lm, a: int, b: int) -> float:
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

def is_finger_extended(lm, tip: int, pip: int) -> bool:
    """Tip above PIP = extended (MediaPipe y increases downward)."""
    return lm[tip].y < lm[pip].y

def is_scroll_pose(lm) -> bool:
    """Index + middle extended; ring + pinky curled below their MCP joint."""
    return (is_finger_extended(lm, 8,  6)  and
            is_finger_extended(lm, 12, 10) and
            not is_finger_extended(lm, 16, 13) and   # ring tip vs ring MCP (more lenient than PIP)
            not is_finger_extended(lm, 20, 17))      # pinky tip vs pinky MCP

def is_fist(lm) -> bool:
    """All four fingertips below their MCP joints."""
    tips = [8, 12, 16, 20]
    mcps = [5,  9, 13, 17]
    return all(lm[t].y > lm[m].y for t, m in zip(tips, mcps))

def is_wrist_rotated(lm) -> bool:
    """Hand flipped so fingers point down: LM12.y > LM0.y + 0.10."""
    return lm[12].y > lm[0].y + 0.10


def check_flick(history: deque) -> str | None:
    """
    Measure wrist (LM 0) velocity over the last FLICK_WINDOW_MS milliseconds.
    Returns dominant-axis direction string or None.

    Two gates suppress false fires from fast cursor sweeps:
      1. Speed > FLICK_MIN_VEL normalized units/s.
      2. Motion must be axis-pure: dominant/minor >= FLICK_AXIS_RATIO.
    History is stored in normalized [0,1] coords to avoid 4480:1440
    aspect-ratio distortion skewing the axis-purity test.
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
    dom   = max(abs(vx), abs(vy))
    minor = min(abs(vx), abs(vy))
    if minor > 0 and dom / minor < FLICK_AXIS_RATIO:
        return None
    return ("right" if vx > 0 else "left") if abs(vx) >= abs(vy) else ("down" if vy > 0 else "up")


# ── Startup sound ─────────────────────────────────────────────────────────────
def play_startup_sound():
    """Triple ascending tone — 880 → 1047 → 1319 Hz (v0.5 signature, vs v0.4 double-beep)."""
    rate, amp       = 22050, 28000
    dur, gap        = 0.10, 0.04
    freqs           = (880, 1047, 1319)

    def burst(frequency: float, duration: float) -> bytes:
        n   = int(rate * duration)
        out = bytearray(n * 2)
        for i in range(n):
            t   = i / rate
            env = min(t / (duration * 0.1), 1.0, (duration - t) / (duration * 0.1))
            struct.pack_into("<h", out, i * 2,
                             int(amp * env * math.sin(2.0 * math.pi * frequency * t)))
        return bytes(out)

    silence = bytes(int(rate * gap) * 2)
    pcm     = silence.join(burst(f, dur) for f in freqs)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1);  wf.setsampwidth(2);  wf.setframerate(rate)
        wf.writeframes(pcm)
    tmp.close()
    subprocess.Popen(["aplay", "-q", tmp.name],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── In-frame button state ─────────────────────────────────────────────────────
_quit_flag              = [False]
_btn_close: tuple | None = None   # (x1, y1, x2, y2) in frame pixels
_btn_min:   tuple | None = None


def _mouse_cb(event, x, y, flags, param):
    """OpenCV mouse callback — handles ✕ (quit) and — (minimize) buttons."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if _btn_close and _btn_close[0] <= x <= _btn_close[2] and _btn_close[1] <= y <= _btn_close[3]:
        _quit_flag[0] = True
    elif _btn_min and _btn_min[0] <= x <= _btn_min[2] and _btn_min[1] <= y <= _btn_min[3]:
        subprocess.Popen(
            ["wmctrl", "-r", ":ACTIVE:", "-b", "add,hidden"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def draw_buttons(frame):
    """Draw ✕ and — buttons top-right of frame; update _btn_close / _btn_min rects."""
    global _btn_close, _btn_min
    h, w = frame.shape[:2]
    pad, bw, bh = 5, 24, 18
    # Close ✕
    cx1, cy1, cx2, cy2 = w - pad - bw, pad, w - pad, pad + bh
    _btn_close = (cx1, cy1, cx2, cy2)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (40, 40, 160), -1)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (160, 160, 160), 1)
    cv2.putText(frame, "X", (cx1 + 6, cy2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    # Minimize —
    mx1, my1, mx2, my2 = cx1 - pad - bw, pad, cx1 - pad, pad + bh
    _btn_min = (mx1, my1, mx2, my2)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (30, 100, 30), -1)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (160, 160, 160), 1)
    cv2.putText(frame, "-", (mx1 + 8, my2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_skeleton(frame, lm, fw: int, fh: int):
    for c in HandLandmarksConnections.HAND_CONNECTIONS:
        ax, ay = int(lm[c.start].x * fw), int(lm[c.start].y * fh)
        bx, by = int(lm[c.end].x   * fw), int(lm[c.end].y   * fh)
        cv2.line(frame, (ax, ay), (bx, by), (60, 160, 60), 1)
    for lmk in lm:
        cv2.circle(frame, (int(lmk.x * fw), int(lmk.y * fh)), 3, (100, 200, 100), -1)


def draw_visibility_warning(frame, lm, fw: int, fh: int):
    """
    Draws a directional arrow toward the frame centre when the index tip
    enters the outer 15% of the camera frame.  Cues the user to move
    their hand back into the active tracking zone.
    """
    x, y  = lm[8].x, lm[8].y
    OUTER = 0.15
    if not (x < OUTER or x > 1.0 - OUTER or y < OUTER or y > 1.0 - OUTER):
        return
    px,  py  = int(x * fw), int(y * fh)
    cfx, cfy = fw // 2, fh // 2
    vx,  vy  = cfx - px, cfy - py
    mag      = math.hypot(vx, vy) or 1.0
    tip_x    = int(px + vx / mag * 45)
    tip_y    = int(py + vy / mag * 45)
    cv2.arrowedLine(frame, (px, py), (tip_x, tip_y), (0, 80, 255), 2, tipLength=0.35)
    cv2.putText(frame, "MOVE IN", (max(px - 28, 2), max(py - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 80, 255), 1)


def draw_hud(frame, locked: bool, gesture: str, fps: float,
             pd_L: float, pd_R: float,
             fist_count: int = 0,
             flash_taskview: bool = False,
             flash_minimize: bool = False):
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

    if fist_count > 0:
        BAR_BLOCKS = 12
        filled  = min(fist_count, BAR_BLOCKS)
        bar     = "\u2588" * filled + "\u2591" * (BAR_BLOCKS - filled)
        bar_txt = f"FIST [{bar}] {fist_count}/{FIST_LOCK_FRAMES}"
        cv2.putText(frame, bar_txt, (8, 18 + 5 * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 200, 255), 1)

    if flash_taskview:
        cv2.putText(frame, "TASK VIEW", (w // 2 - 90, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 200), 3)
    if flash_minimize:
        cv2.putText(frame, "MINIMIZE", (w // 2 - 80, h // 2 + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)


# ── Main loop ─────────────────────────────────────────────────────────────────
TARGET_FPS     = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS


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
        min_hand_presence_confidence=0.70,
        min_tracking_confidence=0.60,
    )

    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    # Re-apply v4l2 tuning AFTER VideoCapture opens the device.
    # cv2.VideoCapture() resets UVC controls to driver defaults on open.
    _tune = Path(__file__).parent / "cam-tune.sh"
    if _tune.exists():
        subprocess.run(["bash", str(_tune)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    play_startup_sound()

    win_name = "Kontrol v0.5"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, _mouse_cb)

    # ── Cursor state ──────────────────────────────────────────────────────────
    cx, cy               = float(SCREEN_W) / 2, float(SCREEN_H) / 2
    prev_tx, prev_ty     = cx, cy
    prev_sent_x, prev_sent_y = cx, cy
    # Stage 1 EMA accumulators — initialize to centre of camera frame
    raw_x_smooth: float  = 0.5
    raw_y_smooth: float  = 0.5

    # ── Gesture state ─────────────────────────────────────────────────────────
    drag_active            = False;  last_drag_end_t    = 0.0
    pinch_held_R           = False;  last_click_R       = 0.0
    scroll_ref_y: float | None = None
    wrist_history          = deque(maxlen=30)
    last_flick_t           = 0.0
    locked                 = False
    fist_hold_count        = 0
    fist_toggled_this_hold = False
    taskview_hold_count    = 0
    last_taskview_t        = 0.0
    taskview_flash_until   = 0.0
    minimize_flash_until   = 0.0
    hand_was_present       = False   # re-entry guard — prevents cursor jump

    # ── HUD / performance state ───────────────────────────────────────────────
    active_gesture = "NONE"
    pd_L_hud = pd_R_hud = 1.0
    fps          = 0.0
    fps_alpha    = 0.1          # rolling EMA smoothing for FPS display
    last_frame_t = time.time()

    print(f"Kontrol v0.5  {SCREEN_W}x{SCREEN_H}"
          f"  zone=[{ZONE_X_MIN:.2f}-{ZONE_X_MAX:.2f}, {ZONE_Y_MIN:.2f}-{ZONE_Y_MAX:.2f}]"
          f"  lm_smooth={LANDMARK_SMOOTH}"
          f"  smooth[{MIN_SMOOTH}-{MAX_SMOOTH}]x{VELOCITY_SCALE}"
          f"  edge_boost={EDGE_BOOST}"
          f"  pinch={PINCH_THRESHOLD}  config={CONF_PATH.name}")
    print(f"  Fist {FIST_MINIMIZE_FRAMES}fr=minimize  {FIST_LOCK_FRAMES}fr=lock"
          f"  WristRotate {TASKVIEW_HOLD_FRAMES}fr=taskview   Q=quit")

    with HandLandmarker.create_from_options(opts) as detector:
        while cap.isOpened():
            frame_start = time.time()

            ok, frame = cap.read()
            if not ok:
                continue

            now = time.time()

            # Rolling FPS EMA
            dt = now - last_frame_t
            last_frame_t = now
            if dt > 0:
                fps = fps * (1.0 - fps_alpha) + (1.0 / dt) * fps_alpha

            if FLIP:
                frame = cv2.flip(frame, 1)

            fh_px, fw_px = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]

                # ── Re-entry: reset state on first frame after hand absence ───
                if not hand_was_present:
                    raw_x_smooth = lm[8].x
                    raw_y_smooth = lm[8].y
                    nx0 = _clamp((raw_x_smooth - ZONE_X_MIN) / (ZONE_X_MAX - ZONE_X_MIN), 0.0, 1.0)
                    ny0 = _clamp((raw_y_smooth - ZONE_Y_MIN) / (ZONE_Y_MAX - ZONE_Y_MIN), 0.0, 1.0)
                    cx, cy             = nx0 * SCREEN_W, ny0 * SCREEN_H
                    prev_tx, prev_ty   = cx, cy
                    prev_sent_x, prev_sent_y = cx, cy
                hand_was_present = True

                wrist_history.append((now, lm[0].x, lm[0].y))   # normalized coords

                pd_L_hud = pinch_dist(lm, 4, 8)
                pd_R_hud = pinch_dist(lm, 4, 16)

                # ── Priority 1: Fist — tiered hold (bypasses lock) ────────────
                fist_now = is_fist(lm)
                if fist_now:
                    fist_hold_count += 1
                    if fist_hold_count >= FIST_LOCK_FRAMES and not fist_toggled_this_hold:
                        if drag_active:
                            mouse_up()
                            drag_active = False
                        locked = not locked
                        fist_toggled_this_hold = True
                else:
                    # Released — check for medium hold (minimize)
                    if (FIST_MINIMIZE_FRAMES <= fist_hold_count < FIST_LOCK_FRAMES
                            and not fist_toggled_this_hold
                            and not locked):
                        # wmctrl is the reliable cross-setup approach for iconify;
                        # KDE has no guaranteed global minimize keybind by default.
                        subprocess.Popen(
                            ["wmctrl", "-r", ":ACTIVE:", "-b", "add,hidden"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                        minimize_flash_until = now + 0.5
                        active_gesture = "MINIMIZE"
                    fist_hold_count        = 0
                    fist_toggled_this_hold = False

                # ── Priority 2: Lock ──────────────────────────────────────────
                if locked:
                    active_gesture = "FIST-LOCKED" if fist_now else "LOCKED"

                # ── Fist in progress (not yet locked) ─────────────────────────
                elif fist_now:
                    active_gesture = f"FIST {fist_hold_count}/{FIST_LOCK_FRAMES}"

                # ── Priority 3: Wrist rotate → task overview ──────────────────
                elif is_wrist_rotated(lm):
                    scroll_ref_y = None
                    if drag_active:
                        mouse_up(); drag_active = False
                    taskview_hold_count += 1
                    if (taskview_hold_count >= TASKVIEW_HOLD_FRAMES
                            and (now - last_taskview_t) > TASKVIEW_COOLDOWN):
                        # Meta+PgUp (KEY_PAGEUP=104)
                        ydocall("key", "125:1", "104:1", "104:0", "125:0", blocking=True)
                        last_taskview_t     = now
                        taskview_flash_until = now + 1.0
                        taskview_hold_count = 0
                    active_gesture = f"ROTATE {taskview_hold_count}/{TASKVIEW_HOLD_FRAMES}"

                # ── Priority 4: Scroll ────────────────────────────────────────
                elif is_scroll_pose(lm):
                    taskview_hold_count = 0
                    if drag_active:
                        mouse_up(); drag_active = False
                    cur_y = lm[8].y
                    if scroll_ref_y is None:
                        scroll_ref_y = cur_y
                    else:
                        dy_s = cur_y - scroll_ref_y
                        scroll_ref_y = cur_y
                        if abs(dy_s) > SCROLL_DEADZONE:
                            ticks = max(1, int(abs(dy_s) * SCROLL_SPEED))
                            if dy_s < 0:
                                scroll_up(ticks)
                                active_gesture = f"SCROLL UP x{ticks}"
                            else:
                                scroll_down(ticks)
                                active_gesture = f"SCROLL DN x{ticks}"
                        else:
                            active_gesture = "SCROLL"

                    for lm_idx in [8, 12]:
                        spx = (int(lm[lm_idx].x * fw_px), int(lm[lm_idx].y * fh_px))
                        cv2.circle(frame, spx, 14, (0, 200, 255), -1)
                        cv2.circle(frame, spx, 14, (255, 255, 255), 2)

                # ── Priority 5-8: Flick / right-click / drag / cursor ─────────
                else:
                    scroll_ref_y        = None
                    taskview_hold_count = 0

                    # Priority 5: Flick — cursor does NOT move during flick
                    flick = check_flick(wrist_history)
                    if flick and (now - last_flick_t) > FLICK_COOLDOWN:
                        tiling_key(flick)
                        last_flick_t   = now
                        active_gesture = f"FLICK {flick.upper()}"
                        wrist_history.clear()

                    else:
                        # ── Double EMA cursor tracking ────────────────────────
                        # Stage 1: fixed-alpha pre-filter on raw landmark
                        raw_x_smooth = raw_x_smooth * (1.0 - LANDMARK_SMOOTH) + lm[8].x * LANDMARK_SMOOTH
                        raw_y_smooth = raw_y_smooth * (1.0 - LANDMARK_SMOOTH) + lm[8].y * LANDMARK_SMOOTH

                        # Zone mapping: [zone_min, zone_max] → [0, 1] → screen pixels
                        nx = _clamp((raw_x_smooth - ZONE_X_MIN) / (ZONE_X_MAX - ZONE_X_MIN), 0.0, 1.0)
                        ny = _clamp((raw_y_smooth - ZONE_Y_MIN) / (ZONE_Y_MAX - ZONE_Y_MIN), 0.0, 1.0)
                        tx = nx * SCREEN_W
                        ty = ny * SCREEN_H

                        # Edge boost: snappier when cursor is near screen edge
                        near_edge = nx < 0.1 or nx > 0.9 or ny < 0.1 or ny > 0.9
                        boost     = EDGE_BOOST if near_edge else 1.0

                        # Stage 2: velocity-adaptive screen-space EMA
                        vel_norm = math.hypot(tx - prev_tx, ty - prev_ty) / SCREEN_DIAG
                        smooth   = _clamp(vel_norm * VELOCITY_SCALE * boost, MIN_SMOOTH, MAX_SMOOTH)
                        cx = cx * (1.0 - smooth) + tx * smooth
                        cy = cy * (1.0 - smooth) + ty * smooth
                        prev_tx, prev_ty = tx, ty

                        dx = round(cx - prev_sent_x)
                        dy = round(cy - prev_sent_y)
                        if max(abs(dx), abs(dy)) >= CURSOR_DEADZONE_PX:
                            move_cursor(dx, dy)
                            prev_sent_x += dx
                            prev_sent_y += dy

                        # Priority 6: Right click (ring + thumb)
                        if pd_R_hud < PINCH_THRESHOLD:
                            if not pinch_held_R and (now - last_click_R) > PINCH_COOLDOWN:
                                if drag_active:
                                    mouse_up(); drag_active = False
                                right_click(); last_click_R = now
                            pinch_held_R   = True
                            active_gesture = f"R-CLICK d={pd_R_hud:.3f}"
                        else:
                            pinch_held_R = False

                        # Priority 7-8: Left pinch — hold = drag, tap = click
                        left_pinched = (pd_L_hud < PINCH_THRESHOLD and
                                        not (pd_R_hud < PINCH_THRESHOLD))
                        if left_pinched:
                            if not drag_active:
                                if (now - last_drag_end_t) > PINCH_COOLDOWN:
                                    mouse_down()
                                    drag_active = True
                            active_gesture = (f"DRAG  d={pd_L_hud:.3f}" if drag_active
                                              else f"PINCH d={pd_L_hud:.3f}")
                        else:
                            if drag_active:
                                mouse_up()
                                drag_active     = False
                                last_drag_end_t = now
                                wrist_history.clear()   # prevent drag momentum triggering flick
                            elif not (pd_R_hud < PINCH_THRESHOLD):
                                active_gesture = "CURSOR"

                draw_skeleton(frame, lm, fw_px, fh_px)
                draw_visibility_warning(frame, lm, fw_px, fh_px)

            else:
                # Hand lost — reset transient state
                hand_was_present       = False
                fist_hold_count        = 0
                fist_toggled_this_hold = False
                scroll_ref_y           = None
                taskview_hold_count    = 0
                if drag_active:
                    mouse_up()
                    drag_active = False
                active_gesture = "NONE"

            draw_hud(
                frame, locked, active_gesture, fps, pd_L_hud, pd_R_hud,
                fist_count=fist_hold_count,
                flash_taskview=(now < taskview_flash_until),
                flash_minimize=(now < minimize_flash_until),
            )
            draw_buttons(frame)
            cv2.imshow(win_name, frame)

            if _quit_flag[0] or (cv2.waitKey(1) & 0xFF == ord("q")):
                print("Quit.")
                break

            # 20 fps cap — sleep remainder of frame interval
            elapsed   = time.time() - frame_start
            remaining = FRAME_INTERVAL - elapsed
            if remaining > 0.002:
                time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
