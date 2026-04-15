#!/usr/bin/env python3
"""
Kontrol v1.3 — Hand gesture mouse control
MediaPipe Tasks API (0.10+) → ydotool (no pynput, Wayland-safe)

Gesture priority order (strict — higher number never fires if lower active):
  1. Bunch (all 5 tips together, hold ~1 s, fire on RELEASE) → show desktop / restore
  2. Pinky+Thumb (LM 4+20) held + wrist direction            → KDE tile
  3. Peace sign  (index+middle up, rest folded) + swipe      → task switcher
  4. Middle+Thumb (LM 4+12) + vertical wrist movement        → scroll
  5. Ring+Thumb  (LM 4+16)                                   → right click
  6. Index+Thumb (LM 4+8)  hold/tap                          → drag / left click
  7. Index fingertip (LM 8) — 5-stage EMA pipeline           → cursor

Detection phases:
  SEARCHING — high thresholds, 3 confident frames to lock
  LOCKED    — lower thresholds, tolerates 5 miss frames before unlocking
  Hot-reload on transition (~50 ms acceptable)

Config: kontrol.conf  |  Launch: ./run.sh
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import configparser
import math
import os
import struct
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import cv2
import mediapipe as mp

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONF_PATH  = Path(__file__).parent / "kontrol.conf"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

_DEFAULTS: dict[str, dict[str, str]] = {
    "screen": {"width": "4480", "height": "1440"},
    "camera": {"id": "0", "flip": "true"},
    "mapping": {
        "zone_x_min": "0.15", "zone_x_max": "0.85",
        "zone_y_min": "0.10", "zone_y_max": "0.90",
    },
    "smoothing": {
        "landmark_smooth":    "0.35",
        "min_smooth":         "0.06",
        "max_smooth":         "0.40",
        "velocity_scale":     "4.0",
        "cursor_deadzone_px": "2",
    },
    "detection": {
        "detect_thresh_search":   "0.75",
        "presence_thresh_search": "0.75",
        "track_thresh_search":    "0.65",
        "detect_thresh_locked":   "0.60",
        "presence_thresh_locked": "0.60",
        "track_thresh_locked":    "0.50",
        "lock_frames":            "3",
        "unlock_frames":          "5",
    },
    "gestures": {
        "pinch_threshold":     "0.048",
        "pinch_cooldown":      "0.35",
        "scroll_deadzone":     "0.008",
        "scroll_speed":        "6.0",
        "scroll_vel_alpha":    "0.30",
        "scroll_max_ticks":    "8",
        "palm_hold_frames":    "20",
        "palm_cooldown":       "2.0",
        "tile_move_threshold": "0.050",
        "tile_cooldown":       "0.8",
        "task_move_threshold": "0.06",
        "task_cooldown":       "0.5",
        "bunch_threshold":     "0.10",
    },
    "system": {"ydotool_socket": "/run/user/1000/.ydotool_socket"},
    "camera_tuning": {
        "auto_exposure":          "1",
        "exposure_time_absolute": "300",
        "gain":                   "100",
        "brightness":             "160",
        "contrast":               "130",
        "saturation":             "128",
        "backlight_compensation": "1",
        "power_line_frequency":   "1",
        "focus_autos":            "0",
        "focus_absolute":         "30",
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

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SCREEN_W           = _cfg.getint("screen",    "width")
SCREEN_H           = _cfg.getint("screen",    "height")
SCREEN_DIAG        = math.hypot(SCREEN_W, SCREEN_H)

CAM_ID             = _cfg.getint("camera",    "id")
FLIP               = _cfg.getboolean("camera","flip")

ZONE_X_MIN         = _cfg.getfloat("mapping", "zone_x_min")
ZONE_X_MAX         = _cfg.getfloat("mapping", "zone_x_max")
ZONE_Y_MIN         = _cfg.getfloat("mapping", "zone_y_min")
ZONE_Y_MAX         = _cfg.getfloat("mapping", "zone_y_max")

LANDMARK_SMOOTH    = _cfg.getfloat("smoothing", "landmark_smooth")
MIN_SMOOTH         = _cfg.getfloat("smoothing", "min_smooth")
MAX_SMOOTH         = _cfg.getfloat("smoothing", "max_smooth")
VELOCITY_SCALE     = _cfg.getfloat("smoothing", "velocity_scale")
CURSOR_DEADZONE_PX = _cfg.getint("smoothing",   "cursor_deadzone_px")

DT_SEARCH     = _cfg.getfloat("detection", "detect_thresh_search")
PR_SEARCH     = _cfg.getfloat("detection", "presence_thresh_search")
TR_SEARCH     = _cfg.getfloat("detection", "track_thresh_search")
DT_LOCKED     = _cfg.getfloat("detection", "detect_thresh_locked")
PR_LOCKED     = _cfg.getfloat("detection", "presence_thresh_locked")
TR_LOCKED     = _cfg.getfloat("detection", "track_thresh_locked")
LOCK_FRAMES   = _cfg.getint("detection",   "lock_frames")
UNLOCK_FRAMES = _cfg.getint("detection",   "unlock_frames")

PINCH_THRESHOLD  = _cfg.getfloat("gestures", "pinch_threshold")
PINCH_COOLDOWN   = _cfg.getfloat("gestures", "pinch_cooldown")
SCROLL_DEADZONE  = _cfg.getfloat("gestures", "scroll_deadzone")
SCROLL_SPEED     = _cfg.getfloat("gestures", "scroll_speed")
SCROLL_VEL_ALPHA = _cfg.getfloat("gestures", "scroll_vel_alpha")
SCROLL_MAX_TICKS = _cfg.getint("gestures",   "scroll_max_ticks")
PALM_HOLD_FRAMES = _cfg.getint("gestures",   "palm_hold_frames")
PALM_COOLDOWN    = _cfg.getfloat("gestures", "palm_cooldown")
TILE_THRESHOLD   = _cfg.getfloat("gestures", "tile_move_threshold")
TILE_COOLDOWN    = _cfg.getfloat("gestures", "tile_cooldown")
TASK_THRESHOLD   = _cfg.getfloat("gestures", "task_move_threshold")
TASK_COOLDOWN    = _cfg.getfloat("gestures", "task_cooldown")
BUNCH_THRESHOLD  = _cfg.getfloat("gestures", "bunch_threshold")

YDOTOOL_SOCKET   = _cfg.get("system", "ydotool_socket")

TARGET_FPS     = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Landmark indices to apply EMA smoothing (gesture-critical nodes)
_LMKS_TO_SMOOTH = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]

# Raw Linux keycodes (input-event-codes.h)
# KEY_LEFTMETA=125  KEY_UP=103  KEY_DOWN=108  KEY_LEFT=105  KEY_RIGHT=106
_TILING_KEYS: dict[str, tuple] = {
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


# ── DETECTOR FACTORY ─────────────────────────────────────────────────────────
def build_detector(detect_t: float, presence_t: float, track_t: float) -> HandLandmarker:
    """Create a HandLandmarker (IMAGE mode) with the given confidence thresholds."""
    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=detect_t,
        min_hand_presence_confidence=presence_t,
        min_tracking_confidence=track_t,
    )
    return HandLandmarker.create_from_options(opts)


# ── YDOTOOL HELPERS ───────────────────────────────────────────────────────────
def ydocall(*args, blocking: bool = False) -> None:
    cmd = ["ydotool", *[str(a) for a in args]]
    if blocking:
        subprocess.run(cmd, env=os.environ,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(cmd, env=os.environ,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mouse_down() -> None:  ydocall("click", "0x40")
def mouse_up()   -> None:  ydocall("click", "0x80")
def right_click()-> None:  ydocall("click", "0xC1")

def tiling_key(direction: str) -> None:
    seq = _TILING_KEYS.get(direction)
    if seq:
        ydocall("key", *seq, blocking=True)


# ── FOCUS GUARD ───────────────────────────────────────────────────────────────
_focus_cache: list = [0.0, False]

def _kontrol_is_active(win_name: str) -> bool:
    """True if Kontrol window is the active X window. Cached 0.25 s."""
    now = time.monotonic()
    if now - _focus_cache[0] < 0.25:
        return _focus_cache[1]
    try:
        r1 = subprocess.run(["xprop", "-root", "_NET_ACTIVE_WINDOW"],
                            capture_output=True, text=True, timeout=0.05)
        wid = r1.stdout.strip().split()[-1]
        r2 = subprocess.run(["xprop", "-id", wid, "WM_NAME"],
                            capture_output=True, text=True, timeout=0.05)
        _focus_cache[1] = win_name in r2.stdout
    except Exception:
        _focus_cache[1] = False
    _focus_cache[0] = now
    return _focus_cache[1]


# ── LANDMARK HELPERS ─────────────────────────────────────────────────────────
def pdist(lm, a: int, b: int) -> float:
    """Normalised distance between two raw landmarks."""
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


# Module-level EMA state for smoothed landmarks; cleared on hand loss
_smooth_lm: dict[int, tuple[float, float]] = {}


def reset_smooth() -> None:
    _smooth_lm.clear()


def smooth_landmarks(lm, alpha: float) -> dict[int, tuple[float, float]]:
    """
    EMA-smooth the gesture-critical landmarks.
    Returns {index: (sx, sy)} in normalised [0,1] coords.
    First call per landmark snaps to current position (no jump).
    """
    result: dict[int, tuple[float, float]] = {}
    for i in _LMKS_TO_SMOOTH:
        rx, ry = lm[i].x, lm[i].y
        if i not in _smooth_lm:
            _smooth_lm[i] = (rx, ry)
        sx, sy = _smooth_lm[i]
        sx += (rx - sx) * alpha
        sy += (ry - sy) * alpha
        _smooth_lm[i] = (sx, sy)
        result[i] = (sx, sy)
    return result


def pdist_s(slm: dict, a: int, b: int) -> float:
    """Distance between two smoothed landmarks."""
    ax, ay = slm[a];  bx, by = slm[b]
    return math.hypot(ax - bx, ay - by)


def is_bunch_s(slm: dict) -> bool:
    """All 5 fingertips converge to one point — smoothed."""
    t = BUNCH_THRESHOLD
    return (pdist_s(slm, 4, 8)  < t and pdist_s(slm, 4, 12) < t and
            pdist_s(slm, 4, 16) < t and pdist_s(slm, 4, 20) < t)


def is_pinky_thumb_s (slm: dict) -> bool: return pdist_s(slm, 4, 20) < PINCH_THRESHOLD
def is_middle_thumb_s(slm: dict) -> bool: return pdist_s(slm, 4, 12) < PINCH_THRESHOLD
def is_ring_thumb_s  (slm: dict) -> bool: return pdist_s(slm, 4, 16) < PINCH_THRESHOLD
def is_index_thumb_s (slm: dict) -> bool: return pdist_s(slm, 4,  8) < PINCH_THRESHOLD


def is_peace_sign(lm) -> bool:
    """Index+middle extended up; ring+pinky folded; no thumb pinch."""
    return (lm[8].y < lm[5].y and lm[12].y < lm[9].y and
            lm[16].y > lm[13].y and lm[20].y > lm[17].y and
            pdist(lm, 4, 8)  > PINCH_THRESHOLD and
            pdist(lm, 4, 12) > PINCH_THRESHOLD and
            pdist(lm, 4, 16) > PINCH_THRESHOLD and
            pdist(lm, 4, 20) > PINCH_THRESHOLD)


# ── CAMERA ────────────────────────────────────────────────────────────────────
def apply_camera_settings() -> None:
    """Apply v4l2 controls from [camera_tuning]. Call after VideoCapture opens."""
    dev = f"/dev/video{CAM_ID}"
    ct  = _cfg["camera_tuning"]

    def v4l2_set(ctrl: str, val: str) -> None:
        try:
            subprocess.run(["v4l2-ctl", "-d", dev, f"--set-ctrl={ctrl}={val}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=2)
            print(f"[CAM] {ctrl} = {val} \u2713")
        except Exception as e:
            print(f"[CAM] {ctrl} SKIPPED ({e})")

    time.sleep(0.3)
    v4l2_set("auto_exposure",              ct.get("auto_exposure",            "1"))
    v4l2_set("exposure_time_absolute",     ct.get("exposure_time_absolute",   "300"))
    v4l2_set("gain",                       ct.get("gain",                     "100"))
    v4l2_set("brightness",                 ct.get("brightness",               "160"))
    v4l2_set("contrast",                   ct.get("contrast",                 "130"))
    v4l2_set("saturation",                 ct.get("saturation",               "128"))
    v4l2_set("backlight_compensation",     ct.get("backlight_compensation",   "1"))
    v4l2_set("power_line_frequency",       ct.get("power_line_frequency",     "1"))
    v4l2_set("focus_automatic_continuous", ct.get("focus_autos",              "0"))
    v4l2_set("focus_absolute",             ct.get("focus_absolute",           "30"))


# ── HUD COLOUR PALETTE (BGR) ──────────────────────────────────────────────────
_C_CYAN   = (220, 200,   0)
_C_WHITE  = (255, 255, 255)
_C_GREEN  = ( 50, 220,  50)
_C_RED    = ( 50,  50, 220)
_C_YELLOW = ( 50, 220, 220)
_C_BLUE   = (220, 100,  50)
_C_ORANGE = ( 50, 140, 255)
_C_GREY   = (140, 140, 140)
_FONT     = cv2.FONT_HERSHEY_SIMPLEX


# ── DRAW FUNCTIONS ────────────────────────────────────────────────────────────
def draw_panel(frame, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    """Semi-transparent dark rectangle for HUD background."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_progress_bar(frame, x: int, y: int, w: int, h: int,
                      progress: float, color: tuple) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    filled = int(w * max(0.0, min(1.0, progress)))
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 120), 1)


def draw_skeleton(frame, lm, slm: dict, fw: int, fh: int) -> None:
    """Styled hand skeleton with active-pinch ring highlights."""
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    for a, b in CONNECTIONS:
        cv2.line(frame, (int(lm[a].x*fw), int(lm[a].y*fh)),
                        (int(lm[b].x*fw), int(lm[b].y*fh)), (80, 80, 80), 1)

    for i in range(21):
        cv2.circle(frame, (int(lm[i].x*fw), int(lm[i].y*fh)), 3, (160, 160, 160), -1)

    # Wrist — white, larger
    cv2.circle(frame, (int(lm[0].x*fw), int(lm[0].y*fh)), 6, _C_WHITE, 2)

    # Index tip — cursor controller, always highlighted green
    ix, iy = int(lm[8].x*fw), int(lm[8].y*fh)
    cv2.circle(frame, (ix, iy), 10, (180, 255, 180), -1)
    cv2.circle(frame, (ix, iy), 10, _C_WHITE, 1)

    # Thumb tip
    cv2.circle(frame, (int(lm[4].x*fw), int(lm[4].y*fh)), 8, (180, 180, 255), -1)

    # Active pinch ring highlights
    if pdist_s(slm, 4, 8) < PINCH_THRESHOLD:
        cv2.circle(frame, (ix, iy), 14, _C_RED, 2)
    if pdist_s(slm, 4, 12) < PINCH_THRESHOLD:
        cv2.circle(frame, (int(lm[12].x*fw), int(lm[12].y*fh)), 14, _C_BLUE, 2)
    if pdist_s(slm, 4, 20) < PINCH_THRESHOLD:
        cv2.circle(frame, (int(lm[20].x*fw), int(lm[20].y*fh)), 14, _C_YELLOW, 2)


def draw_zone_warning(frame, slm: dict, fw: int, fh: int) -> None:
    """Arrows when index tip is within 5% of zone boundary (smoothed coords)."""
    x, y   = slm[8]
    margin = 0.05
    col    = _C_YELLOW
    if x < ZONE_X_MIN + margin:
        cv2.putText(frame, "< MOVE IN", (4, fh//2), _FONT, 0.60, col, 2)
    elif x > ZONE_X_MAX - margin:
        cv2.putText(frame, "MOVE IN >", (fw-110, fh//2), _FONT, 0.60, col, 2)
    if y < ZONE_Y_MIN + margin:
        cv2.putText(frame, "^ MOVE IN", (fw//3, 22), _FONT, 0.60, col, 2)
    elif y > ZONE_Y_MAX - margin:
        cv2.putText(frame, "v MOVE IN", (fw//3, fh-8), _FONT, 0.60, col, 2)


def draw_detection_phase(frame, phase: str, fw: int) -> None:
    """Pulsing SRCH or solid LOCK indicator in top-right."""
    if phase == "LOCKED":
        dot_color = _C_GREEN
        label     = "LOCK"
    else:
        pulse     = 0.5 + 0.5 * math.sin(time.time() * 4.0)
        b         = int(100 + 100 * pulse)
        dot_color = (b, b, b)
        label     = "SRCH"
    cv2.circle(frame, (fw - 15, 15), 6, dot_color, -1)
    cv2.putText(frame, label, (fw - 55, 20), _FONT, 0.40, dot_color, 1)


def draw_hud(frame, gesture: str, fps: float,
             pd_it: float, pd_mt: float, pd_rt: float, pd_pt: float,
             palm_count: int = 0, drag_active: bool = False,
             tile_held: bool = False, hand_detected: bool = True,
             palm_closing: bool = False, flash_msg: str = "",
             flash_until: float = 0.0,
             detection_phase: str = "SEARCHING") -> None:
    fh, fw = frame.shape[:2]
    now    = time.time()

    # Border
    if not hand_detected:          state = "no_hand"
    elif palm_closing:             state = "palm"
    elif drag_active:              state = "drag"
    elif tile_held:                state = "tile"
    else:                          state = "tracking"
    _BORDER_COLORS = {
        "no_hand":  ( 50,  50, 180),
        "tracking": ( 50, 180,  50),
        "drag":     (180,  80,  30),
        "tile":     ( 50, 200, 200),
        "palm":     ( 50, 120, 220),
    }
    cv2.rectangle(frame, (0,0), (fw-1,fh-1),
                  _BORDER_COLORS.get(state, _BORDER_COLORS["tracking"]), 3)

    # Panel background — taller when palm bar is visible
    panel_h = 134 if palm_count > 0 else 112
    draw_panel(frame, 0, 0, 222, panel_h)

    # HUD rows
    x0, y0, dy = 8, 18, 18

    def lbl(row: int, text: str) -> None:
        cv2.putText(frame, text, (x0, y0 + row*dy), _FONT, 0.42, _C_CYAN, 1)

    def val(row: int, text: str, color: tuple = _C_WHITE, thick: int = 1) -> None:
        cv2.putText(frame, text, (x0+68, y0+row*dy), _FONT, 0.42, color, thick)

    # Row 0: FPS — green ≥20, yellow 15–19, red <15
    lbl(0, "[FPS]")
    fps_col = _C_GREEN if fps >= 20 else (_C_YELLOW if fps >= 15 else _C_RED)
    val(0, f"{fps:5.1f}", fps_col)

    # Row 1: detection phase
    lbl(1, "[HAND]")
    val(1, detection_phase, _C_GREEN if detection_phase == "LOCKED" else _C_GREY)

    # Row 2: mode
    lbl(2, "[MODE]")
    val(2, "TRACKING" if hand_detected else "NO HAND",
        _C_GREEN if hand_detected else _C_RED)

    # Row 3: active gesture — bold and yellow when active
    lbl(3, "[ACT]")
    is_idle = gesture in ("CURSOR", "NO HAND", "NONE")
    val(3, gesture, _C_WHITE if is_idle else _C_YELLOW, 1 if is_idle else 2)

    # Row 4: pinch distances — green when below threshold
    lbl(4, "[PIN]")
    cx0 = x0 + 68
    for label, dist in (("I", pd_it), ("M", pd_mt), ("P", pd_pt)):
        color = _C_GREEN if dist < PINCH_THRESHOLD else _C_GREY
        seg   = f"{label}={dist:.3f} "
        cv2.putText(frame, seg, (cx0, y0+4*dy), _FONT, 0.38, color, 1)
        (tw, _), _ = cv2.getTextSize(seg, _FONT, 0.38, 1)
        cx0 += tw

    # Row 5: palm progress bar (only when closing)
    if palm_count > 0:
        lbl(5, "[PALM]")
        progress  = palm_count / PALM_HOLD_FRAMES
        bar_color = _C_GREEN if progress >= 0.5 else _C_ORANGE
        draw_progress_bar(frame, x0+68, y0+5*dy-12, 120, 10, progress, bar_color)

    # Detection phase indicator (top-right)
    draw_detection_phase(frame, detection_phase, fw)

    # Flash message — centred, shadowed
    if flash_msg and now < flash_until:
        (tw, th), _ = cv2.getTextSize(flash_msg, _FONT, 1.4, 2)
        tx = (fw - tw) // 2
        ty = (fh + th) // 2
        cv2.putText(frame, flash_msg, (tx+2, ty+2), _FONT, 1.4, (0,0,0), 4)
        cv2.putText(frame, flash_msg, (tx,   ty),   _FONT, 1.4, _C_YELLOW, 2)


# ── IN-FRAME BUTTONS ─────────────────────────────────────────────────────────
_quit_flag               = [False]
_mouse_xy                = [0, 0]
_btn_close: tuple | None = None
_btn_min:   tuple | None = None


def _mouse_cb(event, x, y, flags, param) -> None:
    _mouse_xy[0], _mouse_xy[1] = x, y
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if _btn_close and _btn_close[0] <= x <= _btn_close[2] and \
                      _btn_close[1] <= y <= _btn_close[3]:
        _quit_flag[0] = True
    elif _btn_min and _btn_min[0] <= x <= _btn_min[2] and \
                      _btn_min[1] <= y <= _btn_min[3]:
        subprocess.Popen(["wmctrl", "-r", ":ACTIVE:", "-b", "add,hidden"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def draw_buttons(frame) -> None:
    """Hover-aware [—] and [✕] buttons in top-right corner."""
    global _btn_close, _btn_min
    fh_f, fw_f = frame.shape[:2]
    mx, my     = _mouse_xy
    pad, bw, bh = 5, 28, 20

    def _btn(rect: tuple, label: str) -> None:
        x1, y1, x2, y2 = rect
        bg = (80,80,80) if (x1<=mx<=x2 and y1<=my<=y2) else (40,40,40)
        cv2.rectangle(frame, (x1,y1), (x2,y2), bg, -1)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (120,120,120), 1)
        (tw,th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        cv2.putText(frame, label,
                    (x1+(x2-x1-tw)//2, y1+(y2-y1+th)//2),
                    _FONT, 0.45, (220,220,220), 1)

    cx1 = fw_f - pad - bw
    _btn_close = (cx1, pad, fw_f-pad, pad+bh);  _btn(_btn_close, "x")
    mx1 = cx1 - pad - bw
    _btn_min   = (mx1, pad, cx1-pad,  pad+bh);  _btn(_btn_min,   "-")


# ── AUDIO ─────────────────────────────────────────────────────────────────────
def play_startup_sound() -> None:
    """Three-tone ascending chime on startup."""
    rate, amp = 22050, 28000

    def burst(freq: float, dur: float) -> bytes:
        n   = int(rate * dur)
        out = bytearray(n * 2)
        for i in range(n):
            t   = i / rate
            env = min(t/(dur*0.1), 1.0, (dur-t)/(dur*0.1))
            struct.pack_into("<h", out, i*2,
                             int(amp * env * math.sin(2*math.pi*freq*t)))
        return bytes(out)

    silence = bytes(int(rate * 0.04) * 2)
    pcm     = silence.join(burst(f, 0.10) for f in (880, 1047, 1319))
    tmp     = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(pcm)
    tmp.close()
    subprocess.Popen(["aplay", "-q", tmp.name],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def play_palm_beep() -> None:
    """Short descending two-tone palm-trigger beep."""
    rate, amp = 22050, 16000

    def tone(freq: float, dur: float) -> list:
        n = int(rate * dur)
        return [int(amp * math.sin(2*math.pi*freq*i/rate)) for i in range(n)]

    samples = tone(600, 0.08) + tone(400, 0.08)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    tmp.close()
    subprocess.Popen(["aplay", "-q", tmp.name],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── CURSOR PIPELINE ───────────────────────────────────────────────────────────
def run_cursor_pipeline(
    lm, raw_x_s, raw_y_s, cx, cy, prev_tx, prev_ty, prev_sent_x, prev_sent_y,
    reentry: bool,
) -> tuple:
    """
    5-stage cursor pipeline.
    reentry=True → snap to current position, send no delta.
    """
    lx, ly = lm[8].x, lm[8].y

    # Stage 1 — landmark pre-smooth
    if raw_x_s is None:
        raw_x_s, raw_y_s = lx, ly
    raw_x_s += (lx - raw_x_s) * LANDMARK_SMOOTH
    raw_y_s += (ly - raw_y_s) * LANDMARK_SMOOTH

    # Stage 2 — zone mapping → screen pixels
    nx = max(0.0, min(1.0, (raw_x_s - ZONE_X_MIN) / (ZONE_X_MAX - ZONE_X_MIN)))
    ny = max(0.0, min(1.0, (raw_y_s - ZONE_Y_MIN) / (ZONE_Y_MAX - ZONE_Y_MIN)))
    tx, ty = nx * SCREEN_W, ny * SCREEN_H

    # Stage 3 — re-entry guard
    if reentry:
        cx, cy     = tx, ty
        prev_tx, prev_ty = tx, ty
        prev_sent_x, prev_sent_y = int(tx), int(ty)
        return raw_x_s, raw_y_s, cx, cy, prev_tx, prev_ty, prev_sent_x, prev_sent_y

    # Stage 4 — velocity-adaptive EMA
    vel_norm = math.hypot(tx - prev_tx, ty - prev_ty) / SCREEN_DIAG
    prev_tx, prev_ty = tx, ty
    alpha = max(MIN_SMOOTH, min(MAX_SMOOTH, vel_norm * VELOCITY_SCALE))
    cx += (tx - cx) * alpha
    cy += (ty - cy) * alpha

    # Stage 5 — integer delta with deadzone
    dx = round(cx - prev_sent_x)
    dy = round(cy - prev_sent_y)
    if max(abs(dx), abs(dy)) >= CURSOR_DEADZONE_PX:
        subprocess.Popen(
            ["ydotool", "mousemove", "-x", str(dx), "-y", str(dy)],
            env=os.environ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        prev_sent_x += dx
        prev_sent_y += dy

    return raw_x_s, raw_y_s, cx, cy, prev_tx, prev_ty, prev_sent_x, prev_sent_y


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def run() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Download:\n  curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    apply_camera_settings()
    for _ in range(5):
        cap.read()
    apply_camera_settings()

    _ok, _frame = cap.read()
    if _ok:
        print(f"[CAM] brightness mean: {_frame.mean():.1f}  (target >80)")

    play_startup_sound()

    win_name = "Kontrol v1.3"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 640, 480)
    cv2.setMouseCallback(win_name, _mouse_cb)

    # ── DETECTION PHASE ───────────────────────────────────────────────────────
    detection_phase          = "SEARCHING"
    consecutive_detections   = 0
    consecutive_no_detection = 0
    phase_changes            = 0

    detector = build_detector(DT_SEARCH, PR_SEARCH, TR_SEARCH)

    # ── CURSOR STATE ──────────────────────────────────────────────────────────
    raw_x_s: float | None  = None
    raw_y_s: float | None  = None
    cx: float              = SCREEN_W / 2
    cy: float              = SCREEN_H / 2
    prev_tx: float         = cx
    prev_ty: float         = cy
    prev_sent_x: int       = int(cx)
    prev_sent_y: int       = int(cy)
    hand_was_present: bool = False
    was_gesturing: bool    = False

    # ── GESTURE STATE — bunch (priority 1) ────────────────────────────────────
    palm_frames:     int   = 0
    palm_was_closed: bool  = False
    last_palm_t:     float = 0.0
    palm_minimized:  bool  = False
    palm_beep_fired: bool  = False

    # ── GESTURE STATE — pinky+thumb tiling (priority 2) ──────────────────────
    pt_held:      bool  = False
    tile_start_x: float = 0.0   # wrist x at pinch entry (absolute delta)
    tile_start_y: float = 0.0   # wrist y at pinch entry
    tile_fired:   bool  = False
    last_tile_t:  float = 0.0

    # ── GESTURE STATE — peace sign task switcher (priority 3) ─────────────────
    peace_held:    bool  = False
    task_wrist_xs: list  = []
    last_task_t:   float = 0.0

    # ── GESTURE STATE — middle+thumb scroll (priority 4) ─────────────────────
    mt_held:      bool         = False
    scroll_ref_y: float | None = None
    scroll_vel:   float        = 0.0

    # ── GESTURE STATE — ring+thumb right click (priority 5) ───────────────────
    rt_held:       bool  = False
    last_rclick_t: float = 0.0

    # ── GESTURE STATE — index+thumb left click / drag (priority 6) ───────────
    it_held:         bool  = False
    it_start_t:      float = 0.0
    drag_active:     bool  = False
    last_drag_end_t: float = 0.0

    # ── HUD STATE ─────────────────────────────────────────────────────────────
    active_gesture: str   = "NONE"
    flash_msg:      str   = ""
    flash_until:    float = 0.0
    fps:            float = 0.0
    fps_alpha:      float = 0.1
    last_frame_t:   float = time.time()
    pd_it: float = 1.0
    pd_mt: float = 1.0
    pd_rt: float = 1.0
    pd_pt: float = 1.0

    print(
        f"Kontrol v1.3  {SCREEN_W}x{SCREEN_H}"
        f"  cam=/dev/video{CAM_ID}"
        f"  pinch={PINCH_THRESHOLD}  bunch={BUNCH_THRESHOLD}/{PALM_HOLD_FRAMES}fr"
        f"  phase=SEARCHING"
    )

    while cap.isOpened():
        frame_start = time.perf_counter()

        ok, frame = cap.read()
        if not ok:
            continue

        now = time.time()
        dt = now - last_frame_t
        last_frame_t = now
        if dt > 0:
            fps = fps * (1.0 - fps_alpha) + (1.0 / dt) * fps_alpha

        if FLIP:
            frame = cv2.flip(frame, 1)

        fh_px, fw_px = frame.shape[:2]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)   # IMAGE mode — no timestamp

        # ── DETECTION PHASE MANAGEMENT ────────────────────────────────────────
        if result.hand_landmarks:
            consecutive_no_detection = 0
            consecutive_detections  += 1
            if detection_phase == "SEARCHING" and \
                    consecutive_detections >= LOCK_FRAMES:
                detection_phase = "LOCKED"
                phase_changes  += 1
                detector.close()
                detector = build_detector(DT_LOCKED, PR_LOCKED, TR_LOCKED)
                print(f"[PHASE] → LOCKED  ({consecutive_detections} consecutive frames)")
        else:
            consecutive_detections   = 0
            consecutive_no_detection += 1
            if detection_phase == "LOCKED" and \
                    consecutive_no_detection >= UNLOCK_FRAMES:
                detection_phase = "SEARCHING"
                phase_changes  += 1
                detector.close()
                detector = build_detector(DT_SEARCH, PR_SEARCH, TR_SEARCH)
                print(f"[PHASE] → SEARCHING  ({consecutive_no_detection} misses)")

        # ── PER-FRAME PROCESSING ──────────────────────────────────────────────
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            first_frame      = not hand_was_present
            hand_was_present = True

            # Smooth all gesture-critical landmarks once per frame
            slm = smooth_landmarks(lm, LANDMARK_SMOOTH)

            # Smoothed pinch distances for HUD display
            pd_it = pdist_s(slm, 4,  8)
            pd_mt = pdist_s(slm, 4, 12)
            pd_rt = pdist_s(slm, 4, 16)
            pd_pt = pdist_s(slm, 4, 20)

            # Smoothed wrist coords (used for tiling / scroll reference)
            wrist_x, wrist_y = slm[0]

            # ════════════════════════════════════════════════════════════════
            # PRIORITY 1 — BUNCH → SHOW DESKTOP / RESTORE
            # ════════════════════════════════════════════════════════════════
            if is_bunch_s(slm):
                palm_frames    += 1
                palm_was_closed = True
                was_gesturing   = True

                if drag_active:
                    mouse_up()
                    drag_active = False

                if palm_frames == PALM_HOLD_FRAMES // 2 and not palm_beep_fired:
                    print("[PALM] halfway — keep closed")
                if palm_frames == PALM_HOLD_FRAMES and not palm_beep_fired:
                    print("[PALM] firing")
                    play_palm_beep()
                    palm_beep_fired = True

                active_gesture = f"PALM {palm_frames}/{PALM_HOLD_FRAMES}"

            else:
                if palm_was_closed:
                    if palm_frames >= PALM_HOLD_FRAMES and \
                            (now - last_palm_t) > PALM_COOLDOWN:
                        subprocess.run(
                            ["qdbus", "org.kde.kglobalaccel", "/component/kwin",
                             "org.kde.kglobalaccel.Component.invokeShortcut",
                             "Show Desktop"],
                            env=os.environ,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                        palm_minimized = not palm_minimized
                        flash_msg   = "SHOW DESKTOP" if palm_minimized else "RESTORE"
                        flash_until = now + 1.5
                        last_palm_t = now
                    palm_frames     = 0
                    palm_was_closed = False
                    palm_beep_fired = False

                # ════════════════════════════════════════════════════════════
                # PRIORITY 2 — PINKY+THUMB → TILE (absolute wrist delta)
                # ════════════════════════════════════════════════════════════
                if is_pinky_thumb_s(slm):
                    active_gesture = f"TILE HOLD  P={pd_pt:.3f}"
                    was_gesturing  = True

                    if not pt_held:
                        tile_start_x = wrist_x   # record entry position
                        tile_start_y = wrist_y
                        tile_fired   = False
                        pt_held      = True

                    if not tile_fired and (now - last_tile_t) > TILE_COOLDOWN:
                        dx_abs = wrist_x - tile_start_x
                        dy_abs = wrist_y - tile_start_y
                        adx, ady = abs(dx_abs), abs(dy_abs)
                        if adx > TILE_THRESHOLD or ady > TILE_THRESHOLD:
                            direction = (
                                ("right" if dx_abs > 0 else "left") if adx > ady
                                else ("down" if dy_abs > 0 else "up")
                            )
                            tiling_key(direction)
                            tile_fired     = True
                            last_tile_t    = now
                            flash_msg      = f"TILE {direction.upper()}"
                            flash_until    = now + 0.8
                            active_gesture = f"TILE {direction.upper()}"

                else:
                    if pt_held:
                        tile_fired = False
                        pt_held    = False

                    # ════════════════════════════════════════════════════════
                    # PRIORITY 3 — PEACE SIGN → TASK SWITCHER
                    # ════════════════════════════════════════════════════════
                    if is_peace_sign(lm):
                        active_gesture = "TASK SWITCH"
                        was_gesturing  = True

                        if not peace_held:
                            task_wrist_xs.clear()
                            peace_held = True
                        task_wrist_xs.append(wrist_x)

                        if len(task_wrist_xs) >= 2 and \
                                (now - last_task_t) > TASK_COOLDOWN:
                            dx = task_wrist_xs[-1] - task_wrist_xs[0]
                            if abs(dx) > TASK_THRESHOLD:
                                if dx > 0:
                                    ydocall("key", "56:1","15:1","15:0","56:0",
                                            blocking=True)
                                    active_gesture = "TASK \u2192"
                                else:
                                    ydocall("key", "56:1","42:1","15:1",
                                            "15:0","42:0","56:0", blocking=True)
                                    active_gesture = "TASK \u2190"
                                last_task_t = now
                                task_wrist_xs.clear()

                    else:
                        if peace_held:
                            peace_held = False
                            task_wrist_xs.clear()

                        # ════════════════════════════════════════════════════
                        # PRIORITY 4 — MIDDLE+THUMB → SCROLL (velocity EMA)
                        # ════════════════════════════════════════════════════
                        if is_middle_thumb_s(slm):
                            was_gesturing = True

                            if not mt_held:
                                mt_held      = True
                                scroll_ref_y = wrist_y
                                scroll_vel   = 0.0

                            raw_dy       = wrist_y - scroll_ref_y
                            scroll_ref_y = wrist_y
                            scroll_vel   = (scroll_vel * (1 - SCROLL_VEL_ALPHA) +
                                            raw_dy * SCROLL_VEL_ALPHA)

                            if abs(scroll_vel) > SCROLL_DEADZONE:
                                ticks   = min(SCROLL_MAX_TICKS,
                                              max(1, int(abs(scroll_vel) * SCROLL_SPEED)))
                                wheel_y = -ticks if scroll_vel < 0 else ticks
                                ydocall("mousemove", "--wheel",
                                        "-x", "0", "-y", str(wheel_y))
                                active_gesture = (
                                    f"SCROLL {'UP' if scroll_vel < 0 else 'DOWN'} x{ticks}"
                                )
                            else:
                                active_gesture = f"SCROLL HOLD  M={pd_mt:.3f}"

                        else:
                            if mt_held:
                                mt_held      = False
                                scroll_ref_y = None
                                scroll_vel   = 0.0

                            # ════════════════════════════════════════════════
                            # PRIORITY 5 — RING+THUMB → RIGHT CLICK
                            # ════════════════════════════════════════════════
                            if is_ring_thumb_s(slm):
                                if not rt_held and \
                                        (now - last_rclick_t) > PINCH_COOLDOWN:
                                    if not _kontrol_is_active(win_name):
                                        right_click()
                                    last_rclick_t  = now
                                    active_gesture = f"R-CLICK  R={pd_rt:.3f}"
                                rt_held       = True
                                was_gesturing = True

                            else:
                                rt_held = False

                                # ════════════════════════════════════════════
                                # PRIORITY 6 — INDEX+THUMB → LEFT CLICK / DRAG
                                # ════════════════════════════════════════════
                                if is_index_thumb_s(slm):
                                    if not it_held:
                                        it_held    = True
                                        it_start_t = now

                                    held_t = now - it_start_t
                                    if held_t > PINCH_COOLDOWN and \
                                            not drag_active and \
                                            (now - last_drag_end_t) > PINCH_COOLDOWN:
                                        if not _kontrol_is_active(win_name):
                                            mouse_down()
                                            drag_active = True

                                    if drag_active:
                                        reentry = first_frame or was_gesturing
                                        (raw_x_s, raw_y_s, cx, cy,
                                         prev_tx, prev_ty,
                                         prev_sent_x, prev_sent_y) = \
                                            run_cursor_pipeline(
                                                lm, raw_x_s, raw_y_s,
                                                cx, cy, prev_tx, prev_ty,
                                                prev_sent_x, prev_sent_y,
                                                reentry=reentry)
                                        was_gesturing  = False
                                        active_gesture = f"DRAG  I={pd_it:.3f}"
                                    else:
                                        was_gesturing  = True
                                        active_gesture = f"PINCH  I={pd_it:.3f}"

                                else:
                                    if it_held:
                                        if drag_active:
                                            mouse_up()
                                            drag_active     = False
                                            last_drag_end_t = now
                                        elif (now - it_start_t) < PINCH_COOLDOWN and \
                                                (now - last_drag_end_t) > PINCH_COOLDOWN:
                                            if not _kontrol_is_active(win_name):
                                                mouse_down()
                                                mouse_up()
                                            active_gesture = "L-CLICK"
                                        it_held = False

                                    # ════════════════════════════════════════
                                    # PRIORITY 7 — CURSOR
                                    # ════════════════════════════════════════
                                    reentry = first_frame or was_gesturing
                                    (raw_x_s, raw_y_s, cx, cy,
                                     prev_tx, prev_ty,
                                     prev_sent_x, prev_sent_y) = \
                                        run_cursor_pipeline(
                                            lm, raw_x_s, raw_y_s,
                                            cx, cy, prev_tx, prev_ty,
                                            prev_sent_x, prev_sent_y,
                                            reentry=reentry)
                                    active_gesture = "CURSOR"
                                    was_gesturing  = False

            draw_skeleton(frame, lm, slm, fw_px, fh_px)
            draw_zone_warning(frame, slm, fw_px, fh_px)

        else:
            # ── HAND LOST ─────────────────────────────────────────────────────
            if hand_was_present:
                raw_x_s = raw_y_s = None
                hand_was_present  = False
                reset_smooth()

            if drag_active:
                mouse_up()
                drag_active = False

            it_held = rt_held = mt_held = pt_held = peace_held = False
            scroll_ref_y    = None
            scroll_vel      = 0.0
            palm_frames     = 0
            palm_was_closed = False
            palm_beep_fired = False
            task_wrist_xs.clear()
            tile_fired     = False
            was_gesturing  = True
            active_gesture = "NO HAND"

        draw_hud(
            frame,
            gesture         = active_gesture,
            fps             = fps,
            pd_it           = pd_it,
            pd_mt           = pd_mt,
            pd_rt           = pd_rt,
            pd_pt           = pd_pt,
            palm_count      = palm_frames,
            drag_active     = drag_active,
            tile_held       = pt_held,
            hand_detected   = bool(result.hand_landmarks),
            palm_closing    = palm_frames > 0,
            flash_msg       = flash_msg,
            flash_until     = flash_until,
            detection_phase = detection_phase,
        )
        draw_buttons(frame)
        cv2.imshow(win_name, frame)

        if _quit_flag[0] or (cv2.waitKey(1) & 0xFF == ord("q")):
            print(f"Quit. Detection phase changes: {phase_changes}")
            break

        elapsed   = time.perf_counter() - frame_start
        remaining = FRAME_INTERVAL - elapsed
        if remaining > 0.002:
            time.sleep(remaining)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
