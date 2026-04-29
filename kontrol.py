#!/usr/bin/env python3
"""
Kontrol v1.4 — Hand gesture mouse control
MediaPipe Tasks API (0.10+) → ydotool (no pynput, Wayland-safe)

Gesture priority order (strict — higher number never fires if lower active):
  1. Bunch (all 5 tips together, hold ~1 s, fire on RELEASE) → Show Desktop
  2. Pinky+Thumb (LM 4+20) held + wrist direction            → KDE tile
  3. Three-finger pinch (Thumb+Index+Middle, LM 4+8+12)      → KDE Overview
  4. Peace sign  (index+middle up, ring+pinky folded) + swipe → Alt+Tab task switch
  5. Middle+Thumb (LM 4+12) + vertical wrist movement        → scroll
  6. Ring+Thumb  (LM 4+16)                                   → right click
  7. Index+Thumb (LM 4+8)  hold/tap                          → drag / left click
  8. Index fingertip (LM 8) — 5-stage EMA pipeline           → cursor

KWin D-Bus: org.kde.kglobalaccel.Component.invokeShortcut via /component/kwin
Config: kontrol.conf (INI format, same directory)
Launch: cd /home/K/Storage/Projects/Kontrol && ./run.sh
"""

import configparser
import math
import os
import struct
import subprocess
import tempfile
import time
import wave
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONF_PATH  = Path(__file__).parent / "kontrol.conf"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

_DEFAULTS: dict[str, dict[str, str]] = {
    "screen":   {"width": "4480", "height": "1440"},
    "camera":   {"id": "0", "flip": "true"},
    "mapping":  {
        "zone_x_min": "0.15", "zone_x_max": "0.85",
        "zone_y_min": "0.10", "zone_y_max": "0.90",
    },
    "smoothing": {
        "landmark_smooth": "0.35", "min_smooth": "0.06",
        "max_smooth": "0.40", "velocity_scale": "4.0",
        "cursor_deadzone_px": "2",
    },
    "gestures": {
        "pinch_threshold":        "0.048",
        "pinch_cooldown":         "0.35",
        "three_finger_threshold": "0.058",
        "three_finger_cooldown":  "1.5",
        "scroll_deadzone":        "0.008",
        "scroll_speed":           "6.0",
        "scroll_vel_alpha":       "0.30",
        "scroll_max_ticks":       "8",
        "palm_cooldown":          "2.0",
        "bunch_threshold":        "0.10",
        "bunch_hold_frames":      "12",
        "tile_move_threshold":    "0.050",
        "tile_cooldown":          "0.8",
        "task_move_threshold":    "0.06",
        "task_cooldown":          "0.5",
    },
    "detection": {
        "detection_confidence": "0.50",
        "presence_confidence":  "0.50",
        "tracking_confidence":  "0.50",
    },
    "system": {"ydotool_socket": "/run/user/1000/.ydotool_socket"},
    "camera_tuning": {
        "auto_exposure": "1", "exposure_time_absolute": "300",
        "gain": "100", "brightness": "160", "contrast": "130",
        "saturation": "128", "backlight_compensation": "1",
        "power_line_frequency": "1", "focus_autos": "0", "focus_absolute": "30",
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
SCREEN_W   = _cfg.getint("screen", "width")
SCREEN_H   = _cfg.getint("screen", "height")
SCREEN_DIAG = math.hypot(SCREEN_W, SCREEN_H)

CAM_ID = _cfg.getint("camera", "id")
FLIP   = _cfg.getboolean("camera", "flip")

ZONE_X_MIN = _cfg.getfloat("mapping", "zone_x_min")
ZONE_X_MAX = _cfg.getfloat("mapping", "zone_x_max")
ZONE_Y_MIN = _cfg.getfloat("mapping", "zone_y_min")
ZONE_Y_MAX = _cfg.getfloat("mapping", "zone_y_max")

LANDMARK_SMOOTH    = _cfg.getfloat("smoothing", "landmark_smooth")
MIN_SMOOTH         = _cfg.getfloat("smoothing", "min_smooth")
MAX_SMOOTH         = _cfg.getfloat("smoothing", "max_smooth")
VELOCITY_SCALE     = _cfg.getfloat("smoothing", "velocity_scale")
CURSOR_DEADZONE_PX = _cfg.getint("smoothing",   "cursor_deadzone_px")

PINCH_THRESHOLD       = _cfg.getfloat("gestures", "pinch_threshold")
PINCH_COOLDOWN        = _cfg.getfloat("gestures", "pinch_cooldown")
THREE_FINGER_T        = _cfg.getfloat("gestures", "three_finger_threshold")
THREE_FINGER_COOLDOWN = _cfg.getfloat("gestures", "three_finger_cooldown")
SCROLL_DEADZONE       = _cfg.getfloat("gestures", "scroll_deadzone")
SCROLL_SPEED          = _cfg.getfloat("gestures", "scroll_speed")
SCROLL_VEL_ALPHA      = _cfg.getfloat("gestures", "scroll_vel_alpha")
SCROLL_MAX_TICKS      = _cfg.getint("gestures",   "scroll_max_ticks")
PALM_COOLDOWN         = _cfg.getfloat("gestures", "palm_cooldown")
BUNCH_THRESHOLD       = _cfg.getfloat("gestures", "bunch_threshold")
BUNCH_HOLD_FRAMES     = _cfg.getint("gestures",   "bunch_hold_frames")
TILE_THRESHOLD        = _cfg.getfloat("gestures", "tile_move_threshold")
TILE_COOLDOWN         = _cfg.getfloat("gestures", "tile_cooldown")
TASK_THRESHOLD        = _cfg.getfloat("gestures", "task_move_threshold")
TASK_COOLDOWN         = _cfg.getfloat("gestures", "task_cooldown")

DETECTION_CONF = _cfg.getfloat("detection", "detection_confidence")
PRESENCE_CONF  = _cfg.getfloat("detection", "presence_confidence")
TRACKING_CONF  = _cfg.getfloat("detection", "tracking_confidence")

YDOTOOL_SOCKET = _cfg.get("system", "ydotool_socket")

TARGET_FPS     = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS

os.environ["YDOTOOL_SOCKET"] = YDOTOOL_SOCKET

BaseOptions              = mp.tasks.BaseOptions
HandLandmarker           = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions    = mp.tasks.vision.HandLandmarkerOptions
RunningMode              = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections

# ── VISUALIZATION CONSTANTS ───────────────────────────────────────────────────
FINGER_COLORS = {
    "thumb":  (180,  50, 180),
    "index":  ( 50,  50, 220),
    "middle": ( 50, 180,  50),
    "ring":   ( 50, 200, 200),
    "pinky":  (200, 100,  50),
    "wrist":  (255, 255, 255),
}

LANDMARK_FINGER = {
    0:  "wrist",
    1:  "thumb",  2:  "thumb",  3:  "thumb",  4:  "thumb",
    5:  "index",  6:  "index",  7:  "index",  8:  "index",
    9:  "middle", 10: "middle", 11: "middle", 12: "middle",
    13: "ring",   14: "ring",   15: "ring",   16: "ring",
    17: "pinky",  18: "pinky",  19: "pinky",  20: "pinky",
}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(5,17),
]

SHOW_LM_NUMBERS: bool = False
SHOW_LM_INFO:    bool = False

# ── KWIN D-BUS ────────────────────────────────────────────────────────────────
_KWIN_SHORTCUTS: dict[str, str] = {
    "right":        "Window Quick Tile Right",
    "left":         "Window Quick Tile Left",
    "up":           "Window Quick Tile Top",
    "down":         "Window Quick Tile Bottom",
    "maximize":     "Window Maximize",
    "minimize":     "Window Minimize",
    "task_view":    "Overview",
    "show_desktop": "Show Desktop",
}

_KWIN_KEYCODES: dict[str, tuple[str, ...]] = {
    "right":     ("125:1", "106:1", "106:0", "125:0"),
    "left":      ("125:1", "105:1", "105:0", "125:0"),
    "up":        ("125:1", "103:1", "103:0", "125:0"),
    "down":      ("125:1", "108:1", "108:0", "125:0"),
    "task_view": ("125:1", "104:1", "104:0", "125:0"),
}

KWIN_DBUS: bool = False


def kwin_dbus_available() -> bool:
    try:
        r = subprocess.run(
            ["qdbus", "org.kde.KWin", "/component/kwin"],
            capture_output=True, timeout=1.0,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def kwin_call(shortcut_name: str) -> bool:
    try:
        r = subprocess.run(
            ["qdbus", "org.kde.KWin", "/component/kwin",
             "org.kde.kglobalaccel.Component.invokeShortcut",
             shortcut_name],
            capture_output=True, timeout=0.5,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def fire_kwin(action: str) -> None:
    if KWIN_DBUS and action in _KWIN_SHORTCUTS:
        if kwin_call(_KWIN_SHORTCUTS[action]):
            return
    if action in _KWIN_KEYCODES:
        ydocall("key", *_KWIN_KEYCODES[action], blocking=True)


# ── YDOTOOL HELPERS ───────────────────────────────────────────────────────────
def ydocall(*args, blocking: bool = False) -> None:
    cmd = ["ydotool", *[str(a) for a in args]]
    if blocking:
        subprocess.run(cmd, env=os.environ,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(cmd, env=os.environ,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mouse_down() -> None:
    ydocall("click", "0x40")


def mouse_up() -> None:
    ydocall("click", "0x80")


def right_click() -> None:
    ydocall("click", "0xC1")


def scroll_up(ticks: int = 1) -> None:
    ydocall("mousemove", "--wheel", "-x", "0", "-y", str(-ticks))


def scroll_down(ticks: int = 1) -> None:
    ydocall("mousemove", "--wheel", "-x", "0", "-y", str(ticks))


# ── FOCUS GUARD ───────────────────────────────────────────────────────────────
_focus_cache: list = [0.0, False]


def _kontrol_is_active(win_name: str) -> bool:
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


# ── LANDMARK HELPERS ──────────────────────────────────────────────────────────
def pdist(lm, a: int, b: int) -> float:
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def is_index_thumb(lm) -> bool:
    return pdist(lm, 4, 8) < PINCH_THRESHOLD


def is_middle_thumb(lm) -> bool:
    return pdist(lm, 4, 12) < PINCH_THRESHOLD


def is_ring_thumb(lm) -> bool:
    return pdist(lm, 4, 16) < PINCH_THRESHOLD


def is_pinky_thumb(lm) -> bool:
    return pdist(lm, 4, 20) < PINCH_THRESHOLD


def is_three_finger_pinch(lm, thresh: float) -> bool:
    """Thumb (LM4) close to BOTH index (LM8) AND middle (LM12) simultaneously."""
    return pdist(lm, 4, 8) < thresh and pdist(lm, 4, 12) < thresh


def is_peace_sign(lm) -> bool:
    index_up   = lm[8].y  < lm[5].y
    middle_up  = lm[12].y < lm[9].y
    ring_down  = lm[16].y > lm[13].y
    pinky_down = lm[20].y > lm[17].y
    no_pinch   = (pdist(lm, 4, 8)  > PINCH_THRESHOLD and
                  pdist(lm, 4, 12) > PINCH_THRESHOLD and
                  pdist(lm, 4, 16) > PINCH_THRESHOLD and
                  pdist(lm, 4, 20) > PINCH_THRESHOLD)
    return index_up and middle_up and ring_down and pinky_down and no_pinch


def is_bunch(lm) -> bool:
    return (
        pdist(lm, 4, 8)  < BUNCH_THRESHOLD and
        pdist(lm, 4, 12) < BUNCH_THRESHOLD and
        pdist(lm, 4, 16) < BUNCH_THRESHOLD and
        pdist(lm, 4, 20) < BUNCH_THRESHOLD
    )


# ── CAMERA ────────────────────────────────────────────────────────────────────
def apply_camera_settings() -> None:
    dev = f"/dev/video{CAM_ID}"
    ct  = _cfg["camera_tuning"]

    def v4l2_set(ctrl: str, value: str) -> None:
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", dev, f"--set-ctrl={ctrl}={value}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=2,
            )
            print(f"[CAM] {ctrl} = {value} ✓")
        except Exception as exc:
            print(f"[CAM] {ctrl} SKIPPED ({exc})")

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


# ── DRAW FUNCTIONS ────────────────────────────────────────────────────────────
def lm_radius(lm, i: int, base: int = 4, scale: int = 3) -> int:
    depth_factor = 1.0 - max(-0.5, min(0.5, lm[i].z * 4))
    return max(2, int(base + scale * depth_factor))


def draw_skeleton(frame, lm, fw: int, fh: int, active_pinches: dict) -> None:
    # 1. Connections
    for a, b in HAND_CONNECTIONS:
        color = FINGER_COLORS[LANDMARK_FINGER[max(a, b)]]
        dim   = tuple(int(c * 0.55) for c in color)
        cv2.line(frame,
                 (int(lm[a].x * fw), int(lm[a].y * fh)),
                 (int(lm[b].x * fw), int(lm[b].y * fh)),
                 dim, 1, cv2.LINE_AA)

    # 2. All 21 dots
    for i in range(21):
        px = int(lm[i].x * fw); py = int(lm[i].y * fh)
        color = FINGER_COLORS[LANDMARK_FINGER[i]]
        r     = lm_radius(lm, i)
        cv2.circle(frame, (px, py), r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), r, (220, 220, 220), 1, cv2.LINE_AA)

    # 3. Fingertip highlight rings
    for tip in (4, 8, 12, 16, 20):
        px = int(lm[tip].x * fw); py = int(lm[tip].y * fh)
        color = FINGER_COLORS[LANDMARK_FINGER[tip]]
        r     = lm_radius(lm, tip, base=6, scale=4)
        cv2.circle(frame, (px, py), r + 4, color, 2, cv2.LINE_AA)

    # 4. Wrist
    wx = int(lm[0].x * fw); wy = int(lm[0].y * fh)
    cv2.circle(frame, (wx, wy), 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (wx, wy), 8, (180, 180, 180), 2, cv2.LINE_AA)

    # 5. Dynamic pinch lines
    def draw_pinch_line(a: int, b: int, active_color: tuple) -> None:
        dist = pdist(lm, a, b)
        t    = max(0.0, min(1.0, 1.0 - dist / (PINCH_THRESHOLD * 2)))
        col  = tuple(int(80 * (1 - t) + active_color[c] * t) for c in range(3))
        cv2.line(frame,
                 (int(lm[a].x * fw), int(lm[a].y * fh)),
                 (int(lm[b].x * fw), int(lm[b].y * fh)),
                 col, 1 + int(t * 3), cv2.LINE_AA)

    draw_pinch_line(4, 8,  ( 50,  50, 220))
    draw_pinch_line(4, 12, ( 50, 180,  50))
    draw_pinch_line(4, 20, ( 50, 200, 200))

    # 6. Three-finger triangle
    if active_pinches.get("three_finger"):
        pts = np.array([
            [int(lm[4].x * fw),  int(lm[4].y * fh)],
            [int(lm[8].x * fw),  int(lm[8].y * fh)],
            [int(lm[12].x * fw), int(lm[12].y * fh)],
        ], dtype=np.int32)
        cv2.polylines(frame, [pts], True, (220, 50, 220), 2, cv2.LINE_AA)

    # 7. Landmark numbers (toggle 'n')
    if SHOW_LM_NUMBERS:
        for i in range(21):
            cv2.putText(frame, str(i),
                        (int(lm[i].x * fw) + 4, int(lm[i].y * fh) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)


def draw_lm_info(frame, lm, fw: int, fh: int) -> None:
    tips  = [0, 4, 8, 12, 16, 20]
    names = {0: "WRST", 4: "THMB", 8: "INDX", 12: "MID", 16: "RING", 20: "PNKY"}
    x0    = max(0, fw - 200)
    for row, idx in enumerate(tips):
        label = f"LM{idx:02d} {names[idx]}: {lm[idx].x:.3f},{lm[idx].y:.3f},{lm[idx].z:.3f}"
        cv2.putText(frame, label, (x0, 16 + row * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 220, 255), 1)


def draw_zone_warning(frame, lm, fw: int, fh: int) -> None:
    x, y   = lm[8].x, lm[8].y
    margin = 0.05
    font   = cv2.FONT_HERSHEY_SIMPLEX
    col    = (0, 220, 220)
    if x < ZONE_X_MIN + margin:
        cv2.putText(frame, "< MOVE IN", (4, fh // 2), font, 0.50, col, 1)
    if x > ZONE_X_MAX - margin:
        cv2.putText(frame, "MOVE IN >", (fw - 90, fh // 2), font, 0.50, col, 1)
    if y < ZONE_Y_MIN + margin:
        cv2.putText(frame, "^ MOVE IN", (fw // 2 - 44, 18), font, 0.50, col, 1)
    if y > ZONE_Y_MAX - margin:
        cv2.putText(frame, "v MOVE IN", (fw // 2 - 44, fh - 6), font, 0.50, col, 1)


def draw_hud(frame, gesture: str, fps: float,
             pd_it: float, pd_mt: float, pd_rt: float, pd_pt: float, pd_3f: float,
             palm_count: int = 0,
             drag_active: bool = False,
             tile_held: bool = False,
             three_finger_active: bool = False,
             hand_detected: bool = True,
             palm_closing: bool = False,
             flash_msg: str = "",
             flash_until: float = 0.0,
             flash_color: tuple = (0, 255, 220)) -> None:
    h, w = frame.shape[:2]
    now  = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX

    if not hand_detected:
        border_col = (0, 0, 200)
    elif palm_closing:
        border_col = (0, 120, 255)
    elif three_finger_active:
        border_col = (220, 50, 220)
    elif drag_active:
        border_col = (200, 80, 0)
    elif tile_held:
        border_col = (0, 200, 220)
    else:
        border_col = (30, 200, 50)

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_col, 3)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (290, 108), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    lc = (0, 200, 200)
    vc = (220, 220, 220)
    x0, y0, dy = 8, 15, 20

    def hud_line(row: int, label: str, value: str) -> None:
        cv2.putText(frame, label, (x0, y0 + row * dy), font, 0.42, lc, 1)
        cv2.putText(frame, value, (x0 + 68, y0 + row * dy), font, 0.42, vc, 1)

    hud_line(0, "[FPS]    ", f"{fps:5.1f}")
    hud_line(1, "[HAND]   ", "DETECTED" if hand_detected else "NO HAND")
    hud_line(2, "[GESTURE]", gesture)
    hud_line(3, "[PINCH]  ", f"I={pd_it:.3f} M={pd_mt:.3f} 3F={pd_3f:.3f}")
    hud_line(4, "         ", f"P={pd_pt:.3f} R={pd_rt:.3f}")

    if palm_count > 0:
        BAR_W  = 10
        filled = min(palm_count * BAR_W // BUNCH_HOLD_FRAMES, BAR_W)
        bar    = "█" * filled + "░" * (BAR_W - filled)
        cv2.putText(frame, f"[PALM] {bar} {palm_count}/{BUNCH_HOLD_FRAMES}",
                    (x0, y0 + 5 * dy), font, 0.40, (0, 180, 255), 1)

    if flash_msg and now < flash_until:
        text_sz = cv2.getTextSize(flash_msg, font, 1.2, 2)[0]
        tx = (w - text_sz[0]) // 2
        ty = h // 2
        cv2.putText(frame, flash_msg, (tx, ty), font, 1.2, flash_color, 2)


# ── BUTTONS ───────────────────────────────────────────────────────────────────
_quit_flag             = [False]
_btn_close: tuple | None = None
_btn_min:   tuple | None = None


def _mouse_cb(event, x, y, flags, param) -> None:
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
    global _btn_close, _btn_min
    h, w        = frame.shape[:2]
    pad, bw, bh = 5, 28, 20

    cx1 = w - pad - bw; cy1 = pad; cx2 = w - pad; cy2 = pad + bh
    _btn_close = (cx1, cy1, cx2, cy2)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (40, 40, 150), -1)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (140, 140, 140), 1)
    cv2.putText(frame, "X", (cx1 + 8, cy2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    mx1 = cx1 - pad - bw; my1 = pad; mx2 = cx1 - pad; my2 = pad + bh
    _btn_min = (mx1, my1, mx2, my2)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (25, 90, 25), -1)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (140, 140, 140), 1)
    cv2.putText(frame, "-", (mx1 + 9, my2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


# ── STARTUP SOUND ─────────────────────────────────────────────────────────────
def play_startup_sound() -> None:
    rate, amp = 22050, 28000
    dur, gap  = 0.10, 0.04
    freqs     = (880, 1047, 1319)

    def burst(freq: float, duration: float) -> bytes:
        n   = int(rate * duration)
        out = bytearray(n * 2)
        for i in range(n):
            t   = i / rate
            env = min(t / (duration * 0.1), 1.0, (duration - t) / (duration * 0.1))
            struct.pack_into("<h", out, i * 2,
                             int(amp * env * math.sin(2.0 * math.pi * freq * t)))
        return bytes(out)

    silence = bytes(int(rate * gap) * 2)
    pcm     = silence.join(burst(f, dur) for f in freqs)
    tmp     = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(pcm)
    tmp.close()
    subprocess.Popen(["aplay", "-q", tmp.name],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── CURSOR PIPELINE ───────────────────────────────────────────────────────────
def run_cursor_pipeline(
    lm,
    raw_x_s: float | None, raw_y_s: float | None,
    cx: float, cy: float,
    prev_tx: float, prev_ty: float,
    prev_sent_x: int, prev_sent_y: int,
    reentry: bool,
) -> tuple:
    lx, ly = lm[8].x, lm[8].y

    if raw_x_s is None:
        raw_x_s, raw_y_s = lx, ly
    raw_x_s += (lx - raw_x_s) * LANDMARK_SMOOTH
    raw_y_s += (ly - raw_y_s) * LANDMARK_SMOOTH

    nx = max(0.0, min(1.0, (raw_x_s - ZONE_X_MIN) / (ZONE_X_MAX - ZONE_X_MIN)))
    ny = max(0.0, min(1.0, (raw_y_s - ZONE_Y_MIN) / (ZONE_Y_MAX - ZONE_Y_MIN)))
    tx = nx * SCREEN_W
    ty = ny * SCREEN_H

    if reentry:
        cx, cy = tx, ty
        prev_tx, prev_ty = tx, ty
        prev_sent_x, prev_sent_y = int(tx), int(ty)
        return raw_x_s, raw_y_s, cx, cy, prev_tx, prev_ty, prev_sent_x, prev_sent_y

    vel_px   = math.hypot(tx - prev_tx, ty - prev_ty)
    vel_norm = vel_px / SCREEN_DIAG
    prev_tx, prev_ty = tx, ty

    alpha = max(MIN_SMOOTH, min(MAX_SMOOTH, vel_norm * VELOCITY_SCALE))
    cx += (tx - cx) * alpha
    cy += (ty - cy) * alpha

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
    global KWIN_DBUS, SHOW_LM_NUMBERS, SHOW_LM_INFO

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Download:\n  curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

    KWIN_DBUS = kwin_dbus_available()
    print(f"[KWIN] D-Bus {'available ✓' if KWIN_DBUS else 'unavailable, using keycodes'}")

    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=DETECTION_CONF,
        min_hand_presence_confidence=PRESENCE_CONF,
        min_tracking_confidence=TRACKING_CONF,
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

    win_name = "Kontrol v1.4"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, _mouse_cb)

    # ── CURSOR STATE ──────────────────────────────────────────────────────────
    raw_x_s: float | None = None
    raw_y_s: float | None = None
    cx:  float = SCREEN_W / 2
    cy:  float = SCREEN_H / 2
    prev_tx:     float = cx
    prev_ty:     float = cy
    prev_sent_x: int   = int(cx)
    prev_sent_y: int   = int(cy)
    hand_was_present: bool = False
    was_gesturing:    bool = False

    # ── GESTURE STATE — bunch (priority 1) ───────────────────────────────────
    palm_frames:     int   = 0
    palm_was_closed: bool  = False
    last_palm_t:     float = 0.0
    palm_minimized:  bool  = False

    # ── GESTURE STATE — pinky+thumb tiling (priority 2) ──────────────────────
    pt_held:      bool  = False
    tile_start_x: float = 0.0
    tile_start_y: float = 0.0
    tile_fired:   bool  = False
    last_tile_t:  float = 0.0

    # ── GESTURE STATE — three-finger pinch (priority 3) ──────────────────────
    three_finger_held:   bool  = False
    last_three_finger_t: float = 0.0

    # ── GESTURE STATE — peace sign (priority 4) ───────────────────────────────
    peace_held:    bool  = False
    task_wrist_xs: deque = deque(maxlen=8)
    last_task_t:   float = 0.0

    # ── GESTURE STATE — middle+thumb scroll (priority 5) ─────────────────────
    mt_held:      bool         = False
    scroll_ref_y: float | None = None
    scroll_vel:   float        = 0.0

    # ── GESTURE STATE — ring+thumb right click (priority 6) ──────────────────
    rt_held:       bool  = False
    last_rclick_t: float = 0.0

    # ── GESTURE STATE — index+thumb click/drag (priority 7) ──────────────────
    it_held:         bool  = False
    it_start_t:      float = 0.0
    drag_active:     bool  = False
    last_drag_end_t: float = 0.0

    # ── HUD STATE ─────────────────────────────────────────────────────────────
    active_gesture: str   = "NONE"
    flash_msg:      str   = ""
    flash_until:    float = 0.0
    flash_color:    tuple = (0, 255, 220)
    fps:            float = 0.0
    fps_alpha:      float = 0.1
    last_frame_t:   float = time.time()
    pd_it: float = 1.0
    pd_mt: float = 1.0
    pd_rt: float = 1.0
    pd_pt: float = 1.0
    pd_3f: float = 1.0

    print(
        f"Kontrol v1.4  {SCREEN_W}x{SCREEN_H}"
        f"  cam=/dev/video{CAM_ID}"
        f"  pinch={PINCH_THRESHOLD}  3f={THREE_FINGER_T}"
        f"  bunch={BUNCH_THRESHOLD}/{BUNCH_HOLD_FRAMES}fr"
        f"  [n]=LM nums  [i]=LM info  [q]=quit"
    )

    _last_ts_ms: int = 0

    with HandLandmarker.create_from_options(opts) as detector:
        while cap.isOpened():
            frame_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                continue

            now = time.time()
            dt  = now - last_frame_t
            last_frame_t = now
            if dt > 0:
                fps = fps * (1.0 - fps_alpha) + (1.0 / dt) * fps_alpha

            _ts_ms      = max(int(now * 1000), _last_ts_ms + 1)
            _last_ts_ms = _ts_ms

            if FLIP:
                frame = cv2.flip(frame, 1)

            fh_px, fw_px = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect_for_video(mp_image, _ts_ms)

            three_finger_active = False

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]

                first_frame      = not hand_was_present
                hand_was_present = True

                pd_it = pdist(lm, 4, 8)
                pd_mt = pdist(lm, 4, 12)
                pd_rt = pdist(lm, 4, 16)
                pd_pt = pdist(lm, 4, 20)
                pd_3f = min(pd_it, pd_mt)

                task_wrist_xs.append(lm[0].x)

                # ════════════════════════════════════════════════════════════
                # PRIORITY 1 — BUNCH → SHOW DESKTOP
                # ════════════════════════════════════════════════════════════
                if is_bunch(lm):
                    palm_frames    += 1
                    palm_was_closed = True
                    was_gesturing   = True

                    if drag_active:
                        mouse_up()
                        drag_active = False

                    active_gesture = f"PALM {palm_frames}/{BUNCH_HOLD_FRAMES}"

                else:
                    if palm_was_closed:
                        if (palm_frames >= BUNCH_HOLD_FRAMES
                                and (now - last_palm_t) > PALM_COOLDOWN):
                            subprocess.run(
                                ["qdbus", "org.kde.kglobalaccel", "/component/kwin",
                                 "org.kde.kglobalaccel.Component.invokeShortcut",
                                 "Show Desktop"],
                                env=os.environ,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            )
                            palm_minimized = not palm_minimized
                            flash_msg   = "SHOW DESKTOP" if palm_minimized else "RESTORE"
                            flash_color = (0, 255, 220)
                            flash_until = now + 1.0
                            last_palm_t = now
                        palm_frames     = 0
                        palm_was_closed = False

                    # ════════════════════════════════════════════════════════
                    # PRIORITY 2 — PINKY+THUMB → TILE
                    # ════════════════════════════════════════════════════════
                    if is_pinky_thumb(lm):
                        active_gesture = f"TILE HOLD  P={pd_pt:.3f}"
                        was_gesturing  = True

                        if not pt_held:
                            tile_start_x = lm[0].x
                            tile_start_y = lm[0].y
                            tile_fired   = False
                        pt_held = True

                        if not tile_fired:
                            dx_total = lm[0].x - tile_start_x
                            dy_total = lm[0].y - tile_start_y
                            adx, ady = abs(dx_total), abs(dy_total)

                            if (adx > TILE_THRESHOLD or ady > TILE_THRESHOLD) \
                                    and (now - last_tile_t) > TILE_COOLDOWN:
                                direction = (
                                    ("right" if dx_total > 0 else "left") if adx > ady
                                    else ("down" if dy_total > 0 else "up")
                                )
                                fire_kwin(direction)
                                tile_fired     = True
                                last_tile_t    = now
                                flash_msg      = f"TILE {direction.upper()}"
                                flash_color    = (0, 255, 220)
                                flash_until    = now + 1.0
                                active_gesture = f"TILE {direction.upper()}"

                    else:
                        if pt_held:
                            tile_fired = False
                            pt_held    = False

                        # ════════════════════════════════════════════════════
                        # PRIORITY 3 — THREE-FINGER PINCH → OVERVIEW
                        # Must be checked BEFORE middle+thumb (superset)
                        # ════════════════════════════════════════════════════
                        if is_three_finger_pinch(lm, THREE_FINGER_T):
                            three_finger_active = True
                            was_gesturing       = True

                            if (not three_finger_held
                                    and (now - last_three_finger_t) > THREE_FINGER_COOLDOWN):
                                fire_kwin("task_view")
                                last_three_finger_t = now
                                flash_msg   = "TASK VIEW"
                                flash_color = (220, 50, 220)
                                flash_until = now + 1.0
                            three_finger_held = True
                            active_gesture    = "TASK VIEW"

                        else:
                            three_finger_held = False

                            # ════════════════════════════════════════════════
                            # PRIORITY 4 — PEACE SIGN → ALT+TAB
                            # ════════════════════════════════════════════════
                            if is_peace_sign(lm):
                                active_gesture = "TASK SWITCH"
                                was_gesturing  = True

                                if not peace_held:
                                    task_wrist_xs.clear()
                                    peace_held = True

                                if len(task_wrist_xs) >= 2 \
                                        and (now - last_task_t) > TASK_COOLDOWN:
                                    dx = task_wrist_xs[-1] - task_wrist_xs[0]
                                    if abs(dx) > TASK_THRESHOLD:
                                        if dx > 0:
                                            ydocall("key", "56:1", "15:1", "15:0", "56:0",
                                                    blocking=True)
                                            active_gesture = "TASK →"
                                        else:
                                            ydocall("key", "56:1", "42:1", "15:1",
                                                    "15:0", "42:0", "56:0", blocking=True)
                                            active_gesture = "TASK ←"
                                        last_task_t = now
                                        task_wrist_xs.clear()

                            else:
                                if peace_held:
                                    peace_held = False
                                    task_wrist_xs.clear()

                                # ════════════════════════════════════════════
                                # PRIORITY 5 — MIDDLE+THUMB → SCROLL
                                # ════════════════════════════════════════════
                                if is_middle_thumb(lm):
                                    was_gesturing = True

                                    if not mt_held:
                                        mt_held      = True
                                        scroll_ref_y = lm[0].y
                                        scroll_vel   = 0.0

                                    dy_norm      = lm[0].y - scroll_ref_y
                                    scroll_ref_y = lm[0].y
                                    scroll_vel   = (scroll_vel * (1.0 - SCROLL_VEL_ALPHA)
                                                    + dy_norm * SCROLL_VEL_ALPHA)

                                    if abs(scroll_vel) > SCROLL_DEADZONE:
                                        ticks   = max(1, min(SCROLL_MAX_TICKS,
                                                             int(abs(scroll_vel) * SCROLL_SPEED)))
                                        wheel_y = -ticks if scroll_vel < 0 else ticks
                                        ydocall("mousemove", "--wheel",
                                                "-x", "0", "-y", str(wheel_y))
                                        active_gesture = f"SCROLL {'UP' if scroll_vel < 0 else 'DOWN'} x{ticks}"
                                    else:
                                        active_gesture = f"SCROLL HOLD  M={pd_mt:.3f}"

                                else:
                                    if mt_held:
                                        mt_held      = False
                                        scroll_ref_y = None
                                        scroll_vel   = 0.0

                                    # ════════════════════════════════════════
                                    # PRIORITY 6 — RING+THUMB → RIGHT CLICK
                                    # ════════════════════════════════════════
                                    if is_ring_thumb(lm):
                                        if not rt_held \
                                                and (now - last_rclick_t) > PINCH_COOLDOWN:
                                            if not _kontrol_is_active(win_name):
                                                right_click()
                                            last_rclick_t  = now
                                            active_gesture = f"R-CLICK  R={pd_rt:.3f}"
                                        rt_held       = True
                                        was_gesturing = True

                                    else:
                                        rt_held = False

                                        # ════════════════════════════════════
                                        # PRIORITY 7 — INDEX+THUMB → CLICK/DRAG
                                        # ════════════════════════════════════
                                        if is_index_thumb(lm):
                                            if not it_held:
                                                it_held    = True
                                                it_start_t = now

                                            held_t = now - it_start_t
                                            if (held_t > PINCH_COOLDOWN
                                                    and not drag_active
                                                    and (now - last_drag_end_t) > PINCH_COOLDOWN):
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
                                                elif ((now - it_start_t) < PINCH_COOLDOWN
                                                      and (now - last_drag_end_t) > PINCH_COOLDOWN):
                                                    mouse_down()
                                                    mouse_up()
                                                    active_gesture = "L-CLICK"
                                                it_held = False

                                            # ════════════════════════════════
                                            # PRIORITY 8 — CURSOR
                                            # ════════════════════════════════
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

                draw_skeleton(frame, lm, fw_px, fh_px, {
                    "three_finger": three_finger_active,
                })
                draw_zone_warning(frame, lm, fw_px, fh_px)
                if SHOW_LM_INFO:
                    draw_lm_info(frame, lm, fw_px, fh_px)

            else:
                if hand_was_present:
                    raw_x_s = raw_y_s = None
                    hand_was_present  = False

                if drag_active:
                    mouse_up()
                    drag_active = False

                it_held           = False
                rt_held           = False
                mt_held           = False
                pt_held           = False
                peace_held        = False
                three_finger_held = False
                scroll_ref_y      = None
                scroll_vel        = 0.0
                palm_frames       = 0
                palm_was_closed   = False
                task_wrist_xs.clear()
                tile_fired        = False
                was_gesturing     = True
                active_gesture    = "NO HAND"

            draw_hud(
                frame,
                gesture             = active_gesture,
                fps                 = fps,
                pd_it               = pd_it,
                pd_mt               = pd_mt,
                pd_rt               = pd_rt,
                pd_pt               = pd_pt,
                pd_3f               = pd_3f,
                palm_count          = palm_frames,
                drag_active         = drag_active,
                tile_held           = pt_held,
                three_finger_active = three_finger_active,
                hand_detected       = bool(result.hand_landmarks),
                palm_closing        = palm_frames > 0,
                flash_msg           = flash_msg,
                flash_until         = flash_until,
                flash_color         = flash_color,
            )
            draw_buttons(frame)
            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if _quit_flag[0] or key == ord("q"):
                print("Quit.")
                break
            elif key == ord("n"):
                SHOW_LM_NUMBERS = not SHOW_LM_NUMBERS
                print(f"[HUD] landmark numbers {'ON' if SHOW_LM_NUMBERS else 'OFF'}")
            elif key == ord("i"):
                SHOW_LM_INFO = not SHOW_LM_INFO
                print(f"[HUD] LM info overlay {'ON' if SHOW_LM_INFO else 'OFF'}")

            elapsed   = time.perf_counter() - frame_start
            remaining = FRAME_INTERVAL - elapsed
            if remaining > 0.002:
                time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
