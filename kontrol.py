#!/usr/bin/env python3
"""
Kontrol v1.1 — Hand gesture mouse control
MediaPipe Tasks API (0.10+) → ydotool (no pynput, Wayland-safe)

Gesture priority order:
  1. Palm close (hold palm_hold_frames, fire on release) → minimize / restore toggle
  2. Tracking lock (legacy state — no entry gesture in v1.1)
  3. Pinky+Thumb held + wrist direction                  → KDE window tiling
  4. Middle+Thumb pinched + vertical wrist move          → scroll
  5. Middle+Thumb pinched hold/tap                       → drag / left click
  6. Index+Thumb pinch                                   → right click
  7. Index fingertip (LM 8) — double EMA                → cursor

Config: kontrol.conf (INI format, same directory)
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
        "width":  "4480",
        "height": "1440",
    },
    "camera": {
        "id":   "2",
        "flip": "true",
    },
    "mapping": {
        "zone_x_min": "0.15",
        "zone_x_max": "0.85",
        "zone_y_min": "0.10",
        "zone_y_max": "0.90",
    },
    "smoothing": {
        "landmark_smooth":    "0.35",
        "min_smooth":         "0.06",
        "max_smooth":         "0.40",
        "velocity_scale":     "4.0",
        "cursor_deadzone_px": "2",
    },
    "gestures": {
        "pinch_threshold":      "0.055",
        "pinch_cooldown":       "0.35",
        "scroll_deadzone":      "0.010",
        "scroll_speed":         "6.0",
        "palm_hold_frames":     "20",
        "palm_cooldown":        "2.0",
        "tile_move_threshold":  "0.06",
        "tile_window_frames":   "8",
        "tile_cooldown":        "0.8",
    },
    "system": {
        "ydotool_socket": "/run/user/1000/.ydotool_socket",
    },
    "camera_tuning": {
        "auto_exposure":           "1",
        "exposure_time_absolute":  "300",
        "gain":                    "100",
        "brightness":              "160",
        "contrast":                "130",
        "saturation":              "128",
        "sharpness":               "128",
        "backlight_compensation":  "1",
        "power_line_frequency":    "1",
        "focus_autos":             "0",
        "focus_absolute":          "30",
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

SCREEN_W            = _cfg.getint("screen",    "width")
SCREEN_H            = _cfg.getint("screen",    "height")
CAM_ID              = _cfg.getint("camera",    "id")
FLIP                = _cfg.getboolean("camera","flip")
ZONE_X_MIN          = _cfg.getfloat("mapping", "zone_x_min")
ZONE_X_MAX          = _cfg.getfloat("mapping", "zone_x_max")
ZONE_Y_MIN          = _cfg.getfloat("mapping", "zone_y_min")
ZONE_Y_MAX          = _cfg.getfloat("mapping", "zone_y_max")
LANDMARK_SMOOTH     = _cfg.getfloat("smoothing", "landmark_smooth")
MIN_SMOOTH          = _cfg.getfloat("smoothing", "min_smooth")
MAX_SMOOTH          = _cfg.getfloat("smoothing", "max_smooth")
VELOCITY_SCALE      = _cfg.getfloat("smoothing", "velocity_scale")
CURSOR_DEADZONE_PX  = _cfg.getint("smoothing",   "cursor_deadzone_px")
PINCH_THRESHOLD     = _cfg.getfloat("gestures",  "pinch_threshold")
PINCH_COOLDOWN      = _cfg.getfloat("gestures",  "pinch_cooldown")
SCROLL_DEADZONE     = _cfg.getfloat("gestures",  "scroll_deadzone")
SCROLL_SPEED        = _cfg.getfloat("gestures",  "scroll_speed")
PALM_HOLD_FRAMES    = _cfg.getint("gestures",    "palm_hold_frames")
PALM_COOLDOWN       = _cfg.getfloat("gestures",  "palm_cooldown")
TILE_MOVE_THRESHOLD = _cfg.getfloat("gestures",  "tile_move_threshold")
TILE_WINDOW_FRAMES  = _cfg.getint("gestures",    "tile_window_frames")
TILE_COOLDOWN       = _cfg.getfloat("gestures",  "tile_cooldown")
YDOTOOL_SOCKET      = _cfg.get("system",         "ydotool_socket")

SCREEN_DIAG = math.hypot(SCREEN_W, SCREEN_H)


# ── Camera v4l2 tuning ────────────────────────────────────────────────────────
def apply_camera_settings():
    """
    Apply v4l2 camera settings from kontrol.conf [camera_tuning].
    Call AFTER cv2.VideoCapture opens and FOURCC/resolution are set.
    """
    dev = f"/dev/video{CAM_ID}"
    ct  = _cfg["camera_tuning"]

    def v4l2_set(ctrl: str, value: str):
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", dev, f"--set-ctrl={ctrl}={value}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=2,
            )
            print(f"[CAM] {ctrl} = {value} \u2713")
        except Exception as exc:
            print(f"[CAM] {ctrl} SKIPPED ({exc})")

    # Order matters: auto_exposure=1 must precede exposure_time_absolute;
    # focus_automatic_continuous=0 must precede focus_absolute.
    v4l2_set("auto_exposure",              ct.get("auto_exposure",            "1"))
    v4l2_set("exposure_time_absolute",     ct.get("exposure_time_absolute",   "300"))
    v4l2_set("gain",                       ct.get("gain",                     "100"))
    v4l2_set("brightness",                 ct.get("brightness",               "160"))
    v4l2_set("contrast",                   ct.get("contrast",                 "130"))
    v4l2_set("sharpness",                  ct.get("sharpness",                "128"))
    v4l2_set("saturation",                 ct.get("saturation",               "128"))
    v4l2_set("backlight_compensation",     ct.get("backlight_compensation",   "1"))
    v4l2_set("power_line_frequency",       ct.get("power_line_frequency",     "1"))
    v4l2_set("focus_automatic_continuous", ct.get("focus_autos",              "0"))
    v4l2_set("focus_absolute",             ct.get("focus_absolute",           "30"))


# ── Camera control panel ──────────────────────────────────────────────────────
class CamPanel:
    """Overlay panel for live v4l2 control. Toggle with the CAM button."""

    CONTROLS = [
        ("brightness",      "Brightness",   0,   255,   5),
        ("contrast",        "Contrast",     0,   255,   5),
        ("sharpness",       "Sharpness",    0,   255,   5),
        ("focus_absolute",  "Focus",        0,   250,   5),
        ("gain",            "ISO / Gain",   0,   255,   5),
    ]

    _PW    = 215
    _PH    = 152
    _ROW_H = 24
    _PAD   = 8
    _BW    = 16

    _DEFAULTS = {
        "brightness":     160,
        "contrast":       130,
        "sharpness":      128,
        "focus_absolute": 30,
        "gain":           100,
    }

    def __init__(self, cfg: configparser.ConfigParser):
        ct = cfg["camera_tuning"]
        self.values: dict[str, int] = {
            "brightness":     int(ct.get("brightness",     "160")),
            "contrast":       int(ct.get("contrast",       "130")),
            "sharpness":      int(ct.get("sharpness",      "128")),
            "focus_absolute": int(ct.get("focus_absolute", "30")),
            "gain":           int(ct.get("gain",           "100")),
        }
        self.visible = False
        self._btns: dict[tuple[str, str], tuple[int, int, int, int]] = {}

    def toggle(self):
        self.visible = not self.visible

    def _apply(self, ctrl: str, value: int):
        dev = f"/dev/video{CAM_ID}"
        subprocess.Popen(
            ["v4l2-ctl", "-d", dev, f"--set-ctrl={ctrl}={value}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def handle_click(self, x: int, y: int) -> bool:
        if not self.visible:
            return False
        for (ctrl, sign), (x1, y1, x2, y2) in self._btns.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                for key, _, mn, mx, step in self.CONTROLS:
                    if key == ctrl:
                        delta = step if sign == "+" else -step
                        self.values[ctrl] = max(mn, min(mx, self.values[ctrl] + delta))
                        self._apply(ctrl, self.values[ctrl])
                        return True
        return False

    def draw(self, frame):
        if not self.visible:
            return
        fh, fw = frame.shape[:2]
        px = fw - self._PW - 2
        py = 30

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + self._PW, py + self._PH),
                      (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        cv2.rectangle(frame, (px, py), (px + self._PW, py + self._PH),
                      (70, 150, 70), 1)
        cv2.putText(frame, "CAM CONTROLS",
                    (px + self._PAD, py + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 210, 90), 1)
        cv2.line(frame,
                 (px, py + 19), (px + self._PW, py + 19),
                 (50, 100, 50), 1)

        self._btns.clear()
        for i, (ctrl, label, mn, mx, _) in enumerate(self.CONTROLS):
            ry  = py + 26 + i * self._ROW_H
            val = self.values[ctrl]

            cv2.putText(frame, label,
                        (px + self._PAD, ry + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (195, 195, 195), 1)

            bm_x1, bm_x2 = px + 112, px + 112 + self._BW
            bm_y1, bm_y2 = ry + 1, ry + self._ROW_H - 3
            cv2.rectangle(frame, (bm_x1, bm_y1), (bm_x2, bm_y2), (55, 55, 115), -1)
            cv2.rectangle(frame, (bm_x1, bm_y1), (bm_x2, bm_y2), (110, 110, 175), 1)
            cv2.putText(frame, "-", (bm_x1 + 4, bm_y2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)
            self._btns[(ctrl, "-")] = (bm_x1, bm_y1, bm_x2, bm_y2)

            at_default = (val == self._DEFAULTS.get(ctrl, -1))
            val_col    = (175, 175, 175) if at_default else (80, 210, 255)
            cv2.putText(frame, f"{val:3d}",
                        (bm_x2 + 5, ry + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, val_col, 1)

            bp_x1, bp_x2 = bm_x2 + 36, bm_x2 + 36 + self._BW
            bp_y1, bp_y2 = ry + 1, ry + self._ROW_H - 3
            cv2.rectangle(frame, (bp_x1, bp_y1), (bp_x2, bp_y2), (25, 90, 25), -1)
            cv2.rectangle(frame, (bp_x1, bp_y1), (bp_x2, bp_y2), (70, 165, 70), 1)
            cv2.putText(frame, "+", (bp_x1 + 3, bp_y2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 240, 160), 1)
            self._btns[(ctrl, "+")] = (bp_x1, bp_y1, bp_x2, bp_y2)


_cam_panel = CamPanel(_cfg)

# Raw Linux keycodes (input-event-codes.h)
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


# ── ydotool helpers ───────────────────────────────────────────────────────────
def ydocall(*args, blocking: bool = False):
    """
    Issue a ydotool command.
    blocking=False (default) — Popen fire-and-forget (cursor / click calls).
    blocking=True            — subprocess.run, wait for completion (key seqs).
    """
    cmd = ["ydotool", *[str(a) for a in args]]
    if blocking:
        subprocess.run(cmd, env=os.environ,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(cmd, env=os.environ,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mouse_down():  ydocall("click", "0x40")
def mouse_up():    ydocall("click", "0x80")


def left_click():
    ydocall("click", "0x40")
    ydocall("click", "0x80")


def right_click(): ydocall("click", "0xC1")


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


def is_index_thumb_pinched(lm) -> bool:
    return pinch_dist(lm, 4, 8) < PINCH_THRESHOLD


def is_middle_thumb_pinched(lm) -> bool:
    return pinch_dist(lm, 4, 12) < PINCH_THRESHOLD


def is_pinky_thumb_pinched(lm) -> bool:
    return pinch_dist(lm, 4, 20) < PINCH_THRESHOLD


def is_palm_closed(lm) -> bool:
    """All 5 fingers fully curled — tips below their MCP/IP joints."""
    return (lm[4].y  > lm[2].y  and   # thumb tip below IP
            lm[8].y  > lm[5].y  and   # index tip below MCP
            lm[12].y > lm[9].y  and   # middle tip below MCP
            lm[16].y > lm[13].y and   # ring tip below MCP
            lm[20].y > lm[17].y)      # pinky tip below MCP


# ── Cursor pipeline (shared between cursor mode and drag mode) ─────────────
def _run_cursor_pipeline(lm, raw_x_s, raw_y_s,
                         cx, cy, prev_tx, prev_ty,
                         prev_sent_x, prev_sent_y,
                         reentry: bool):
    """
    Runs the 5-stage cursor pipeline.
    Returns updated (raw_x_s, raw_y_s, cx, cy, prev_tx, prev_ty,
                     prev_sent_x, prev_sent_y).
    reentry=True: snap EMA to current position, send no delta.
    """
    lx, ly = lm[8].x, lm[8].y

    # Stage 1 — landmark pre-smooth
    if raw_x_s is None:
        raw_x_s, raw_y_s = lx, ly
    raw_x_s = raw_x_s + (lx - raw_x_s) * LANDMARK_SMOOTH
    raw_y_s = raw_y_s + (ly - raw_y_s) * LANDMARK_SMOOTH

    # Stage 2 — zone mapping
    nx = max(0.0, min(1.0, (raw_x_s - ZONE_X_MIN) / (ZONE_X_MAX - ZONE_X_MIN)))
    ny = max(0.0, min(1.0, (raw_y_s - ZONE_Y_MIN) / (ZONE_Y_MAX - ZONE_Y_MIN)))
    tx = nx * SCREEN_W
    ty = ny * SCREEN_H

    # Stage 3 — velocity + re-entry guard
    if reentry:
        cx, cy = tx, ty
        prev_tx, prev_ty = tx, ty
        prev_sent_x, prev_sent_y = int(tx), int(ty)
        return raw_x_s, raw_y_s, cx, cy, prev_tx, prev_ty, prev_sent_x, prev_sent_y

    vel_px   = math.hypot(tx - prev_tx, ty - prev_ty)
    vel_norm = vel_px / SCREEN_DIAG
    smooth   = max(MIN_SMOOTH, min(MAX_SMOOTH, vel_norm * VELOCITY_SCALE))
    prev_tx, prev_ty = tx, ty

    # Stage 4 — screen-space EMA
    cx = cx + (tx - cx) * smooth
    cy = cy + (ty - cy) * smooth

    # Stage 5 — integer delta with deadzone
    dx = round(cx - prev_sent_x)
    dy = round(cy - prev_sent_y)
    if max(abs(dx), abs(dy)) >= CURSOR_DEADZONE_PX:
        subprocess.Popen(
            ["ydotool", "mousemove", "-x", str(dx), "-y", str(dy)],
            env=os.environ,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        prev_sent_x += dx
        prev_sent_y += dy

    return raw_x_s, raw_y_s, cx, cy, prev_tx, prev_ty, prev_sent_x, prev_sent_y


# ── Startup sound ─────────────────────────────────────────────────────────────
def play_startup_sound():
    rate, amp = 22050, 28000
    dur, gap  = 0.10, 0.04
    freqs     = (880, 1047, 1319)

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
_quit_flag               = [False]
_btn_close: tuple | None = None
_btn_min:   tuple | None = None
_btn_cam:   tuple | None = None


def _mouse_cb(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if _btn_close and _btn_close[0] <= x <= _btn_close[2] and _btn_close[1] <= y <= _btn_close[3]:
        _quit_flag[0] = True
    elif _btn_min and _btn_min[0] <= x <= _btn_min[2] and _btn_min[1] <= y <= _btn_min[3]:
        subprocess.Popen(
            ["wmctrl", "-r", ":ACTIVE:", "-b", "add,hidden"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    elif _btn_cam and _btn_cam[0] <= x <= _btn_cam[2] and _btn_cam[1] <= y <= _btn_cam[3]:
        _cam_panel.toggle()
    else:
        _cam_panel.handle_click(x, y)


def draw_buttons(frame):
    global _btn_close, _btn_min, _btn_cam
    h, w = frame.shape[:2]
    pad, bw, bh = 5, 24, 18
    cx1, cy1, cx2, cy2 = w - pad - bw, pad, w - pad, pad + bh
    _btn_close = (cx1, cy1, cx2, cy2)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (40, 40, 160), -1)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (160, 160, 160), 1)
    cv2.putText(frame, "X", (cx1 + 6, cy2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    mx1, my1, mx2, my2 = cx1 - pad - bw, pad, cx1 - pad, pad + bh
    _btn_min = (mx1, my1, mx2, my2)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (30, 100, 30), -1)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (160, 160, 160), 1)
    cv2.putText(frame, "-", (mx1 + 8, my2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cam_w = 30
    camx1, camy1 = mx1 - pad - cam_w, pad
    camx2, camy2 = mx1 - pad, pad + bh
    _btn_cam = (camx1, camy1, camx2, camy2)
    cam_bg = (55, 130, 55) if _cam_panel.visible else (20, 50, 20)
    cv2.rectangle(frame, (camx1, camy1), (camx2, camy2), cam_bg, -1)
    cv2.rectangle(frame, (camx1, camy1), (camx2, camy2), (120, 180, 120), 1)
    cv2.putText(frame, "CAM", (camx1 + 3, camy2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 240, 200), 1)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_skeleton(frame, lm, fw: int, fh: int):
    for c in HandLandmarksConnections.HAND_CONNECTIONS:
        ax, ay = int(lm[c.start].x * fw), int(lm[c.start].y * fh)
        bx, by = int(lm[c.end].x   * fw), int(lm[c.end].y   * fh)
        cv2.line(frame, (ax, ay), (bx, by), (60, 160, 60), 1)
    for lmk in lm:
        cv2.circle(frame, (int(lmk.x * fw), int(lmk.y * fh)), 3, (100, 200, 100), -1)


def draw_visibility_warning(frame, lm, fw: int, fh: int):
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
             pd_it: float, pd_mt: float, pd_pt: float,
             palm_count: int = 0,
             drag_active: bool = False,
             tile_held: bool = False,
             flash_msg: str = "",
             flash_until: float = 0.0):
    h, w  = frame.shape[:2]
    now   = time.time()

    # Border colour: Blue=drag, Yellow=tile, Red=locked, Green=tracking
    if drag_active:
        border_color = (200, 80, 0)    # BGR blue
    elif tile_held:
        border_color = (0, 200, 255)   # BGR yellow
    elif locked:
        border_color = (0, 0, 200)     # BGR red
    else:
        border_color = (30, 190, 50)   # BGR green

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 3)

    mode_txt = "LOCKED" if locked else "TRACKING"
    cv2.putText(frame, mode_txt, (w // 2 - 58, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)

    lines = [
        f"FPS     {fps:5.1f}",
        f"Mode    {mode_txt}",
        f"Gesture {gesture}",
        f"I+T={pd_it:.3f}  M+T={pd_mt:.3f}  P+T={pd_pt:.3f}",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (8, 18 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1)

    if palm_count > 0:
        BAR_B  = 10
        filled = min(palm_count * BAR_B // PALM_HOLD_FRAMES, BAR_B)
        bar    = "\u2588" * filled + "\u2591" * (BAR_B - filled)
        cv2.putText(frame, f"PALM [{bar}] {palm_count}/{PALM_HOLD_FRAMES}",
                    (8, 18 + 4 * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 200, 255), 1)

    if flash_msg and now < flash_until:
        tw = len(flash_msg) * 14
        cv2.putText(frame, flash_msg, (w // 2 - tw // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 200), 3)


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

    apply_camera_settings()
    for _ in range(5):
        cap.read()
    apply_camera_settings()

    play_startup_sound()

    win_name = "Kontrol v1.1"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, _mouse_cb)

    # ── Cursor state ──────────────────────────────────────────────────────────
    raw_x_s: float | None = None
    raw_y_s: float | None = None
    cx, cy                = 0.0, 0.0
    prev_tx, prev_ty      = 0.0, 0.0
    prev_sent_x           = 0
    prev_sent_y           = 0
    hand_was_present      = False
    was_gesturing         = False   # True if last frame ran a non-cursor gesture

    # ── Gesture state ─────────────────────────────────────────────────────────
    locked                = False

    # Palm — gesture 1
    palm_hold_count       = 0
    palm_was_closed       = False
    last_palm_t           = 0.0
    palm_minimized        = False
    flash_msg             = ""
    flash_until           = 0.0

    # Index+Thumb — gesture 2 (right click)
    it_pinch_held         = False
    last_it_click_t       = 0.0

    # Middle+Thumb — gestures 3+4 (left click / drag / scroll)
    mt_pinch_held         = False
    mt_pinch_start_t      = 0.0
    drag_active           = False
    last_drag_end_t       = 0.0
    scroll_ref_y: float | None = None

    # Pinky+Thumb — gesture 5 (tile)
    pt_pinch_held         = False
    tile_history: deque   = deque(maxlen=TILE_WINDOW_FRAMES)
    last_tile_t           = 0.0
    tile_fired            = False

    # ── HUD / perf state ─────────────────────────────────────────────────────
    active_gesture        = "NONE"
    tile_held_hud         = False
    fps                   = 0.0
    fps_alpha             = 0.1
    last_frame_t          = time.time()
    pd_it_hud = pd_mt_hud = pd_pt_hud = 1.0

    print(f"Kontrol v1.1  {SCREEN_W}x{SCREEN_H}"
          f"  zone=[{ZONE_X_MIN:.2f}-{ZONE_X_MAX:.2f}, {ZONE_Y_MIN:.2f}-{ZONE_Y_MAX:.2f}]"
          f"  lm={LANDMARK_SMOOTH}  smooth[{MIN_SMOOTH}-{MAX_SMOOTH}]x{VELOCITY_SCALE}"
          f"  pinch={PINCH_THRESHOLD}  palm={PALM_HOLD_FRAMES}fr  tile={TILE_MOVE_THRESHOLD}")

    with HandLandmarker.create_from_options(opts) as detector:
        while cap.isOpened():
            frame_start = time.time()

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
            result   = detector.detect(mp_image)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]

                first_frame      = not hand_was_present
                hand_was_present = True

                pd_it_hud = pinch_dist(lm, 4, 8)
                pd_mt_hud = pinch_dist(lm, 4, 12)
                pd_pt_hud = pinch_dist(lm, 4, 20)

                # ── Priority 1: Palm close ────────────────────────────────────
                if is_palm_closed(lm):
                    palm_hold_count += 1
                    palm_was_closed  = True
                    # Block all other gestures while palm is closing
                    if drag_active:
                        mouse_up()
                        drag_active = False
                    BAR_B  = 10
                    filled = min(palm_hold_count * BAR_B // PALM_HOLD_FRAMES, BAR_B)
                    bar    = "\u2588" * filled + "\u2591" * (BAR_B - filled)
                    active_gesture = f"PALM [{bar}] {palm_hold_count}/{PALM_HOLD_FRAMES}"
                    was_gesturing  = True

                else:
                    # Palm released — check whether to fire
                    if palm_was_closed:
                        if (palm_hold_count >= PALM_HOLD_FRAMES
                                and (now - last_palm_t) > PALM_COOLDOWN):
                            if palm_minimized:
                                # Restore / maximize: Meta+PgUp (KEY_PAGEUP=104)
                                ydocall("key", "125:1", "104:1", "104:0", "125:0", blocking=True)
                                palm_minimized = False
                                flash_msg      = "RESTORE"
                            else:
                                # Minimize: Meta+PgDown (KEY_PAGEDOWN=109)
                                ydocall("key", "125:1", "109:1", "109:0", "125:0", blocking=True)
                                palm_minimized = True
                                flash_msg      = "MINIMIZE"
                            flash_until = now + 0.8
                            last_palm_t = now
                        palm_hold_count = 0
                        palm_was_closed = False

                    # ── Priority 2: Locked ────────────────────────────────────
                    if locked:
                        active_gesture = "LOCKED"
                        was_gesturing  = True

                    # ── Priority 3: Pinky+Thumb — tile ───────────────────────
                    elif is_pinky_thumb_pinched(lm):
                        tile_held_hud = True
                        if not pt_pinch_held:
                            pt_pinch_held = True
                            tile_history.clear()
                            tile_fired    = False
                        tile_history.append((lm[0].x, lm[0].y))

                        if not tile_fired and len(tile_history) >= 2:
                            xs       = [p[0] for p in tile_history]
                            ys       = [p[1] for p in tile_history]
                            dx_total = xs[-1] - xs[0]
                            dy_total = ys[-1] - ys[0]
                            thr      = TILE_MOVE_THRESHOLD

                            direction = None
                            if abs(dx_total) > abs(dy_total):
                                if   dx_total >  thr: direction = "right"
                                elif dx_total < -thr: direction = "left"
                            else:
                                if   dy_total < -thr: direction = "up"
                                elif dy_total >  thr: direction = "down"

                            if direction and (now - last_tile_t) > TILE_COOLDOWN:
                                tiling_key(direction)
                                last_tile_t  = now
                                tile_fired   = True
                                flash_msg    = f"TILE {direction.upper()}"
                                flash_until  = now + 0.6
                                tile_history.clear()

                        active_gesture = f"TILE-HOLD P+T={pd_pt_hud:.3f}"
                        was_gesturing  = True

                    # ── Priority 4+5: Middle+Thumb — scroll / drag / click ────
                    elif is_middle_thumb_pinched(lm):
                        tile_held_hud = False
                        if pt_pinch_held:
                            pt_pinch_held = False

                        if not mt_pinch_held:
                            mt_pinch_held    = True
                            mt_pinch_start_t = now
                            scroll_ref_y     = lm[0].y

                        # Per-frame wrist y delta for scroll
                        dy_norm      = lm[0].y - scroll_ref_y
                        scroll_ref_y = lm[0].y

                        if abs(dy_norm) > SCROLL_DEADZONE:
                            # Scroll mode — do NOT move cursor this frame
                            ticks = max(1, int(abs(dy_norm) * SCROLL_SPEED))
                            if dy_norm < 0:
                                scroll_up(ticks)
                                active_gesture = f"SCROLL UP x{ticks}"
                            else:
                                scroll_down(ticks)
                                active_gesture = f"SCROLL DN x{ticks}"
                            was_gesturing = True
                        else:
                            # Drag mode — cursor follows hand
                            held_t = now - mt_pinch_start_t
                            if held_t > PINCH_COOLDOWN and not drag_active:
                                mouse_down()
                                drag_active = True

                            if drag_active:
                                active_gesture = f"DRAG M+T d={pd_mt_hud:.3f}"
                                reentry = first_frame or was_gesturing
                                (raw_x_s, raw_y_s, cx, cy, prev_tx, prev_ty,
                                 prev_sent_x, prev_sent_y) = _run_cursor_pipeline(
                                    lm, raw_x_s, raw_y_s,
                                    cx, cy, prev_tx, prev_ty,
                                    prev_sent_x, prev_sent_y,
                                    reentry=reentry,
                                )
                                was_gesturing = False
                            else:
                                active_gesture = f"PINCH M+T d={pd_mt_hud:.3f}"
                                was_gesturing  = True

                    else:
                        # Middle+Thumb released — fire click or end drag
                        if mt_pinch_held:
                            if drag_active:
                                mouse_up()
                                drag_active     = False
                                last_drag_end_t = now
                            elif (now - mt_pinch_start_t) < PINCH_COOLDOWN:
                                if (now - last_drag_end_t) > PINCH_COOLDOWN:
                                    left_click()
                                    active_gesture = "L-CLICK"
                            mt_pinch_held = False
                            scroll_ref_y  = None

                        tile_held_hud = False
                        if pt_pinch_held:
                            pt_pinch_held = False

                        # ── Priority 6: Index+Thumb — right click ─────────────
                        if is_index_thumb_pinched(lm):
                            if not it_pinch_held and (now - last_it_click_t) > PINCH_COOLDOWN:
                                right_click()
                                last_it_click_t = now
                                active_gesture  = f"R-CLICK I+T={pd_it_hud:.3f}"
                            it_pinch_held = True
                            was_gesturing = True

                        else:
                            it_pinch_held = False

                            # ── Cursor pipeline ───────────────────────────────
                            reentry = first_frame or was_gesturing
                            (raw_x_s, raw_y_s, cx, cy, prev_tx, prev_ty,
                             prev_sent_x, prev_sent_y) = _run_cursor_pipeline(
                                lm, raw_x_s, raw_y_s,
                                cx, cy, prev_tx, prev_ty,
                                prev_sent_x, prev_sent_y,
                                reentry=reentry,
                            )
                            active_gesture = "CURSOR"
                            was_gesturing  = False

                draw_skeleton(frame, lm, fw_px, fh_px)
                draw_visibility_warning(frame, lm, fw_px, fh_px)

            else:
                # Hand lost — reset all transient state
                if hand_was_present:
                    raw_x_s          = None
                    raw_y_s          = None
                    hand_was_present = False
                if drag_active:
                    mouse_up()
                    drag_active = False
                mt_pinch_held    = False
                it_pinch_held    = False
                pt_pinch_held    = False
                palm_hold_count  = 0
                palm_was_closed  = False
                scroll_ref_y     = None
                tile_history.clear()
                tile_fired       = False
                tile_held_hud    = False
                active_gesture   = "NONE"

            draw_hud(
                frame, locked, active_gesture, fps,
                pd_it_hud, pd_mt_hud, pd_pt_hud,
                palm_count=palm_hold_count,
                drag_active=drag_active,
                tile_held=tile_held_hud,
                flash_msg=flash_msg,
                flash_until=flash_until,
            )
            _cam_panel.draw(frame)
            draw_buttons(frame)
            cv2.imshow(win_name, frame)

            if _quit_flag[0] or (cv2.waitKey(1) & 0xFF == ord("q")):
                print("Quit.")
                break

            elapsed   = time.time() - frame_start
            remaining = FRAME_INTERVAL - elapsed
            if remaining > 0.002:
                time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
