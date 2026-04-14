#!/usr/bin/env python3
"""
Kontrol v1.3 — Hand gesture mouse control
MediaPipe Tasks API (0.10+) → ydotool (no pynput, Wayland-safe)

Gesture priority order (strict — higher number never fires if lower active):
  1. Bunch (all 5 tips together, hold ~0.6 s, fire on RELEASE) → minimize / restore
  2. Pinky+Thumb (LM 4+20) held + wrist direction            → KDE tile
  3. Peace sign  (index+middle up, ring+pinky folded) + swipe → task switcher
  4. Middle+Thumb (LM 4+12) + vertical wrist movement        → scroll
  5. Ring+Thumb  (LM 4+16)                                   → right click
  6. Index+Thumb (LM 4+8)  hold/tap                          → drag / left click
  7. Index fingertip (LM 8) — 5-stage EMA pipeline           → cursor

Config: kontrol.conf (INI format, same directory)
Launch: cd /home/K/Storage/Projects/Kontrol && ./run.sh
"""

# ── Imports — stdlib then third-party, alphabetical within each ───────────────
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

# ── CONFIG LOADING ────────────────────────────────────────────────────────────
CONF_PATH  = Path(__file__).parent / "kontrol.conf"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

_DEFAULTS: dict[str, dict[str, str]] = {
    "screen": {
        "width":  "4480",
        "height": "1440",
    },
    "camera": {
        "id":   "0",      # C920 confirmed on /dev/video0
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
        "pinch_threshold":     "0.055",
        "pinch_cooldown":      "0.35",
        "scroll_deadzone":     "0.010",
        "scroll_speed":        "6.0",
        "palm_cooldown":       "2.0",
        "tile_move_threshold":  "0.06",
        "tile_window_frames":   "8",
        "tile_cooldown":        "0.8",
        "task_move_threshold":  "0.06",
        "task_cooldown":        "0.5",
        "bunch_threshold":      "0.10",
        "bunch_hold_frames":    "12",
    },
    "detection": {
        "detection_confidence": "0.50",
        "presence_confidence":  "0.50",
        "tracking_confidence":  "0.50",
    },
    "system": {
        "ydotool_socket": "/run/user/1000/.ydotool_socket",
    },
    "camera_tuning": {
        "auto_exposure":          "1",
        "exposure_time_absolute": "300",
        "gain":                   "100",
        "brightness":             "160",
        "contrast":               "130",
        "saturation":             "128",
        "sharpness":              "128",
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

# ── CONSTANTS (derived from config) ──────────────────────────────────────────
SCREEN_W           = _cfg.getint("screen",    "width")
SCREEN_H           = _cfg.getint("screen",    "height")
SCREEN_DIAG        = math.hypot(SCREEN_W, SCREEN_H)   # ~4706 px

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

PINCH_THRESHOLD    = _cfg.getfloat("gestures", "pinch_threshold")
PINCH_COOLDOWN     = _cfg.getfloat("gestures", "pinch_cooldown")
SCROLL_DEADZONE    = _cfg.getfloat("gestures", "scroll_deadzone")
SCROLL_SPEED       = _cfg.getfloat("gestures", "scroll_speed")
PALM_COOLDOWN      = _cfg.getfloat("gestures", "palm_cooldown")
BUNCH_THRESHOLD    = _cfg.getfloat("gestures", "bunch_threshold")   # all-tips proximity
BUNCH_HOLD_FRAMES  = _cfg.getint("gestures",   "bunch_hold_frames") # frames to hold before fire
TILE_THRESHOLD     = _cfg.getfloat("gestures", "tile_move_threshold")
TILE_WINDOW        = _cfg.getint("gestures",   "tile_window_frames")
TILE_COOLDOWN      = _cfg.getfloat("gestures", "tile_cooldown")
TASK_THRESHOLD     = _cfg.getfloat("gestures", "task_move_threshold")
TASK_COOLDOWN      = _cfg.getfloat("gestures", "task_cooldown")

DETECTION_CONF     = _cfg.getfloat("detection", "detection_confidence")
PRESENCE_CONF      = _cfg.getfloat("detection", "presence_confidence")
TRACKING_CONF      = _cfg.getfloat("detection", "tracking_confidence")

YDOTOOL_SOCKET     = _cfg.get("system", "ydotool_socket")

TARGET_FPS         = 20
FRAME_INTERVAL     = 1.0 / TARGET_FPS

# Raw Linux keycodes (input-event-codes.h)
# KEY_LEFTMETA=125  KEY_UP=103  KEY_DOWN=108  KEY_LEFT=105  KEY_RIGHT=106
# KEY_PAGEUP=104    KEY_PAGEDOWN=109
# KEY_LEFTALT=56    KEY_TAB=15  KEY_LEFTSHIFT=42
_TILING_KEYS: dict[str, tuple[str, ...]] = {
    "right": ("125:1", "106:1", "106:0", "125:0"),  # Meta+Right
    "left":  ("125:1", "105:1", "105:0", "125:0"),  # Meta+Left
    "up":    ("125:1", "103:1", "103:0", "125:0"),  # Meta+Up
    "down":  ("125:1", "108:1", "108:0", "125:0"),  # Meta+Down
}

os.environ["YDOTOOL_SOCKET"] = YDOTOOL_SOCKET

# MediaPipe Tasks API (NOT mp.solutions — removed in 0.10+)
BaseOptions              = mp.tasks.BaseOptions
HandLandmarker           = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions    = mp.tasks.vision.HandLandmarkerOptions
RunningMode              = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections


# ── YDOTOOL HELPERS ───────────────────────────────────────────────────────────
def ydocall(*args, blocking: bool = False) -> None:
    """
    Issue a ydotool command.
    blocking=False → Popen fire-and-forget (cursor moves, clicks).
    blocking=True  → subprocess.run, wait for completion (key sequences).
    """
    cmd = ["ydotool", *[str(a) for a in args]]
    if blocking:
        subprocess.run(cmd, env=os.environ,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(cmd, env=os.environ,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mouse_down() -> None:
    ydocall("click", "0x40")   # left button down


def mouse_up() -> None:
    ydocall("click", "0x80")   # left button up


def right_click() -> None:
    ydocall("click", "0xC1")   # right click (down+up combined)


def scroll_up(ticks: int = 1) -> None:
    ydocall("mousemove", "--wheel", "-x", "0", "-y", str(-ticks))  # wheel up


def scroll_down(ticks: int = 1) -> None:
    ydocall("mousemove", "--wheel", "-x", "0", "-y", str(ticks))   # wheel down


def tiling_key(direction: str) -> None:
    seq = _TILING_KEYS.get(direction)
    if seq:
        ydocall("key", *seq, blocking=True)   # Meta+direction → KDE tile


# ── LANDMARK HELPERS (pure, no side effects) ─────────────────────────────────
def pdist(lm, a: int, b: int) -> float:
    """Normalised Euclidean distance between two landmarks."""
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def is_index_thumb(lm) -> bool:
    """LM 4 (thumb tip) + LM 8 (index tip) pinch → LEFT CLICK / DRAG."""
    return pdist(lm, 4, 8) < PINCH_THRESHOLD


def is_middle_thumb(lm) -> bool:
    """LM 4 (thumb tip) + LM 12 (middle tip) pinch → LEFT CLICK / DRAG / SCROLL."""
    return pdist(lm, 4, 12) < PINCH_THRESHOLD


def is_ring_thumb(lm) -> bool:
    """LM 4 (thumb tip) + LM 16 (ring tip) pinch → RIGHT CLICK."""
    return pdist(lm, 4, 16) < PINCH_THRESHOLD


def is_pinky_thumb(lm) -> bool:
    """LM 4 (thumb tip) + LM 20 (pinky tip) pinch → TILE."""
    return pdist(lm, 4, 20) < PINCH_THRESHOLD


def is_peace_sign(lm) -> bool:
    """
    Index (LM 8) + middle (LM 12) extended upward; ring (LM 16) + pinky (LM 20)
    folded; no thumb pinch active. → TASK SWITCHER swipe.
    """
    index_up   = lm[8].y  < lm[5].y    # index tip above index MCP
    middle_up  = lm[12].y < lm[9].y    # middle tip above middle MCP
    ring_down  = lm[16].y > lm[13].y   # ring tip below ring MCP
    pinky_down = lm[20].y > lm[17].y   # pinky tip below pinky MCP
    # Thumb must not be pinching — avoids conflict with any pinch gesture
    no_pinch   = (pdist(lm, 4, 8)  > PINCH_THRESHOLD and
                  pdist(lm, 4, 12) > PINCH_THRESHOLD and
                  pdist(lm, 4, 16) > PINCH_THRESHOLD and
                  pdist(lm, 4, 20) > PINCH_THRESHOLD)
    return index_up and middle_up and ring_down and pinky_down and no_pinch


def is_bunch(lm) -> bool:
    """
    All 5 fingertips converge to a single point — thumb touches all other tips.
    Like pinching all fingers together (Italian hand gesture / 'bunch').
    Uses BUNCH_THRESHOLD (looser than PINCH_THRESHOLD — 4 simultaneous contacts).
    """
    return (
        pdist(lm, 4, 8)  < BUNCH_THRESHOLD and   # thumb tip ↔ index tip
        pdist(lm, 4, 12) < BUNCH_THRESHOLD and   # thumb tip ↔ middle tip
        pdist(lm, 4, 16) < BUNCH_THRESHOLD and   # thumb tip ↔ ring tip
        pdist(lm, 4, 20) < BUNCH_THRESHOLD        # thumb tip ↔ pinky tip
    )


# ── CAMERA ────────────────────────────────────────────────────────────────────
def apply_camera_settings() -> None:
    """
    Apply v4l2 controls from kontrol.conf [camera_tuning].
    MUST be called AFTER cv2.VideoCapture opens and FOURCC/resolution are set.
    Two-pass: call once before warm-up reads, once after (stream-on reset).
    """
    dev = f"/dev/video{CAM_ID}"
    ct  = _cfg["camera_tuning"]

    def v4l2_set(ctrl: str, value: str) -> None:
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", dev, f"--set-ctrl={ctrl}={value}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=2,
            )
            print(f"[CAM] {ctrl} = {value} \u2713")
        except Exception as exc:
            print(f"[CAM] {ctrl} SKIPPED ({exc})")

    time.sleep(0.3)   # let UVC settle after VideoCapture opens

    # auto_exposure=1 FIRST — enables manual mode; subsequent controls only work after
    v4l2_set("auto_exposure",              ct.get("auto_exposure",            "1"))
    v4l2_set("exposure_time_absolute",     ct.get("exposure_time_absolute",   "300"))
    v4l2_set("gain",                       ct.get("gain",                     "100"))
    v4l2_set("brightness",                 ct.get("brightness",               "160"))
    v4l2_set("contrast",                   ct.get("contrast",                 "130"))
    v4l2_set("sharpness",                  ct.get("sharpness",                "128"))
    v4l2_set("saturation",                 ct.get("saturation",               "128"))
    v4l2_set("backlight_compensation",     ct.get("backlight_compensation",   "1"))
    v4l2_set("power_line_frequency",       ct.get("power_line_frequency",     "1"))
    # focus: disable auto first so focus_absolute takes effect
    v4l2_set("focus_automatic_continuous", ct.get("focus_autos",              "0"))
    v4l2_set("focus_absolute",             ct.get("focus_absolute",           "30"))


# ── DRAW FUNCTIONS ────────────────────────────────────────────────────────────
def draw_skeleton(frame, lm, fw: int, fh: int) -> None:
    """Draw MediaPipe hand skeleton onto frame."""
    for c in HandLandmarksConnections.HAND_CONNECTIONS:
        ax = int(lm[c.start].x * fw)
        ay = int(lm[c.start].y * fh)
        bx = int(lm[c.end].x   * fw)
        by = int(lm[c.end].y   * fh)
        cv2.line(frame, (ax, ay), (bx, by), (60, 160, 60), 1)
    for lmk in lm:
        cv2.circle(frame, (int(lmk.x * fw), int(lmk.y * fh)), 3, (100, 200, 100), -1)


def draw_fingertip_markers(frame, lm, fw: int, fh: int,
                            it_pinched: bool, mt_pinched: bool,
                            rt_pinched: bool, pt_pinched: bool) -> None:
    """Coloured circles on active pinch fingertips."""
    if it_pinched:
        # index tip (LM 8) — green when index+thumb pinched (left click)
        cv2.circle(frame, (int(lm[8].x * fw), int(lm[8].y * fh)),
                   12, (0, 200, 0), -1)
    if mt_pinched:
        # middle tip (LM 12) — blue when middle+thumb pinched (scroll)
        cv2.circle(frame, (int(lm[12].x * fw), int(lm[12].y * fh)),
                   12, (200, 80, 0), -1)
    if rt_pinched:
        # ring tip (LM 16) — red when ring+thumb pinched (right click)
        cv2.circle(frame, (int(lm[16].x * fw), int(lm[16].y * fh)),
                   12, (0, 0, 220), -1)
    if pt_pinched:
        # pinky tip (LM 20) — yellow when pinky+thumb pinched (tile)
        cv2.circle(frame, (int(lm[20].x * fw), int(lm[20].y * fh)),
                   12, (0, 220, 220), -1)


def draw_zone_warning(frame, lm, fw: int, fh: int) -> None:
    """Warn when index fingertip is near the zone boundary."""
    x, y   = lm[8].x, lm[8].y   # index fingertip (LM 8) normalised
    margin = 0.05                 # warn when within 5% of zone edge
    font   = cv2.FONT_HERSHEY_SIMPLEX
    col    = (0, 220, 220)        # yellow

    if x < ZONE_X_MIN + margin:
        cv2.putText(frame, "< MOVE IN", (4, fh // 2),
                    font, 0.50, col, 1)
    if x > ZONE_X_MAX - margin:
        cv2.putText(frame, "MOVE IN >", (fw - 90, fh // 2),
                    font, 0.50, col, 1)
    if y < ZONE_Y_MIN + margin:
        cv2.putText(frame, "^ MOVE IN", (fw // 2 - 44, 18),
                    font, 0.50, col, 1)
    if y > ZONE_Y_MAX - margin:
        cv2.putText(frame, "v MOVE IN", (fw // 2 - 44, fh - 6),
                    font, 0.50, col, 1)


def draw_hud(frame, gesture: str, fps: float,
             pd_it: float, pd_mt: float, pd_rt: float, pd_pt: float,
             palm_count: int = 0,
             drag_active: bool = False,
             tile_held: bool = False,
             hand_detected: bool = True,
             palm_closing: bool = False,
             flash_msg: str = "",
             flash_until: float = 0.0) -> None:
    """
    HUD overlay: border, status panel, flash messages.
    All gesture logic lives in run() — this function only draws.
    """
    h, w  = frame.shape[:2]
    now   = time.time()
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # ── Border colour ─────────────────────────────────────────────────────────
    if not hand_detected:
        border_col = (0, 0, 200)       # red = no hand
    elif palm_closing:
        border_col = (0, 120, 255)     # orange = palm closing
    elif drag_active:
        border_col = (200, 80, 0)      # blue = drag active
    elif tile_held:
        border_col = (0, 200, 220)     # yellow = tile hold
    else:
        border_col = (30, 200, 50)     # green = tracking / cursor

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_col, 3)

    # ── Status panel — translucent dark background ─────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 108), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # ── HUD text lines ────────────────────────────────────────────────────────
    # label colour: cyan; value colour: white
    lc = (0, 200, 200)   # cyan labels
    vc = (220, 220, 220) # white values
    x0, y0, dy = 8, 15, 20

    def hud_line(row: int, label: str, value: str) -> None:
        cv2.putText(frame, label, (x0, y0 + row * dy), font, 0.42, lc, 1)
        cv2.putText(frame, value, (x0 + 68, y0 + row * dy), font, 0.42, vc, 1)

    hud_line(0, "[FPS]    ", f"{fps:5.1f}")
    hud_line(1, "[HAND]   ", "DETECTED" if hand_detected else "NO HAND")
    hud_line(2, "[GESTURE]", gesture)
    hud_line(3, "[PINCH]  ", f"I={pd_it:.3f} M={pd_mt:.3f} R={pd_rt:.3f} P={pd_pt:.3f}")

    # Palm progress bar — only shown when palm is closing
    if palm_count > 0:
        BAR_W   = 10
        filled  = min(palm_count * BAR_W // BUNCH_HOLD_FRAMES, BAR_W)
        bar     = "\u2588" * filled + "\u2591" * (BAR_W - filled)
        cv2.putText(frame, f"[PALM] {bar} {palm_count}/{BUNCH_HOLD_FRAMES}",
                    (x0, y0 + 4 * dy), font, 0.40, (0, 180, 255), 1)

    # ── Flash message — centred, large, 1.0 s ─────────────────────────────
    if flash_msg and now < flash_until:
        text_sz = cv2.getTextSize(flash_msg, font, 1.2, 2)[0]
        tx = (w - text_sz[0]) // 2
        ty = h // 2
        cv2.putText(frame, flash_msg, (tx, ty), font, 1.2, (0, 255, 220), 2)


# ── In-frame UI buttons ───────────────────────────────────────────────────────
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
        subprocess.Popen(
            ["wmctrl", "-r", ":ACTIVE:", "-b", "add,hidden"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def draw_buttons(frame) -> None:
    """Draw [—] and [✕] buttons in top-right corner."""
    global _btn_close, _btn_min
    h, w       = frame.shape[:2]
    pad, bw, bh = 5, 28, 20

    # [✕] close button
    cx1 = w - pad - bw
    cy1 = pad
    cx2 = w - pad
    cy2 = pad + bh
    _btn_close = (cx1, cy1, cx2, cy2)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (40, 40, 150), -1)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (140, 140, 140), 1)
    cv2.putText(frame, "X", (cx1 + 8, cy2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # [—] minimise button
    mx1 = cx1 - pad - bw
    my1 = pad
    mx2 = cx1 - pad
    my2 = pad + bh
    _btn_min = (mx1, my1, mx2, my2)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (25, 90, 25), -1)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (140, 140, 140), 1)
    cv2.putText(frame, "-", (mx1 + 9, my2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


# ── STARTUP SOUND ─────────────────────────────────────────────────────────────
def play_startup_sound() -> None:
    """Three-tone ascending chime on startup."""
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
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
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
    """
    5-stage cursor pipeline. Returns updated state tuple.
    reentry=True → snap to current position, send no delta (avoids cursor jump).

    Returns: (raw_x_s, raw_y_s, cx, cy, prev_tx, prev_ty, prev_sent_x, prev_sent_y)
    """
    lx, ly = lm[8].x, lm[8].y   # index fingertip (LM 8) normalised position

    # STAGE 1 — Landmark pre-smooth (fixed α = LANDMARK_SMOOTH)
    # Kills high-frequency MediaPipe jitter BEFORE zone mapping.
    if raw_x_s is None:
        raw_x_s, raw_y_s = lx, ly   # first detection: snap to landmark
    raw_x_s += (lx - raw_x_s) * LANDMARK_SMOOTH
    raw_y_s += (ly - raw_y_s) * LANDMARK_SMOOTH

    # STAGE 2 — Zone mapping: [ZONE_MIN, ZONE_MAX] → [0,1] → screen pixels
    # Hand only needs to reach zone boundary to hit screen edge.
    nx = max(0.0, min(1.0, (raw_x_s - ZONE_X_MIN) / (ZONE_X_MAX - ZONE_X_MIN)))
    ny = max(0.0, min(1.0, (raw_y_s - ZONE_Y_MIN) / (ZONE_Y_MAX - ZONE_Y_MIN)))
    tx = nx * SCREEN_W
    ty = ny * SCREEN_H

    # STAGE 3 — Re-entry guard + velocity
    if reentry:
        # Snap all EMA state to current mapped position.
        # No delta sent — cursor stays where it was.
        cx, cy = tx, ty
        prev_tx, prev_ty = tx, ty
        prev_sent_x, prev_sent_y = int(tx), int(ty)
        return raw_x_s, raw_y_s, cx, cy, prev_tx, prev_ty, prev_sent_x, prev_sent_y

    vel_px   = math.hypot(tx - prev_tx, ty - prev_ty)   # screen px per frame
    vel_norm = vel_px / SCREEN_DIAG                       # normalised 0.0–~0.15
    prev_tx, prev_ty = tx, ty

    # STAGE 4 — Velocity-adaptive screen-space EMA
    # Fast → high alpha → snappy.  Slow → low alpha → jitter-free.
    alpha = max(MIN_SMOOTH, min(MAX_SMOOTH, vel_norm * VELOCITY_SCALE))
    cx += (tx - cx) * alpha
    cy += (ty - cy) * alpha

    # STAGE 5 — Integer delta with deadzone
    # Only fire ydotool if movement exceeds CURSOR_DEADZONE_PX.
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


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def run() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Download:\n  curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,   # temporal tracking, strictly increasing timestamps
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
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # always latest frame, no stale-buffer stalls

    apply_camera_settings()   # pass 1 — before stream-on
    for _ in range(5):
        cap.read()            # warm-up reads — triggers VIDIOC_STREAMON
    apply_camera_settings()   # pass 2 — after stream-on reset re-applies controls

    # Brightness verification
    _ok, _frame = cap.read()
    if _ok:
        _mean = _frame.mean()
        print(f"[CAM] brightness mean: {_mean:.1f}  (target >80)")

    play_startup_sound()

    win_name = "Kontrol v1.3"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, _mouse_cb)

    # ── CURSOR STATE ──────────────────────────────────────────────────────────
    raw_x_s: float | None = None   # landmark pre-smooth x (None = not yet seen)
    raw_y_s: float | None = None   # landmark pre-smooth y
    cx: float             = SCREEN_W / 2   # EMA cursor x in screen pixels
    cy: float             = SCREEN_H / 2   # EMA cursor y in screen pixels
    prev_tx: float        = cx             # previous EMA target x
    prev_ty: float        = cy             # previous EMA target y
    prev_sent_x: int      = int(cx)        # last integer x sent to ydotool
    prev_sent_y: int      = int(cy)        # last integer y sent to ydotool
    hand_was_present: bool = False         # was hand detected last frame?
    was_gesturing: bool    = False         # last frame ran a non-cursor gesture?

    # ── GESTURE STATE — bunch (all tips together, priority 1) ────────────────
    palm_frames:     int   = 0      # consecutive frames bunch held
    palm_was_closed: bool  = False  # True while bunch is held this sequence
    last_palm_t:     float = 0.0    # timestamp of last palm trigger
    palm_minimized:  bool  = False  # toggle: True=minimized, False=restored

    # ── GESTURE STATE — pinky+thumb tiling (priority 2) ──────────────────────
    pt_held:       bool  = False                       # pinky+thumb currently pinched?
    tile_wrist_xs: deque = deque(maxlen=TILE_WINDOW)   # wrist x history
    tile_wrist_ys: deque = deque(maxlen=TILE_WINDOW)   # wrist y history
    tile_fired:    bool  = False   # True = tile already fired this hold
    last_tile_t:   float = 0.0    # timestamp of last tile fire

    # ── GESTURE STATE — peace sign task switcher (priority 3) ─────────────────
    peace_held:     bool  = False                      # peace sign currently held?
    task_wrist_xs:  deque = deque(maxlen=8)            # wrist x history for swipe
    last_task_t:    float = 0.0    # timestamp of last task switch

    # ── GESTURE STATE — middle+thumb scroll (priority 4) ─────────────────────
    mt_held:      bool        = False   # middle+thumb currently pinched?
    scroll_ref_y: float | None = None   # wrist y reference for per-frame delta

    # ── GESTURE STATE — ring+thumb right click (priority 5) ───────────────────
    rt_held:       bool  = False   # ring+thumb currently pinched?
    last_rclick_t: float = 0.0    # timestamp of last right click

    # ── GESTURE STATE — index+thumb left click / drag (priority 6) ───────────
    it_held:         bool  = False   # index+thumb currently pinched?
    it_start_t:      float = 0.0    # timestamp when index+thumb pinch started
    drag_active:     bool  = False   # left mouse button currently held?
    last_drag_end_t: float = 0.0    # timestamp when last drag released

    # ── HUD / PERF STATE ──────────────────────────────────────────────────────
    active_gesture: str   = "NONE"
    flash_msg:      str   = ""
    flash_until:    float = 0.0
    fps:            float = 0.0
    fps_alpha:      float = 0.1    # EMA alpha for FPS display smoothing
    last_frame_t:   float = time.time()
    pd_it: float = 1.0   # index+thumb distance display
    pd_mt: float = 1.0   # middle+thumb distance display
    pd_rt: float = 1.0   # ring+thumb distance display
    pd_pt: float = 1.0   # pinky+thumb distance display

    print(
        f"Kontrol v1.3  {SCREEN_W}x{SCREEN_H}"
        f"  cam=/dev/video{CAM_ID}"
        f"  zone=[{ZONE_X_MIN:.2f}-{ZONE_X_MAX:.2f}, {ZONE_Y_MIN:.2f}-{ZONE_Y_MAX:.2f}]"
        f"  lm={LANDMARK_SMOOTH}  smooth[{MIN_SMOOTH}-{MAX_SMOOTH}]x{VELOCITY_SCALE}"
        f"  pinch={PINCH_THRESHOLD}  bunch={BUNCH_THRESHOLD}/{BUNCH_HOLD_FRAMES}fr"
    )

    _last_ts_ms: int = 0   # VIDEO mode requires strictly increasing timestamps

    with HandLandmarker.create_from_options(opts) as detector:
        while cap.isOpened():
            frame_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                continue

            now = time.time()

            # FPS tracking — exponential moving average
            dt = now - last_frame_t
            last_frame_t = now
            if dt > 0:
                fps = fps * (1.0 - fps_alpha) + (1.0 / dt) * fps_alpha

            # Monotonic timestamp for VIDEO mode (strictly increasing)
            _ts_ms      = max(int(now * 1000), _last_ts_ms + 1)
            _last_ts_ms = _ts_ms

            if FLIP:
                frame = cv2.flip(frame, 1)

            fh_px, fw_px = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect_for_video(mp_image, _ts_ms)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]   # single hand always at index 0

                first_frame      = not hand_was_present
                hand_was_present = True

                # Update pinch distances for HUD every frame
                pd_it = pdist(lm, 4, 8)    # index+thumb
                pd_mt = pdist(lm, 4, 12)   # middle+thumb
                pd_rt = pdist(lm, 4, 16)   # ring+thumb
                pd_pt = pdist(lm, 4, 20)   # pinky+thumb

                # Wrist history — always append before any priority check
                tile_wrist_xs.append(lm[0].x)   # LM 0 = wrist
                tile_wrist_ys.append(lm[0].y)
                task_wrist_xs.append(lm[0].x)

                # ════════════════════════════════════════════════════════════════
                # PRIORITY 1 — BUNCH (all 5 fingertips meet at one point)
                # Blocks ALL other gestures. Fires minimize/restore on RELEASE.
                # ════════════════════════════════════════════════════════════════
                if is_bunch(lm):
                    palm_frames    += 1
                    palm_was_closed = True
                    was_gesturing   = True

                    if drag_active:          # never leave button stuck
                        mouse_up()
                        drag_active = False

                    active_gesture = f"PALM {palm_frames}/{BUNCH_HOLD_FRAMES}"

                else:
                    # Bunch released — fire if held long enough
                    if palm_was_closed:
                        if (palm_frames >= BUNCH_HOLD_FRAMES
                                and (now - last_palm_t) > PALM_COOLDOWN):
                            # KDE "Show Desktop" — built-in toggle:
                            # first call hides all windows, second restores them.
                            subprocess.run(
                                ["qdbus", "org.kde.kglobalaccel",
                                 "/component/kwin",
                                 "org.kde.kglobalaccel.Component.invokeShortcut",
                                 "Show Desktop"],
                                env=os.environ,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            )
                            palm_minimized = not palm_minimized
                            flash_msg      = "SHOW DESKTOP" if palm_minimized else "RESTORE"
                            flash_until    = now + 1.0
                            last_palm_t    = now
                        palm_frames     = 0
                        palm_was_closed = False

                    # ════════════════════════════════════════════════════════════
                    # PRIORITY 2 — PINKY+THUMB → TILE
                    # ════════════════════════════════════════════════════════════
                    if is_pinky_thumb(lm):
                        active_gesture = f"TILE HOLD  P={pd_pt:.3f}"
                        was_gesturing  = True

                        if not pt_held:
                            tile_wrist_xs.clear()   # discard pre-pinch history
                            tile_wrist_ys.clear()
                            tile_fired = False
                        pt_held = True

                        if not tile_fired and len(tile_wrist_xs) >= 2:
                            dx_total = tile_wrist_xs[-1] - tile_wrist_xs[0]
                            dy_total = tile_wrist_ys[-1] - tile_wrist_ys[0]
                            adx, ady = abs(dx_total), abs(dy_total)

                            if (adx > TILE_THRESHOLD or ady > TILE_THRESHOLD) \
                                    and (now - last_tile_t) > TILE_COOLDOWN:
                                direction = (
                                    ("right" if dx_total > 0 else "left") if adx > ady
                                    else ("down" if dy_total > 0 else "up")
                                )
                                tiling_key(direction)   # Meta+direction
                                tile_fired     = True
                                last_tile_t    = now
                                flash_msg      = f"TILE {direction.upper()}"
                                flash_until    = now + 1.0
                                active_gesture = f"TILE {direction.upper()}"
                                tile_wrist_xs.clear()
                                tile_wrist_ys.clear()

                    else:
                        if pt_held:
                            tile_fired = False
                            pt_held    = False

                        # ════════════════════════════════════════════════════════
                        # PRIORITY 3 — PEACE SIGN (index+middle up, rest folded)
                        #              + WRIST SWIPE → TASK SWITCHER
                        # Alt+Tab (next) or Alt+Shift+Tab (prev). Repeatable.
                        # ════════════════════════════════════════════════════════
                        if is_peace_sign(lm):
                            active_gesture = "TASK SWITCH"
                            was_gesturing  = True

                            if not peace_held:
                                task_wrist_xs.clear()   # discard pre-gesture history
                                peace_held = True

                            if len(task_wrist_xs) >= 2 \
                                    and (now - last_task_t) > TASK_COOLDOWN:
                                dx = task_wrist_xs[-1] - task_wrist_xs[0]
                                if abs(dx) > TASK_THRESHOLD:
                                    if dx > 0:
                                        # Swipe right → next task: Alt+Tab
                                        ydocall("key", "56:1", "15:1", "15:0", "56:0",
                                                blocking=True)
                                        active_gesture = "TASK →"
                                    else:
                                        # Swipe left → prev task: Alt+Shift+Tab
                                        ydocall("key", "56:1", "42:1", "15:1",
                                                "15:0", "42:0", "56:0", blocking=True)
                                        active_gesture = "TASK ←"
                                    last_task_t = now
                                    task_wrist_xs.clear()   # ready for next swipe

                        else:
                            if peace_held:
                                peace_held = False
                                task_wrist_xs.clear()

                            # ════════════════════════════════════════════════════
                            # PRIORITY 4 — MIDDLE+THUMB → SCROLL
                            # ════════════════════════════════════════════════════
                            if is_middle_thumb(lm):
                                was_gesturing = True

                                if not mt_held:
                                    mt_held      = True
                                    scroll_ref_y = lm[0].y   # wrist y anchor

                                dy_norm      = lm[0].y - scroll_ref_y
                                scroll_ref_y = lm[0].y

                                if abs(dy_norm) > SCROLL_DEADZONE:
                                    ticks   = max(1, int(abs(dy_norm) * SCROLL_SPEED))
                                    wheel_y = -ticks if dy_norm < 0 else ticks
                                    ydocall("mousemove", "--wheel",
                                            "-x", "0", "-y", str(wheel_y))
                                    active_gesture = f"SCROLL {'UP' if dy_norm < 0 else 'DOWN'} x{ticks}"
                                else:
                                    active_gesture = f"SCROLL HOLD  M={pd_mt:.3f}"

                            else:
                                if mt_held:
                                    mt_held      = False
                                    scroll_ref_y = None

                                # ════════════════════════════════════════════════
                                # PRIORITY 5 — RING+THUMB → RIGHT CLICK
                                # Single fire per entry; cooldown guards repeat.
                                # ════════════════════════════════════════════════
                                if is_ring_thumb(lm):
                                    if not rt_held \
                                            and (now - last_rclick_t) > PINCH_COOLDOWN:
                                        right_click()   # right button down+up
                                        last_rclick_t  = now
                                        active_gesture = f"R-CLICK  R={pd_rt:.3f}"
                                    rt_held       = True
                                    was_gesturing = True

                                else:
                                    rt_held = False

                                    # ════════════════════════════════════════════
                                    # PRIORITY 6 — INDEX+THUMB → LEFT CLICK / DRAG
                                    # Quick tap → click. Hold > cooldown → drag.
                                    # ════════════════════════════════════════════
                                    if is_index_thumb(lm):
                                        if not it_held:
                                            it_held   = True
                                            it_start_t = now

                                        held_t = now - it_start_t
                                        if held_t > PINCH_COOLDOWN and not drag_active \
                                                and (now - last_drag_end_t) > PINCH_COOLDOWN:
                                            mouse_down()   # left button down → drag starts
                                            drag_active = True

                                        if drag_active:
                                            # Cursor moves during drag
                                            reentry = first_frame or was_gesturing
                                            (raw_x_s, raw_y_s, cx, cy,
                                             prev_tx, prev_ty,
                                             prev_sent_x, prev_sent_y) = \
                                                run_cursor_pipeline(
                                                    lm, raw_x_s, raw_y_s,
                                                    cx, cy, prev_tx, prev_ty,
                                                    prev_sent_x, prev_sent_y,
                                                    reentry=reentry,
                                                )
                                            was_gesturing  = False
                                            active_gesture = f"DRAG  I={pd_it:.3f}"
                                        else:
                                            was_gesturing  = True
                                            active_gesture = f"PINCH  I={pd_it:.3f}"

                                    else:
                                        # Index+thumb released — complete the click/drag
                                        if it_held:
                                            if drag_active:
                                                mouse_up()          # left button up
                                                drag_active     = False
                                                last_drag_end_t = now
                                            elif (now - it_start_t) < PINCH_COOLDOWN \
                                                    and (now - last_drag_end_t) > PINCH_COOLDOWN:
                                                # Quick tap: send click (down then up)
                                                mouse_down()
                                                mouse_up()
                                                active_gesture = "L-CLICK"
                                            it_held = False

                                        # ════════════════════════════════════════
                                        # PRIORITY 7 (default) — CURSOR
                                        # ════════════════════════════════════════
                                        reentry = first_frame or was_gesturing
                                        (raw_x_s, raw_y_s, cx, cy,
                                         prev_tx, prev_ty,
                                         prev_sent_x, prev_sent_y) = \
                                            run_cursor_pipeline(
                                                lm, raw_x_s, raw_y_s,
                                                cx, cy, prev_tx, prev_ty,
                                                prev_sent_x, prev_sent_y,
                                                reentry=reentry,
                                            )
                                        active_gesture = "CURSOR"
                                        was_gesturing  = False

                draw_skeleton(frame, lm, fw_px, fh_px)
                draw_fingertip_markers(
                    frame, lm, fw_px, fh_px,
                    it_pinched=is_index_thumb(lm),
                    mt_pinched=is_middle_thumb(lm),
                    rt_pinched=is_ring_thumb(lm),
                    pt_pinched=is_pinky_thumb(lm),
                )
                draw_zone_warning(frame, lm, fw_px, fh_px)

            else:
                # ── HAND LOST ─────────────────────────────────────────────────
                if hand_was_present:
                    raw_x_s = raw_y_s = None
                    hand_was_present = False

                if drag_active:
                    mouse_up()   # never leave left button stuck
                    drag_active = False

                # Reset all transient gesture state
                it_held         = False
                rt_held         = False
                mt_held         = False
                pt_held         = False
                peace_held      = False
                scroll_ref_y    = None
                palm_frames     = 0
                palm_was_closed = False
                tile_wrist_xs.clear()
                tile_wrist_ys.clear()
                task_wrist_xs.clear()
                tile_fired     = False
                was_gesturing  = True   # force re-entry snap on next hand detection
                active_gesture = "NO HAND"

            draw_hud(
                frame,
                gesture       = active_gesture,
                fps           = fps,
                pd_it         = pd_it,
                pd_mt         = pd_mt,
                pd_rt         = pd_rt,
                pd_pt         = pd_pt,
                palm_count    = palm_frames,
                drag_active   = drag_active,
                tile_held     = pt_held,
                hand_detected = bool(result.hand_landmarks),
                palm_closing  = palm_frames > 0,
                flash_msg     = flash_msg,
                flash_until   = flash_until,
            )
            draw_buttons(frame)
            cv2.imshow(win_name, frame)

            if _quit_flag[0] or (cv2.waitKey(1) & 0xFF == ord("q")):
                print("Quit.")
                break

            # FPS cap — sleep remainder of frame interval
            elapsed   = time.perf_counter() - frame_start
            remaining = FRAME_INTERVAL - elapsed
            if remaining > 0.002:
                time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
