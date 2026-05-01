#!/usr/bin/env python3
"""
Kontrol v1.6 — Hand gesture mouse control
MediaPipe Tasks API (0.10+) → ydotool (no pynput, Wayland-safe)

Gesture priority order (strict — higher number never fires if lower active):
  1. Bunch (all 5 tips together, hold ~1 s, fire on RELEASE) → Show Desktop
  2. Pinky+Thumb (LM 4+20) held + wrist direction            → KDE tile
  3. Three-finger pinch (Thumb+Index+Middle, LM 4+8+12)      → KDE Overview
  4. Wrist rotation CW/CCW (knuckle axis angle velocity)      → Alt+Tab / Shift+Alt+Tab
  5. Middle+Thumb (LM 4+12) + vertical wrist movement        → scroll
  6. Ring+Thumb  (LM 4+16)                                   → right click
  7. Index+Thumb (LM 4+8)  hold/tap                          → drag / left click
  8. Index fingertip (LM 8) — 5-stage EMA pipeline           → cursor

KWin D-Bus: org.kde.kglobalaccel.Component.invokeShortcut via /component/kwin
Config: kontrol.conf (INI format, same directory)
Launch: cd /home/K/Storage/Projects/Kontrol && ./run.sh
"""

import argparse
import configparser
import math
import os
import signal
import struct
import subprocess
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

try:
    from flask import Flask, jsonify, request as freq
    _FLASK_OK = True
except ImportError:
    _FLASK_OK = False

# ── CLI & SIGNALS ─────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(description="Kontrol — hand gesture mouse control")
_ap.add_argument("--headless", action="store_true", help="Run without display window")
_args, _ = _ap.parse_known_args()
HEADLESS  = [_args.headless]

_running = [True]


def _sig_handler(signum, _frame) -> None:
    _running[0] = False
    print(f"\n[kontrol] Signal {signum} — shutting down")


signal.signal(signal.SIGTERM, _sig_handler)
signal.signal(signal.SIGINT,  _sig_handler)

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
        "rotation_threshold":       "1.8",
        "rotation_cooldown":        "0.6",
        "rotation_min_frames":      "8",
        "desktop_swipe_threshold":  "0.08",
        "desktop_swipe_cooldown":   "0.6",
        "zoom_threshold":           "0.06",
        "zoom_cooldown":            "0.4",
    },
    "detection": {
        "detection_confidence": "0.50",
        "presence_confidence":  "0.50",
        "tracking_confidence":  "0.50",
    },
    "system": {
        "ydotool_socket":         "/run/user/1000/.ydotool_socket",
        "headless_notifications": "true",
        "api_enabled":            "true",
        "api_host":               "127.0.0.1",
        "api_port":               "5555",
    },
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

# ── CONSTANTS (non-gesture, frozen) ──────────────────────────────────────────
SCREEN_W    = _cfg.getint("screen", "width")
SCREEN_H    = _cfg.getint("screen", "height")
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

# ── MUTABLE GESTURE SETTINGS ─────────────────────────────────────────────────
_SETTINGS: dict[str, float | int] = {
    "pinch_threshold":        _cfg.getfloat("gestures", "pinch_threshold"),
    "pinch_cooldown":         _cfg.getfloat("gestures", "pinch_cooldown"),
    "three_finger_threshold": _cfg.getfloat("gestures", "three_finger_threshold"),
    "three_finger_cooldown":  _cfg.getfloat("gestures", "three_finger_cooldown"),
    "scroll_deadzone":        _cfg.getfloat("gestures", "scroll_deadzone"),
    "scroll_speed":           _cfg.getfloat("gestures", "scroll_speed"),
    "scroll_vel_alpha":       _cfg.getfloat("gestures", "scroll_vel_alpha"),
    "scroll_max_ticks":       _cfg.getint("gestures",   "scroll_max_ticks"),
    "bunch_threshold":        _cfg.getfloat("gestures", "bunch_threshold"),
    "bunch_hold_frames":      _cfg.getint("gestures",   "bunch_hold_frames"),
    "palm_cooldown":          _cfg.getfloat("gestures", "palm_cooldown"),
    "tile_move_threshold":    _cfg.getfloat("gestures", "tile_move_threshold"),
    "tile_cooldown":          _cfg.getfloat("gestures", "tile_cooldown"),
    "rotation_threshold":      _cfg.getfloat("gestures", "rotation_threshold"),
    "rotation_cooldown":       _cfg.getfloat("gestures", "rotation_cooldown"),
    "desktop_swipe_threshold": _cfg.getfloat("gestures", "desktop_swipe_threshold"),
    "desktop_swipe_cooldown":  _cfg.getfloat("gestures", "desktop_swipe_cooldown"),
    "zoom_threshold":          _cfg.getfloat("gestures", "zoom_threshold"),
    "zoom_cooldown":           _cfg.getfloat("gestures", "zoom_cooldown"),
}

STEP_SIZES: dict[str, float | int] = {
    "pinch_threshold":        0.002,
    "pinch_cooldown":         0.05,
    "three_finger_threshold": 0.002,
    "three_finger_cooldown":  0.1,
    "scroll_deadzone":        0.001,
    "scroll_speed":           0.5,
    "scroll_vel_alpha":       0.02,
    "scroll_max_ticks":       1,
    "bunch_threshold":        0.005,
    "bunch_hold_frames":      1,
    "palm_cooldown":          0.1,
    "tile_move_threshold":    0.005,
    "tile_cooldown":          0.05,
    "rotation_threshold":      0.1,
    "rotation_cooldown":       0.05,
    "desktop_swipe_threshold": 0.01,
    "desktop_swipe_cooldown":  0.05,
    "zoom_threshold":          0.005,
    "zoom_cooldown":           0.05,
}

VALUE_CLAMPS: dict[str, tuple] = {
    "pinch_threshold":        (0.010, 0.120),
    "pinch_cooldown":         (0.10,  2.0),
    "three_finger_threshold": (0.010, 0.120),
    "three_finger_cooldown":  (0.5,   3.0),
    "scroll_deadzone":        (0.001, 0.050),
    "scroll_speed":           (1.0,   20.0),
    "scroll_vel_alpha":       (0.05,  0.80),
    "scroll_max_ticks":       (1,     20),
    "bunch_threshold":        (0.020, 0.200),
    "bunch_hold_frames":      (3,     40),
    "palm_cooldown":          (0.5,   5.0),
    "tile_move_threshold":    (0.010, 0.200),
    "tile_cooldown":          (0.1,   3.0),
    "rotation_threshold":      (0.3,   5.0),
    "rotation_cooldown":       (0.1,   2.0),
    "desktop_swipe_threshold": (0.02,  0.30),
    "desktop_swipe_cooldown":  (0.1,   3.0),
    "zoom_threshold":          (0.01,  0.20),
    "zoom_cooldown":           (0.1,   2.0),
}

UNITS: dict[str, str] = {
    "pinch_threshold":        "norm",
    "pinch_cooldown":         "sec",
    "three_finger_threshold": "norm",
    "three_finger_cooldown":  "sec",
    "scroll_deadzone":        "norm",
    "scroll_speed":           "x",
    "scroll_vel_alpha":       "alpha",
    "scroll_max_ticks":       "ticks",
    "bunch_threshold":        "norm",
    "bunch_hold_frames":      "frames",
    "palm_cooldown":          "sec",
    "tile_move_threshold":    "norm",
    "tile_cooldown":          "sec",
    "rotation_threshold":      "rad/s",
    "rotation_cooldown":       "sec",
    "desktop_swipe_threshold": "norm",
    "desktop_swipe_cooldown":  "sec",
    "zoom_threshold":          "norm",
    "zoom_cooldown":           "sec",
}

# ── PANEL STATE (list-wrapped for mutation inside callbacks) ──────────────────
_settings_open  = [False]
_panel_scroll_y = [0]
_hover_row      = [-1]
_changed        = ["", 0.0]   # [key, until]
_mouse_xy       = [0, 0]      # [x, y]
_PANEL_ROWS: list[dict] = []

# ── GESTURE PROFILES ──────────────────────────────────────────────────────────
_active_profile = ["default"]

_PROFILES: dict[str, dict[str, float | int]] = {
    "default": {
        "pinch_threshold": 0.048, "pinch_cooldown": 0.35,
        "three_finger_threshold": 0.058, "three_finger_cooldown": 1.5,
        "scroll_deadzone": 0.008, "scroll_speed": 6.0,
        "scroll_vel_alpha": 0.30, "scroll_max_ticks": 8,
        "bunch_threshold": 0.10, "bunch_hold_frames": 12,
        "palm_cooldown": 2.0, "tile_move_threshold": 0.050,
        "tile_cooldown": 0.8, "rotation_threshold": 1.8,
        "rotation_cooldown": 0.6,
        "desktop_swipe_threshold": 0.08, "desktop_swipe_cooldown": 0.6,
        "zoom_threshold": 0.06, "zoom_cooldown": 0.4,
    },
    "precise": {
        "pinch_threshold": 0.030, "pinch_cooldown": 0.25,
        "three_finger_threshold": 0.040, "three_finger_cooldown": 1.5,
        "scroll_deadzone": 0.005, "scroll_speed": 5.0,
        "scroll_vel_alpha": 0.25, "scroll_max_ticks": 6,
        "bunch_threshold": 0.08, "bunch_hold_frames": 15,
        "palm_cooldown": 2.0, "tile_move_threshold": 0.070,
        "tile_cooldown": 1.0, "rotation_threshold": 2.5,
        "rotation_cooldown": 0.8,
        "desktop_swipe_threshold": 0.12, "desktop_swipe_cooldown": 0.8,
        "zoom_threshold": 0.04, "zoom_cooldown": 0.5,
    },
    "presentation": {
        "pinch_threshold": 0.048, "pinch_cooldown": 0.35,
        "three_finger_threshold": 9.99, "three_finger_cooldown": 1.5,
        "scroll_deadzone": 0.008, "scroll_speed": 6.0,
        "scroll_vel_alpha": 0.30, "scroll_max_ticks": 8,
        "bunch_threshold": 0.10, "bunch_hold_frames": 20,
        "palm_cooldown": 3.0, "tile_move_threshold": 9.99,
        "tile_cooldown": 0.8, "rotation_threshold": 9.99,
        "rotation_cooldown": 0.6,
        "desktop_swipe_threshold": 9.99, "desktop_swipe_cooldown": 0.6,
        "zoom_threshold": 9.99, "zoom_cooldown": 0.4,
    },
}


def switch_profile(name: str) -> None:
    if name not in _PROFILES:
        return
    _active_profile[0] = name
    for key, val in _PROFILES[name].items():
        if key in _SETTINGS:
            _SETTINGS[key] = val
    _save_settings()
    _changed[0] = "__all__"
    _changed[1] = time.time() + 0.5
    print(f"[PROFILE] switched to '{name}'")


def save_custom_profile() -> None:
    _PROFILES["custom"] = dict(_SETTINGS)
    _active_profile[0]  = "custom"
    cfg = configparser.ConfigParser()
    cfg.read(CONF_PATH)
    section = "profile_custom"
    if not cfg.has_section(section):
        cfg.add_section(section)
    for key, val in _SETTINGS.items():
        cfg.set(section, key, str(val) if isinstance(val, int) else f"{val:.4f}")
    with open(CONF_PATH, "w") as f:
        cfg.write(f)
    print("[PROFILE] saved 'custom' to kontrol.conf")


# ── APP PROFILES (auto-switching) ─────────────────────────────────────────────
_APP_PROFILES: dict[str, str] = {}


def load_app_profiles() -> None:
    cfg = configparser.ConfigParser()
    cfg.read(CONF_PATH)
    _APP_PROFILES.clear()
    if cfg.has_section("app_profiles"):
        for app, profile in cfg.items("app_profiles"):
            _APP_PROFILES[app.lower()] = profile


def get_focused_app() -> str:
    try:
        r1 = subprocess.run(["xprop", "-root", "_NET_ACTIVE_WINDOW"],
                            capture_output=True, text=True, timeout=0.05)
        wid = r1.stdout.strip().split()[-1]
        if wid in ("0x0", ""):
            return ""
        r2 = subprocess.run(["xprop", "-id", wid, "WM_CLASS"],
                            capture_output=True, text=True, timeout=0.05)
        parts = r2.stdout.split('"')
        return parts[-2].lower() if len(parts) >= 2 else ""
    except Exception:
        return ""


# ── SHARED API & DIAGNOSTIC STATE ─────────────────────────────────────────────
_api_state: dict = {
    "hand_detected":  False,
    "active_gesture": "NONE",
    "fps":            0.0,
    "cursor_x":       0,
    "cursor_y":       0,
    "current_app":    "",
    "uptime_start":   time.time(),
}

_diag: dict = {
    "raw_lm8_x":    0.0, "raw_lm8_y":    0.0,
    "smooth_lm8_x": 0.0, "smooth_lm8_y": 0.0,
    "zone_nx":      0.0, "zone_ny":       0.0,
    "screen_tx":    0.0, "screen_ty":     0.0,
    "vel_px":       0.0, "ema_alpha":     0.0,
    "dx_sent":      0,   "dy_sent":       0,
}

_diag_gestures: dict = {
    "bunch_val":        0.0,
    "tile_dist":        0.0,
    "three_finger_val": 0.0,
    "swipe_dx":         0.0,
    "zoom_delta":       0.0,
    "rot_ang_vel":      0.0,
    "scroll_vel":       0.0,
    "ring_dist":        0.0,
    "index_dist":       0.0,
}

DIAGNOSTIC = [False]

# ── FLASK REST API ─────────────────────────────────────────────────────────────
if _FLASK_OK:
    _flask_app = Flask("kontrol")

    @_flask_app.route("/status")
    def api_status():
        return jsonify({
            "version":        "1.8",
            "running":        _running[0],
            "headless":       HEADLESS[0],
            "hand_detected":  _api_state["hand_detected"],
            "active_gesture": _api_state["active_gesture"],
            "profile":        _active_profile[0],
            "fps":            round(_api_state["fps"], 1),
            "cursor_x":       _api_state["cursor_x"],
            "cursor_y":       _api_state["cursor_y"],
            "uptime_s":       round(time.time() - _api_state["uptime_start"], 1),
        })

    @_flask_app.route("/gestures")
    def api_gestures():
        return jsonify({"gestures": [
            {"priority": 1, "name": "bunch",
             "description": "All 5 tips → Show Desktop",
             "threshold_key": "bunch_threshold",
             "threshold_value": _SETTINGS["bunch_threshold"], "enabled": True},
            {"priority": 2, "name": "tile",
             "description": "Pinky+Thumb + direction → KDE tile",
             "threshold_key": "tile_move_threshold",
             "threshold_value": _SETTINGS["tile_move_threshold"], "enabled": True},
            {"priority": 3, "name": "three_finger",
             "description": "Thumb+Index+Middle → KDE overview",
             "threshold_key": "three_finger_threshold",
             "threshold_value": _SETTINGS["three_finger_threshold"],
             "enabled": _SETTINGS["three_finger_threshold"] < 0.5},
            {"priority": 4, "name": "desktop_swipe",
             "description": "2 fingers extended + wrist swipe → desktop",
             "threshold_key": "desktop_swipe_threshold",
             "threshold_value": _SETTINGS["desktop_swipe_threshold"], "enabled": True},
            {"priority": 5, "name": "zoom",
             "description": "Index+middle spread/pinch → Ctrl+/-",
             "threshold_key": "zoom_threshold",
             "threshold_value": _SETTINGS["zoom_threshold"],
             "enabled": _SETTINGS["zoom_threshold"] < 9.0},
            {"priority": 6, "name": "wrist_rotation",
             "description": "CW/CCW knuckle rotation → Alt+Tab",
             "threshold_key": "rotation_threshold",
             "threshold_value": _SETTINGS["rotation_threshold"],
             "enabled": _SETTINGS["rotation_threshold"] < 9.0},
            {"priority": 7, "name": "scroll",
             "description": "Middle+Thumb + vertical → scroll",
             "threshold_key": "scroll_deadzone",
             "threshold_value": _SETTINGS["scroll_deadzone"], "enabled": True},
            {"priority": 8, "name": "right_click",
             "description": "Ring+Thumb → right click",
             "threshold_key": "pinch_threshold",
             "threshold_value": _SETTINGS["pinch_threshold"], "enabled": True},
            {"priority": 9, "name": "left_click_drag",
             "description": "Index+Thumb → left click / drag",
             "threshold_key": "pinch_threshold",
             "threshold_value": _SETTINGS["pinch_threshold"], "enabled": True},
            {"priority": 10, "name": "cursor",
             "description": "Index fingertip → cursor movement",
             "threshold_key": None, "threshold_value": None, "enabled": True},
        ]})

    @_flask_app.route("/profile", methods=["POST"])
    def api_profile():
        data = freq.get_json(silent=True) or {}
        name = data.get("name", "")
        if name not in _PROFILES:
            return jsonify({"ok": False, "error": f"profile '{name}' not found"}), 404
        switch_profile(name)
        return jsonify({"ok": True, "profile": name})

    @_flask_app.route("/setting", methods=["POST"])
    def api_setting():
        data = freq.get_json(silent=True) or {}
        key  = data.get("key", "")
        val  = data.get("value")
        if key not in _SETTINGS or val is None:
            return jsonify({"ok": False, "error": "unknown key or missing value"}), 400
        try:
            _SETTINGS[key] = int(val) if isinstance(_SETTINGS[key], int) else float(val)
            _save_settings()
            return jsonify({"ok": True, "key": key, "value": _SETTINGS[key]})
        except (ValueError, TypeError) as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @_flask_app.route("/headless", methods=["POST"])
    def api_headless():
        data = freq.get_json(silent=True) or {}
        HEADLESS[0] = bool(data.get("enabled", True))
        return jsonify({"ok": True, "headless": HEADLESS[0]})

    @_flask_app.route("/stop", methods=["POST"])
    def api_stop():
        _running[0] = False
        return jsonify({"ok": True, "message": "shutting down"})

    @_flask_app.route("/log")
    def api_log():
        log_path = Path.home() / ".local/share/kontrol.log"
        try:
            lines = log_path.read_text().splitlines()[-50:]
        except FileNotFoundError:
            lines = []
        return jsonify({"lines": lines})

    @_flask_app.route("/app-profiles")
    def api_app_profiles():
        return jsonify({
            "app_profiles": _APP_PROFILES,
            "current_app":  _api_state["current_app"],
            "auto_switched": bool(_APP_PROFILES),
        })

    @_flask_app.route("/app-profile", methods=["POST"])
    def api_app_profile():
        data    = freq.get_json(silent=True) or {}
        app_key = data.get("app", "").lower()
        profile = data.get("profile", "")
        if not app_key or profile not in _PROFILES:
            return jsonify({"ok": False, "error": "invalid app or profile"}), 400
        _APP_PROFILES[app_key] = profile
        cfg = configparser.ConfigParser()
        cfg.read(CONF_PATH)
        if not cfg.has_section("app_profiles"):
            cfg.add_section("app_profiles")
        cfg.set("app_profiles", app_key, profile)
        with open(CONF_PATH, "w") as f:
            cfg.write(f)
        return jsonify({"ok": True})

    @_flask_app.route("/diagnostic")
    def api_diagnostic():
        return jsonify({
            "gesture_states":  dict(_diag_gestures),
            "cursor_pipeline": dict(_diag),
        })

    def start_api_server(host: str = "127.0.0.1", port: int = 5555) -> None:
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        t = threading.Thread(
            target=lambda: _flask_app.run(
                host=host, port=port, debug=False, use_reloader=False
            ),
            daemon=True, name="kontrol-api",
        )
        t.start()
        print(f"[API] listening on http://{host}:{port}")

else:
    def start_api_server(host: str = "127.0.0.1", port: int = 5555) -> None:
        print("[API] Flask not available — API disabled")


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
    "right":         "Window Quick Tile Right",
    "left":          "Window Quick Tile Left",
    "up":            "Window Quick Tile Top",
    "down":          "Window Quick Tile Bottom",
    "maximize":      "Window Maximize",
    "minimize":      "Window Minimize",
    "task_view":     "Overview",
    "show_desktop":  "Show Desktop",
    "desktop_right": "Switch to Desktop to the Right",
    "desktop_left":  "Switch to Desktop to the Left",
}

_KWIN_KEYCODES: dict[str, tuple[str, ...]] = {
    "right":         ("125:1", "106:1", "106:0", "125:0"),
    "left":          ("125:1", "105:1", "105:0", "125:0"),
    "up":            ("125:1", "103:1", "103:0", "125:0"),
    "down":          ("125:1", "108:1", "108:0", "125:0"),
    "task_view":     ("125:1", "104:1", "104:0", "125:0"),
    "desktop_right": ("125:1", "29:1", "106:1", "106:0", "29:0", "125:0"),
    "desktop_left":  ("125:1", "29:1", "105:1", "105:0", "29:0", "125:0"),
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


def notify(msg: str) -> None:
    if _cfg.getboolean("system", "headless_notifications", fallback=True):
        subprocess.Popen(
            ["notify-send", "-a", "Kontrol", "-t", "3000", msg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


# ── SETTINGS HELPERS ──────────────────────────────────────────────────────────
def _in_rect(x: int, y: int, rect: tuple | None) -> bool:
    if rect is None:
        return False
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _save_settings() -> None:
    cfg = configparser.ConfigParser()
    cfg.read(CONF_PATH)
    if not cfg.has_section("gestures"):
        cfg.add_section("gestures")
    for key, val in _SETTINGS.items():
        cfg.set("gestures", key, str(val) if isinstance(val, int) else f"{val:.4f}")
    with open(CONF_PATH, "w") as f:
        cfg.write(f)
    print(f"[SETTINGS] saved {CONF_PATH}")


def _adjust_setting(key: str, direction: int) -> None:
    step    = STEP_SIZES[key]
    val     = _SETTINGS[key]
    new_val = val + direction * step
    lo, hi  = VALUE_CLAMPS[key]
    if isinstance(val, int):
        _SETTINGS[key] = int(max(lo, min(hi, round(new_val))))
    else:
        _SETTINGS[key] = round(float(max(lo, min(hi, new_val))), 4)
    _save_settings()
    _changed[0] = key
    _changed[1] = time.time() + 0.5


def _reset_settings() -> None:
    for key in _SETTINGS:
        raw = _DEFAULTS["gestures"][key]
        _SETTINGS[key] = int(raw) if "." not in raw else float(raw)
    _save_settings()
    _changed[0] = "__all__"
    _changed[1] = time.time() + 0.5
    print("[SETTINGS] reset to defaults")


def build_panel_rows() -> list[dict]:
    rows: list[dict] = []
    y = 72

    def section(name: str) -> None:
        nonlocal y
        rows.append({"is_section": True, "label": name, "y": y})
        y += 28

    def row(key: str, label: str) -> None:
        nonlocal y
        rows.append({"is_section": False, "key": key, "label": label, "y": y})
        y += 36

    section("-- PINCH GESTURES --")
    row("pinch_threshold",        "Pinch Threshold")
    row("pinch_cooldown",         "Pinch Cooldown")

    section("-- THREE FINGER --")
    row("three_finger_threshold", "3-Finger Threshold")
    row("three_finger_cooldown",  "3-Finger Cooldown")

    section("-- SCROLL --")
    row("scroll_deadzone",        "Scroll Deadzone")
    row("scroll_speed",           "Scroll Speed")
    row("scroll_vel_alpha",       "Scroll Velocity a")
    row("scroll_max_ticks",       "Scroll Max Ticks")

    section("-- TILE (PINKY+THUMB) --")
    row("tile_move_threshold",    "Tile Move Threshold")
    row("tile_cooldown",          "Tile Cooldown")

    section("-- BUNCH / PALM CLOSE --")
    row("bunch_threshold",        "Bunch Threshold")
    row("bunch_hold_frames",      "Bunch Hold Frames")
    row("palm_cooldown",          "Palm Cooldown")

    section("-- WRIST ROTATION --")
    row("rotation_threshold",     "Rotation Threshold")
    row("rotation_cooldown",      "Rotation Cooldown")

    section("-- DESKTOP SWIPE (2-FINGER) --")
    row("desktop_swipe_threshold", "Swipe Threshold")
    row("desktop_swipe_cooldown",  "Swipe Cooldown")

    section("-- ZOOM (INDEX+MIDDLE SPREAD) --")
    row("zoom_threshold", "Zoom Threshold")
    row("zoom_cooldown",  "Zoom Cooldown")

    return rows


# ── LANDMARK HELPERS ──────────────────────────────────────────────────────────
def pdist(lm, a: int, b: int) -> float:
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def is_index_thumb(lm) -> bool:
    return pdist(lm, 4, 8) < _SETTINGS["pinch_threshold"]


def is_middle_thumb(lm) -> bool:
    return pdist(lm, 4, 12) < _SETTINGS["pinch_threshold"]


def is_ring_thumb(lm) -> bool:
    return pdist(lm, 4, 16) < _SETTINGS["pinch_threshold"]


def is_pinky_thumb(lm) -> bool:
    return pdist(lm, 4, 20) < _SETTINGS["pinch_threshold"]


def is_three_finger_pinch(lm, thresh: float) -> bool:
    """Thumb (LM4) close to BOTH index (LM8) AND middle (LM12) simultaneously."""
    return pdist(lm, 4, 8) < thresh and pdist(lm, 4, 12) < thresh


def knuckle_angle(lm) -> float:
    """Angle of knuckle axis LM17→LM5 in radians. CW rotation → angle decreases."""
    dx = lm[5].x - lm[17].x
    dy = lm[5].y - lm[17].y
    return math.atan2(dy, dx)


def is_bunch(lm) -> bool:
    t = _SETTINGS["bunch_threshold"]
    return (
        pdist(lm, 4, 8)  < t and
        pdist(lm, 4, 12) < t and
        pdist(lm, 4, 16) < t and
        pdist(lm, 4, 20) < t
    )


def is_two_finger_extended(lm) -> bool:
    return (
        lm[8].y  < lm[6].y  and   # index extended
        lm[12].y < lm[10].y and   # middle extended
        lm[16].y > lm[13].y and   # ring bent
        lm[20].y > lm[17].y       # pinky bent
    )


def is_zoom_pose(lm) -> bool:
    idx_up    = lm[8].y  < lm[6].y
    mid_up    = lm[12].y < lm[10].y
    thumb_far = pdist(lm, 4, 8) > 0.08 and pdist(lm, 4, 12) > 0.08
    return idx_up and mid_up and thumb_far


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
    pinch_t = _SETTINGS["pinch_threshold"]

    for a, b in HAND_CONNECTIONS:
        color = FINGER_COLORS[LANDMARK_FINGER[max(a, b)]]
        dim   = tuple(int(c * 0.55) for c in color)
        cv2.line(frame,
                 (int(lm[a].x * fw), int(lm[a].y * fh)),
                 (int(lm[b].x * fw), int(lm[b].y * fh)),
                 dim, 1, cv2.LINE_AA)

    for i in range(21):
        px = int(lm[i].x * fw); py = int(lm[i].y * fh)
        color = FINGER_COLORS[LANDMARK_FINGER[i]]
        r     = lm_radius(lm, i)
        cv2.circle(frame, (px, py), r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), r, (220, 220, 220), 1, cv2.LINE_AA)

    for tip in (4, 8, 12, 16, 20):
        px = int(lm[tip].x * fw); py = int(lm[tip].y * fh)
        color = FINGER_COLORS[LANDMARK_FINGER[tip]]
        r     = lm_radius(lm, tip, base=6, scale=4)
        cv2.circle(frame, (px, py), r + 4, color, 2, cv2.LINE_AA)

    wx = int(lm[0].x * fw); wy = int(lm[0].y * fh)
    cv2.circle(frame, (wx, wy), 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (wx, wy), 8, (180, 180, 180), 2, cv2.LINE_AA)

    def draw_pinch_line(a: int, b: int, active_color: tuple) -> None:
        dist = pdist(lm, a, b)
        t    = max(0.0, min(1.0, 1.0 - dist / (pinch_t * 2)))
        col  = tuple(int(80 * (1 - t) + active_color[c] * t) for c in range(3))
        cv2.line(frame,
                 (int(lm[a].x * fw), int(lm[a].y * fh)),
                 (int(lm[b].x * fw), int(lm[b].y * fh)),
                 col, 1 + int(t * 3), cv2.LINE_AA)

    draw_pinch_line(4, 8,  ( 50,  50, 220))
    draw_pinch_line(4, 12, ( 50, 180,  50))
    draw_pinch_line(4, 20, ( 50, 200, 200))

    if active_pinches.get("three_finger"):
        pts = np.array([
            [int(lm[4].x * fw),  int(lm[4].y * fh)],
            [int(lm[8].x * fw),  int(lm[8].y * fh)],
            [int(lm[12].x * fw), int(lm[12].y * fh)],
        ], dtype=np.int32)
        cv2.polylines(frame, [pts], True, (220, 50, 220), 2, cv2.LINE_AA)

    zoom_delta = active_pinches.get("zoom_delta")
    if zoom_delta is not None:
        current_dist = pdist(lm, 8, 12)
        thickness = max(1, int(current_dist * 15))
        color = (50, 220, 50) if zoom_delta >= 0 else (80, 80, 220)
        cv2.line(frame,
                 (int(lm[8].x * fw),  int(lm[8].y * fh)),
                 (int(lm[12].x * fw), int(lm[12].y * fh)),
                 color, thickness, cv2.LINE_AA)

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
    h, w  = frame.shape[:2]
    now   = time.time()
    font  = cv2.FONT_HERSHEY_SIMPLEX
    bhf   = _SETTINGS["bunch_hold_frames"]

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
    cv2.rectangle(overlay, (0, 0), (290, 128), (10, 10, 10), -1)
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
    hud_line(5, "[PRF]    ", _active_profile[0])

    if palm_count > 0:
        BAR_W  = 10
        filled = min(palm_count * BAR_W // max(1, bhf), BAR_W)
        bar    = "█" * filled + "░" * (BAR_W - filled)
        cv2.putText(frame, f"[PALM] {bar} {palm_count}/{bhf}",
                    (x0, y0 + 5 * dy), font, 0.40, (0, 180, 255), 1)

    if flash_msg and now < flash_until:
        text_sz = cv2.getTextSize(flash_msg, font, 1.2, 2)[0]
        tx = (w - text_sz[0]) // 2
        ty = h // 2
        cv2.putText(frame, flash_msg, (tx, ty), font, 1.2, flash_color, 2)


# ── SETTINGS PANEL DRAW ───────────────────────────────────────────────────────
def draw_settings_panel(frame, now: float) -> None:
    fw = frame.shape[1]
    fh = frame.shape[0]
    sy = _panel_scroll_y[0]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    cv2.rectangle(frame, (0, 0), (fw, 32), (20, 20, 40), -1)
    cv2.putText(frame, "KONTROL  GESTURE SETTINGS",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)

    # ── Profile bar (y=32–64) ─────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 32), (fw, 64), (15, 15, 35), -1)
    _pnames = [p for p in ("default", "precise", "presentation", "custom")
               if p in _PROFILES]
    _pbtn_w = 90
    for _pi, _pname in enumerate(_pnames):
        _bx1 = 5 + _pi * (_pbtn_w + 5)
        _bx2 = _bx1 + _pbtn_w
        _is_act = _active_profile[0] == _pname
        _bg  = (60, 90, 60) if _is_act else (30, 30, 50)
        _bdr = (80, 180, 80) if _is_act else (70, 70, 100)
        cv2.rectangle(frame, (_bx1, 36), (_bx2, 62), _bg, -1)
        cv2.rectangle(frame, (_bx1, 36), (_bx2, 62), _bdr, 1)
        _tc = (100, 255, 100) if _is_act else (160, 160, 160)
        cv2.putText(frame, _pname[:12], (_bx1 + 4, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, _tc, 1)

    mx, my = _mouse_xy[0], _mouse_xy[1]

    for i, row in enumerate(_PANEL_ROWS):
        ry_abs = row["y"] - sy

        if ry_abs + 36 < 64 or ry_abs > fh:
            continue

        if row["is_section"]:
            cv2.rectangle(frame, (0, ry_abs), (fw, ry_abs + 26), (15, 15, 55), -1)
            cv2.putText(frame, row["label"],
                        (10, ry_abs + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 180, 255), 1)
            continue

        row_bg = (50, 50, 60) if i == _hover_row[0] else \
                 (35, 35, 35) if i % 2 else (25, 25, 25)
        cv2.rectangle(frame, (0, ry_abs), (fw, ry_abs + 36), row_bg, -1)

        cv2.putText(frame, row["label"],
                    (10, ry_abs + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        key     = row["key"]
        val     = _SETTINGS[key]
        val_str = str(val) if isinstance(val, int) else f"{val:.3f}"
        all_fl  = (_changed[0] == "__all__" and now < _changed[1])
        key_fl  = (_changed[0] == key       and now < _changed[1])
        vc      = (50, 220, 220) if (all_fl or key_fl) else (255, 255, 255)
        cv2.putText(frame, val_str, (215, ry_abs + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, vc, 1)

        mbx1, mby1, mbx2, mby2 = 295, ry_abs + 7, 323, ry_abs + 29
        m_hov = mbx1 <= mx <= mbx2 and mby1 <= my <= mby2
        cv2.rectangle(frame, (mbx1, mby1), (mbx2, mby2),
                      (80, 80, 80) if m_hov else (45, 45, 45), -1)
        cv2.rectangle(frame, (mbx1, mby1), (mbx2, mby2), (140, 140, 140), 1)
        cv2.putText(frame, "-", (mbx1 + 8, mby1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        pbx1, pby1, pbx2, pby2 = 328, ry_abs + 7, 356, ry_abs + 29
        p_hov = pbx1 <= mx <= pbx2 and pby1 <= my <= pby2
        cv2.rectangle(frame, (pbx1, pby1), (pbx2, pby2),
                      (80, 80, 80) if p_hov else (45, 45, 45), -1)
        cv2.rectangle(frame, (pbx1, pby1), (pbx2, pby2), (140, 140, 140), 1)
        cv2.putText(frame, "+", (pbx1 + 7, pby1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        unit = UNITS.get(key, "")
        cv2.putText(frame, unit, (365, ry_abs + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        cv2.line(frame, (0, ry_abs + 35), (fw, ry_abs + 35), (50, 50, 50), 1)

    total_h   = sum(26 if r["is_section"] else 36 for r in _PANEL_ROWS) + 72
    visible_h = fh - 64
    if total_h > visible_h:
        bar_h = max(20, int(visible_h * visible_h / total_h))
        travel = max(1, total_h - visible_h)
        bar_y  = 64 + int(sy * (visible_h - bar_h) / travel)
        cv2.rectangle(frame, (fw - 6, bar_y), (fw - 2, bar_y + bar_h),
                      (100, 100, 100), -1)

    reset_y = fh - 34
    cv2.rectangle(frame, (10, reset_y), (140, reset_y + 26), (40, 20, 20), -1)
    cv2.rectangle(frame, (10, reset_y), (140, reset_y + 26), (120, 60, 60), 1)
    cv2.putText(frame, "RESET DEFAULTS", (14, reset_y + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 100, 100), 1)

    cv2.putText(frame, "Changes save instantly to kontrol.conf",
                (150, fh - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)


# ── BUTTONS ───────────────────────────────────────────────────────────────────
_quit_flag              = [False]
_btn_close:  tuple | None = None
_btn_min:    tuple | None = None
_btn_settings: tuple | None = None


def _mouse_cb(event, x, y, flags, param) -> None:
    _mouse_xy[0] = x
    _mouse_xy[1] = y

    if not _settings_open[0]:
        if event == cv2.EVENT_LBUTTONDOWN:
            if _in_rect(x, y, _btn_close):
                _quit_flag[0] = True
            elif _in_rect(x, y, _btn_min):
                subprocess.Popen(
                    ["wmctrl", "-r", ":ACTIVE:", "-b", "add,hidden"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            elif _in_rect(x, y, _btn_settings):
                _settings_open[0] = True
        return

    # ── Panel is open ─────────────────────────────────────────────────────────
    if event == cv2.EVENT_LBUTTONDOWN:
        if _in_rect(x, y, _btn_close):
            _settings_open[0]  = False
            _panel_scroll_y[0] = 0
            return

        # Reset button (y range 446–472 for fh=480)
        if _in_rect(x, y, (10, 446, 140, 472)):
            _reset_settings()
            return

        # Profile bar (y=36–62, fixed — does not scroll)
        if 36 <= y <= 62:
            _pnames = [p for p in ("default", "precise", "presentation", "custom")
                       if p in _PROFILES]
            for _pi, _pname in enumerate(_pnames):
                _bx1 = 5 + _pi * 95
                if _bx1 <= x <= _bx1 + 90:
                    switch_profile(_pname)
                    break
            return

        adj_y = y + _panel_scroll_y[0]
        for row in _PANEL_ROWS:
            if row["is_section"]:
                continue
            ry = row["y"]
            if ry <= adj_y <= ry + 36:
                if _in_rect(x, adj_y, (295, ry + 7, 323, ry + 29)):
                    _adjust_setting(row["key"], -1)
                elif _in_rect(x, adj_y, (328, ry + 7, 356, ry + 29)):
                    _adjust_setting(row["key"], +1)
                break

    elif event == cv2.EVENT_MOUSEMOVE:
        adj_y = y + _panel_scroll_y[0]
        _hover_row[0] = -1
        for i, row in enumerate(_PANEL_ROWS):
            if row["is_section"]:
                continue
            if row["y"] <= adj_y <= row["y"] + 36:
                _hover_row[0] = i
                break

    elif event == cv2.EVENT_MOUSEWHEEL:
        total_h   = sum(26 if r["is_section"] else 36 for r in _PANEL_ROWS) + 72
        max_scroll = max(0, total_h - 376)
        delta = -36 if flags > 0 else 36
        _panel_scroll_y[0] = max(0, min(max_scroll, _panel_scroll_y[0] + delta))


def draw_buttons(frame) -> None:
    global _btn_close, _btn_min, _btn_settings
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

    sx1 = mx1 - pad - bw; sy1 = pad; sx2 = mx1 - pad; sy2 = pad + bh
    _btn_settings = (sx1, sy1, sx2, sy2)
    s_active = _settings_open[0]
    s_hov    = sx1 <= _mouse_xy[0] <= sx2 and sy1 <= _mouse_xy[1] <= sy2
    s_bg     = (100, 80, 20) if s_active else (80, 60, 10) if s_hov else (50, 40, 10)
    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), s_bg, -1)
    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (140, 140, 140), 1)
    cv2.putText(frame, "S", (sx1 + 8, sy2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 200, 100), 1)


# ── DIAGNOSTIC OVERLAY ───────────────────────────────────────────────────────
def draw_diagnostic(frame, lm, now: float) -> None:
    fw, fh = frame.shape[1], frame.shape[0]
    font   = cv2.FONT_HERSHEY_SIMPLEX

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (5, 5, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "DIAGNOSTIC  [D to exit]",
                (10, 20), font, 0.50, (100, 100, 255), 1)

    gesture_rows = [
        ("P1 BUNCH",   _diag_gestures["bunch_val"],    _SETTINGS["bunch_threshold"]),
        ("P2 TILE",    _diag_gestures["tile_dist"],     _SETTINGS["tile_move_threshold"]),
        ("P3 3-FIN",   _diag_gestures["three_finger_val"], _SETTINGS["three_finger_threshold"]),
        ("P4 SWIPE",   abs(_diag_gestures["swipe_dx"]), _SETTINGS["desktop_swipe_threshold"]),
        ("P5 ZOOM",    abs(_diag_gestures["zoom_delta"]), _SETTINGS["zoom_threshold"]),
        ("P6 ROT",     abs(_diag_gestures["rot_ang_vel"]), _SETTINGS["rotation_threshold"]),
        ("P7 SCROLL",  abs(_diag_gestures["scroll_vel"]), _SETTINGS["scroll_deadzone"]),
        ("P8 R-CLICK", _diag_gestures["ring_dist"],    _SETTINGS["pinch_threshold"]),
        ("P9 L-CLICK", _diag_gestures["index_dist"],   _SETTINGS["pinch_threshold"]),
    ]
    for i, (name, val, thr) in enumerate(gesture_rows):
        gy    = 38 + i * 26
        ratio = val / thr if thr > 0 else 999
        if ratio < 1.0:
            color  = (50, 220, 50)
            status = "ACTIVE"
        elif ratio < 1.5:
            color  = (50, 220, 220)
            status = f"{ratio:.1f}x"
        else:
            color  = (100, 100, 100)
            status = "---"
        cv2.putText(frame, f"{name:<10} {val:.3f}/{thr:.3f} {status}",
                    (10, gy), font, 0.36, color, 1)

    cv2.putText(frame, "CURSOR PIPELINE",
                (335, 32), font, 0.42, (180, 180, 180), 1)
    pipe_rows = [
        ("Raw LM8",    f"x={_diag['raw_lm8_x']:.3f} y={_diag['raw_lm8_y']:.3f}"),
        ("Smoothed",   f"x={_diag['smooth_lm8_x']:.3f} y={_diag['smooth_lm8_y']:.3f}"),
        ("Zone map",   f"x={_diag['zone_nx']:.3f} y={_diag['zone_ny']:.3f}"),
        ("Screen px",  f"x={_diag['screen_tx']:.0f} y={_diag['screen_ty']:.0f}"),
        ("Velocity",   f"{_diag['vel_px']:.1f} px/frame"),
        ("EMA alpha",  f"{_diag['ema_alpha']:.3f}"),
        ("Delta sent", f"dx={_diag['dx_sent']:+d} dy={_diag['dy_sent']:+d}"),
    ]
    for i, (label, val) in enumerate(pipe_rows):
        py = 50 + i * 22
        cv2.putText(frame, f"{label:<12} {val}",
                    (335, py), font, 0.36, (200, 200, 200), 1)

    if lm:
        cv2.putText(frame, "LANDMARKS (x, y, z)",
                    (10, fh - 112), font, 0.36, (140, 140, 140), 1)
        for i in range(21):
            col = i % 4
            row = i // 4
            lx  = 10  + col * 155
            ly  = fh - 96 + row * 17
            cv2.putText(frame,
                        f"L{i:02d} {lm[i].x:.2f},{lm[i].y:.2f},{lm[i].z:.2f}",
                        (lx, ly), font, 0.28, (120, 120, 120), 1)


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
    _diag["raw_lm8_x"] = lx
    _diag["raw_lm8_y"] = ly

    if raw_x_s is None:
        raw_x_s, raw_y_s = lx, ly
    raw_x_s += (lx - raw_x_s) * LANDMARK_SMOOTH
    raw_y_s += (ly - raw_y_s) * LANDMARK_SMOOTH
    _diag["smooth_lm8_x"] = raw_x_s
    _diag["smooth_lm8_y"] = raw_y_s

    nx = max(0.0, min(1.0, (raw_x_s - ZONE_X_MIN) / (ZONE_X_MAX - ZONE_X_MIN)))
    ny = max(0.0, min(1.0, (raw_y_s - ZONE_Y_MIN) / (ZONE_Y_MAX - ZONE_Y_MIN)))
    tx = nx * SCREEN_W
    ty = ny * SCREEN_H
    _diag["zone_nx"]   = nx
    _diag["zone_ny"]   = ny
    _diag["screen_tx"] = tx
    _diag["screen_ty"] = ty

    if reentry:
        cx, cy = tx, ty
        prev_tx, prev_ty = tx, ty
        prev_sent_x, prev_sent_y = int(tx), int(ty)
        return raw_x_s, raw_y_s, cx, cy, prev_tx, prev_ty, prev_sent_x, prev_sent_y

    vel_px   = math.hypot(tx - prev_tx, ty - prev_ty)
    vel_norm = vel_px / SCREEN_DIAG
    prev_tx, prev_ty = tx, ty
    _diag["vel_px"] = vel_px

    alpha = max(MIN_SMOOTH, min(MAX_SMOOTH, vel_norm * VELOCITY_SCALE))
    _diag["ema_alpha"] = alpha
    cx += (tx - cx) * alpha
    cy += (ty - cy) * alpha

    dx = round(cx - prev_sent_x)
    dy = round(cy - prev_sent_y)
    _diag["dx_sent"] = dx
    _diag["dy_sent"] = dy
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
    global KWIN_DBUS, SHOW_LM_NUMBERS, SHOW_LM_INFO, _PANEL_ROWS

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Download:\n  curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

    KWIN_DBUS   = kwin_dbus_available()
    _PANEL_ROWS = build_panel_rows()
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

    win_name = "Kontrol v1.8"
    if not HEADLESS[0]:
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, _mouse_cb)
        notify("Kontrol started")
    else:
        notify("Kontrol started (headless)")

    load_app_profiles()
    if _cfg.getboolean("system", "api_enabled", fallback=False) and _FLASK_OK:
        _api_host = _cfg.get("system", "api_host", fallback="127.0.0.1")
        _api_port = _cfg.getint("system", "api_port", fallback=5555)
        start_api_server(_api_host, _api_port)

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

    # ── GESTURE STATE — two-finger swipe (priority 4) ────────────────────────
    swipe_start_x: float | None = None
    swipe_fired:   bool         = False
    last_swipe_t:  float        = 0.0

    # ── GESTURE STATE — zoom (priority 5) ────────────────────────────────────
    zoom_ref_dist: float | None = None
    zoom_last_t:   float        = 0.0
    zoom_active:   bool         = False

    # ── GESTURE STATE — wrist rotation (priority 6) ──────────────────────────
    rot_angle_history: deque = deque(maxlen=12)
    rot_last_fired_t:  float = 0.0
    ROT_MIN_FRAMES:    int   = _cfg.getint("gestures", "rotation_min_frames")

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

    # ── AUTO PROFILE STATE ────────────────────────────────────────────────────
    _last_focus_check:    float = 0.0
    _last_focused_app:    str   = ""
    FOCUS_CHECK_INTERVAL: float = 2.0

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
        f"Kontrol v1.8  {SCREEN_W}x{SCREEN_H}"
        f"  cam=/dev/video{CAM_ID}"
        f"  pinch={_SETTINGS['pinch_threshold']}  3f={_SETTINGS['three_finger_threshold']}"
        f"  bunch={_SETTINGS['bunch_threshold']}/{_SETTINGS['bunch_hold_frames']}fr"
        f"  headless={'yes' if HEADLESS[0] else 'no'}"
        f"  [n]=LM  [i]=info  [s]=settings  [h]=headless  [1/2/3]=profile  [d]=diag  [q]=quit"
    )

    _last_ts_ms: int = 0

    with HandLandmarker.create_from_options(opts) as detector:
        while _running[0] and cap.isOpened():
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

                _diag_gestures["index_dist"]       = pd_it
                _diag_gestures["ring_dist"]         = pd_rt
                _diag_gestures["three_finger_val"]  = pd_3f
                _diag_gestures["bunch_val"]         = max(pd_it, pd_mt, pd_rt, pd_pt)
                zoom_active = False

                rot_angle_history.append((now, knuckle_angle(lm)))

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

                    active_gesture = f"PALM {palm_frames}/{_SETTINGS['bunch_hold_frames']}"

                else:
                    if palm_was_closed:
                        if (palm_frames >= _SETTINGS["bunch_hold_frames"]
                                and (now - last_palm_t) > _SETTINGS["palm_cooldown"]):
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

                            if (adx > _SETTINGS["tile_move_threshold"]
                                    or ady > _SETTINGS["tile_move_threshold"]) \
                                    and (now - last_tile_t) > _SETTINGS["tile_cooldown"]:
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
                        # ════════════════════════════════════════════════════
                        if is_three_finger_pinch(lm, _SETTINGS["three_finger_threshold"]):
                            three_finger_active = True
                            was_gesturing       = True

                            if (not three_finger_held
                                    and (now - last_three_finger_t)
                                        > _SETTINGS["three_finger_cooldown"]):
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
                            # PRIORITY 4 — TWO-FINGER SWIPE → DESKTOP
                            # ════════════════════════════════════════════════
                            if is_two_finger_extended(lm) and not swipe_fired:
                                was_gesturing = True
                                if swipe_start_x is None:
                                    swipe_start_x = lm[8].x

                                dx_swipe = lm[8].x - swipe_start_x
                                _diag_gestures["swipe_dx"] = dx_swipe
                                if (abs(dx_swipe) > _SETTINGS["desktop_swipe_threshold"]
                                        and (now - last_swipe_t)
                                            > _SETTINGS["desktop_swipe_cooldown"]):
                                    if dx_swipe > 0:
                                        fire_kwin("desktop_right")
                                        flash_msg = "DESKTOP →"
                                    else:
                                        fire_kwin("desktop_left")
                                        flash_msg = "← DESKTOP"
                                    flash_color    = (0, 220, 220)
                                    flash_until    = now + 0.6
                                    swipe_fired    = True
                                    last_swipe_t   = now
                                    active_gesture = flash_msg
                                else:
                                    active_gesture = f"SWIPE  dx={dx_swipe:+.3f}"

                            elif is_zoom_pose(lm):
                                # ════════════════════════════════════════════
                                # PRIORITY 5 — ZOOM (INDEX+MIDDLE SPREAD)
                                # ════════════════════════════════════════════
                                was_gesturing = True
                                zoom_active   = True
                                if zoom_ref_dist is None:
                                    zoom_ref_dist = pdist(lm, 8, 12)

                                cur_dist      = pdist(lm, 8, 12)
                                zd            = cur_dist - zoom_ref_dist
                                zoom_ref_dist = cur_dist
                                _diag_gestures["zoom_delta"] = zd

                                if (abs(zd) > _SETTINGS["zoom_threshold"]
                                        and (now - zoom_last_t) > _SETTINGS["zoom_cooldown"]):
                                    if zd > 0:
                                        subprocess.run(
                                            ["ydotool", "key",
                                             "29:1", "78:1", "78:0", "29:0"],
                                            env=os.environ,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL,
                                        )
                                        flash_msg = "ZOOM IN"
                                    else:
                                        subprocess.run(
                                            ["ydotool", "key",
                                             "29:1", "74:1", "74:0", "29:0"],
                                            env=os.environ,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL,
                                        )
                                        flash_msg = "ZOOM OUT"
                                    flash_color    = (50, 220, 50)
                                    flash_until    = now + 0.6
                                    zoom_last_t    = now
                                    active_gesture = flash_msg
                                else:
                                    active_gesture = f"ZOOM  d={zd:+.3f}"

                            else:
                                swipe_start_x = None
                                swipe_fired   = False
                                zoom_ref_dist = None

                                # ════════════════════════════════════════════
                                # PRIORITY 6 — WRIST ROTATION → ALT+TAB
                                # ════════════════════════════════════════════
                                rotation_direction = None

                                if len(rot_angle_history) >= ROT_MIN_FRAMES:
                                    t_old, a_old = rot_angle_history[0]
                                    t_new, a_new = rot_angle_history[-1]
                                    rot_dt = t_new - t_old

                                    if rot_dt > 0.05:
                                        raw_delta = a_new - a_old
                                        if raw_delta > math.pi:
                                            raw_delta -= 2 * math.pi
                                        elif raw_delta < -math.pi:
                                            raw_delta += 2 * math.pi

                                        ang_vel = raw_delta / rot_dt
                                        _diag_gestures["rot_ang_vel"] = ang_vel

                                        if abs(ang_vel) > _SETTINGS["rotation_threshold"]:
                                            rotation_direction = "CW" if ang_vel < 0 else "CCW"

                                if rotation_direction is not None:
                                    if (now - rot_last_fired_t) > _SETTINGS["rotation_cooldown"]:
                                        if rotation_direction == "CW":
                                            subprocess.run(
                                                ["ydotool", "key",
                                                 "56:1", "15:1", "15:0", "56:0"],
                                                env=os.environ,
                                                stdout=subprocess.DEVNULL,
                                                stderr=subprocess.DEVNULL,
                                            )
                                            flash_msg      = "ALT+TAB →"
                                            flash_color    = (200, 200, 50)
                                            flash_until    = now + 0.6
                                            active_gesture = "ROT CW → ALT+TAB"
                                        else:
                                            subprocess.run(
                                                ["ydotool", "key",
                                                 "42:1", "56:1", "15:1", "15:0", "42:0", "56:0"],
                                                env=os.environ,
                                                stdout=subprocess.DEVNULL,
                                                stderr=subprocess.DEVNULL,
                                            )
                                            flash_msg      = "← ALT+TAB"
                                            flash_color    = (200, 200, 50)
                                            flash_until    = now + 0.6
                                            active_gesture = "ROT CCW ← ALT+TAB"

                                        rot_last_fired_t = now
                                        rot_angle_history.clear()

                                    was_gesturing = True

                                else:
                                    # ════════════════════════════════════════
                                    # PRIORITY 6 — MIDDLE+THUMB → SCROLL
                                    # ════════════════════════════════════════
                                    if is_middle_thumb(lm):
                                        was_gesturing = True

                                        if not mt_held:
                                            mt_held      = True
                                            scroll_ref_y = lm[0].y
                                            scroll_vel   = 0.0

                                        dy_norm      = lm[0].y - scroll_ref_y
                                        scroll_ref_y = lm[0].y
                                        scroll_vel   = (
                                            scroll_vel * (1.0 - _SETTINGS["scroll_vel_alpha"])
                                            + dy_norm  * _SETTINGS["scroll_vel_alpha"]
                                        )
                                        _diag_gestures["scroll_vel"] = scroll_vel

                                        if abs(scroll_vel) > _SETTINGS["scroll_deadzone"]:
                                            ticks = max(1, min(
                                                _SETTINGS["scroll_max_ticks"],
                                                int(abs(scroll_vel) * _SETTINGS["scroll_speed"])
                                            ))
                                            wheel_y = -ticks if scroll_vel < 0 else ticks
                                            ydocall("mousemove", "--wheel",
                                                    "-x", "0", "-y", str(wheel_y))
                                            active_gesture = (
                                                f"SCROLL {'UP' if scroll_vel < 0 else 'DOWN'}"
                                                f" x{ticks}"
                                            )
                                        else:
                                            active_gesture = f"SCROLL HOLD  M={pd_mt:.3f}"

                                    else:
                                        if mt_held:
                                            mt_held      = False
                                            scroll_ref_y = None
                                            scroll_vel   = 0.0

                                        # ════════════════════════════════════
                                        # PRIORITY 7 — RING+THUMB → RIGHT CLICK
                                        # ════════════════════════════════════
                                        if is_ring_thumb(lm):
                                            if not rt_held and \
                                                    (now - last_rclick_t) > _SETTINGS["pinch_cooldown"]:
                                                if not _kontrol_is_active(win_name):
                                                    right_click()
                                                last_rclick_t  = now
                                                active_gesture = f"R-CLICK  R={pd_rt:.3f}"
                                            rt_held       = True
                                            was_gesturing = True

                                        else:
                                            rt_held = False

                                            # ════════════════════════════════
                                            # PRIORITY 8 — INDEX+THUMB → CLICK/DRAG
                                            # ════════════════════════════════
                                            if is_index_thumb(lm):
                                                if not it_held:
                                                    it_held    = True
                                                    it_start_t = now

                                                held_t = now - it_start_t
                                                if (held_t > _SETTINGS["pinch_cooldown"]
                                                        and not drag_active
                                                        and (now - last_drag_end_t)
                                                            > _SETTINGS["pinch_cooldown"]):
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
                                                    elif ((now - it_start_t) < _SETTINGS["pinch_cooldown"]
                                                          and (now - last_drag_end_t)
                                                              > _SETTINGS["pinch_cooldown"]):
                                                        mouse_down()
                                                        mouse_up()
                                                        active_gesture = "L-CLICK"
                                                    it_held = False

                                                # ════════════════════════════
                                                # PRIORITY 9 — CURSOR
                                                # ════════════════════════════
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
                    "zoom_delta":   _diag_gestures["zoom_delta"] if zoom_active else None,
                })
                if DIAGNOSTIC[0]:
                    draw_diagnostic(frame, lm, now)
                draw_zone_warning(frame, lm, fw_px, fh_px)
                if SHOW_LM_INFO:
                    draw_lm_info(frame, lm, fw_px, fh_px)
                    angle_deg = math.degrees(knuckle_angle(lm))
                    cv2.putText(frame, f"[ROT] {angle_deg:+.1f}°",
                                (max(0, fw_px - 200), 112),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 220, 255), 1)

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
                three_finger_held = False
                scroll_ref_y      = None
                scroll_vel        = 0.0
                palm_frames       = 0
                palm_was_closed   = False
                rot_angle_history.clear()
                tile_fired        = False
                swipe_start_x     = None
                swipe_fired       = False
                zoom_ref_dist     = None
                zoom_active       = False
                was_gesturing     = True
                active_gesture    = "NO HAND"

            _api_state["fps"]           = round(fps, 1)
            _api_state["hand_detected"] = bool(result.hand_landmarks)
            _api_state["active_gesture"] = active_gesture
            _api_state["cursor_x"]      = int(prev_sent_x)
            _api_state["cursor_y"]      = int(prev_sent_y)

            if now - _last_focus_check > FOCUS_CHECK_INTERVAL:
                _last_focus_check = now
                _focused = get_focused_app()
                _api_state["current_app"] = _focused
                if _focused and _focused != _last_focused_app:
                    _last_focused_app = _focused
                    if _focused in _APP_PROFILES:
                        _tgt = _APP_PROFILES[_focused]
                        if _tgt != _active_profile[0]:
                            switch_profile(_tgt)

            if not HEADLESS[0]:
                if _settings_open[0]:
                    draw_settings_panel(frame, now)
                else:
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
                ctrl = key & 0x1F if key != 0xFF else 0xFF
                if _quit_flag[0] or key == ord("q"):
                    print("Quit.")
                    break
                elif key == ord("n"):
                    SHOW_LM_NUMBERS = not SHOW_LM_NUMBERS
                    print(f"[HUD] landmark numbers {'ON' if SHOW_LM_NUMBERS else 'OFF'}")
                elif key == ord("i"):
                    SHOW_LM_INFO = not SHOW_LM_INFO
                    print(f"[HUD] LM info overlay {'ON' if SHOW_LM_INFO else 'OFF'}")
                elif key in (ord("s"), ord("S")):
                    _settings_open[0] = not _settings_open[0]
                    if not _settings_open[0]:
                        _panel_scroll_y[0] = 0
                    print(f"[SETTINGS] panel {'open' if _settings_open[0] else 'closed'}")
                elif key in (ord("h"), ord("H")):
                    HEADLESS[0] = True
                    cv2.destroyAllWindows()
                    notify("Kontrol running headless")
                    print("[HEADLESS] window closed — running headless")
                elif key == ord("1"):
                    switch_profile("default")
                    flash_msg   = "PROFILE: default"
                    flash_color = (100, 220, 100)
                    flash_until = now + 1.0
                elif key == ord("2"):
                    switch_profile("precise")
                    flash_msg   = "PROFILE: precise"
                    flash_color = (100, 220, 100)
                    flash_until = now + 1.0
                elif key == ord("3"):
                    switch_profile("presentation")
                    flash_msg   = "PROFILE: presentation"
                    flash_color = (100, 220, 100)
                    flash_until = now + 1.0
                elif ctrl == 19 and _settings_open[0]:  # Ctrl+S
                    save_custom_profile()
                    flash_msg   = "PROFILE: custom saved"
                    flash_color = (100, 220, 100)
                    flash_until = now + 1.0
                elif key in (ord("d"), ord("D")):
                    DIAGNOSTIC[0] = not DIAGNOSTIC[0]
                    print(f"[DIAG] diagnostic mode {'ON' if DIAGNOSTIC[0] else 'OFF'}")
            else:
                if not _running[0]:
                    break

            elapsed   = time.perf_counter() - frame_start
            remaining = FRAME_INTERVAL - elapsed
            if remaining > 0.002:
                time.sleep(remaining)

    cap.release()
    if not HEADLESS[0]:
        cv2.destroyAllWindows()
    notify("Kontrol stopped")


if __name__ == "__main__":
    run()
