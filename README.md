# Kontrol

Hand-gesture mouse controller for Fedora 43 KDE Plasma 6 (Wayland).  
Runs on CPU-only hardware — no GPU required.

MediaPipe `HandLandmarker` → `ydotool` subprocess calls.  
No X11, no pynput, no root. Wayland-native via XWayland + uinput.

---

## Gesture Map

| Gesture | Detection | Action |
|---------|-----------|--------|
| Index fingertip position | LM 8 xy | Move cursor (velocity-adaptive EMA) |
| Pinch index + thumb | LM 8+4 dist < threshold | **Left click** |
| Pinch ring + thumb | LM 16+4 dist < threshold | **Right click** |
| Index + middle extended, ring + pinky curled — hand up | LM 8,12 above PIP; LM 16,20 curled | **Scroll up** |
| Index + middle extended, ring + pinky curled — hand down | same pose, finger moves down | **Scroll down** |
| Wrist flick right | LM 0 velocity > threshold, dominant axis right | **Meta+Right** (tile right) |
| Wrist flick left | LM 0 velocity > threshold, dominant axis left | **Meta+Left** (tile left) |
| Wrist flick up | LM 0 velocity > threshold, dominant axis up | **Meta+Up** (maximise) |
| Wrist flick down | LM 0 velocity > threshold, dominant axis down | **Meta+Down** (restore/tile down) |
| Fist (hold 6 frames) | All 4 tips below MCP joints | **Toggle tracking lock** |

**Lock state:**  
- Green border + "TRACKING" = normal operation  
- Red border + "LOCKED" = cursor frozen, all gestures suppressed except fist to unlock

---

## HUD (preview window, top-left)

```
FPS      24.3
Mode     TRACKING
Gesture  CURSOR
L-pinch  0.087
R-pinch  0.231
```

---

## Hardware Requirements

- **Webcam**: UVC-compatible USB webcam on `/dev/video0`  
  (built-in webcam disabled on Hyperspace — Logitech C920s or equivalent)
- **OS**: Fedora 43, KDE Plasma 6, Wayland compositor  
- **ydotoold**: must be running as user service  
- **Display**: XWayland (`DISPLAY=:0`) — cv2 preview requires XCB Qt backend

---

## Install

```bash
# 1. Clone / navigate to project
cd /home/K/Storage/Projects/Kontrol

# 2. Create venv (Python 3.14)
python3 -m venv venv
source venv/bin/activate

# 3. Install deps
pip install mediapipe opencv-python

# 4. Download MediaPipe model (float16, ~7.8 MB) — gitignored
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# 5. Ensure ydotoold is running
systemctl --user enable --now ydotoold.service
```

---

## Launch

```bash
# Via alias (set up in ~/.zshrc):
kontrol       # start in background, logs to ~/.local/share/kontrol.log
kontroloff    # stop

# Directly:
cd /home/K/Storage/Projects/Kontrol
./run.sh
```

`run.sh` sets `DISPLAY=:0 QT_QPA_PLATFORM=xcb YDOTOOL_SOCKET=/run/user/1000/.ydotool_socket` and activates the venv.

---

## Configuration

All tuning knobs live in `kontrol.conf` (auto-created on first run):

```ini
[screen]
width  = 1920
height = 1080

[camera]
id   = 0
flip = true

[smoothing]
min_smooth     = 0.08   # EMA floor — stable aim when stationary
max_smooth     = 0.35   # EMA ceiling — snappy on fast swipes
velocity_scale = 2.5    # how aggressively speed maps to smoothing factor

[gestures]
pinch_threshold    = 0.05   # normalized distance to trigger a click
pinch_cooldown     = 0.4    # seconds between clicks (per button)
scroll_deadzone    = 0.012  # min y-delta before scrolling fires
scroll_speed       = 8.0    # ticks = int(abs(delta_y) * scroll_speed)
flick_min_velocity = 450    # px/s wrist velocity to register tiling flick
flick_window_ms    = 120    # measurement window for flick velocity (ms)
flick_cooldown     = 0.8    # cooldown between tiling gestures
fist_hold_frames   = 6      # consecutive frames fist must be held to toggle lock

[system]
ydotool_socket = /run/user/1000/.ydotool_socket
abs_scale_x    = 1.0   # set to 32767/1920 if cursor lands in wrong spot
abs_scale_y    = 1.0
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mediapipe | 0.10.33 | Hand landmark detection |
| opencv-python | 4.13 | Camera capture + preview window |
| ydotool | 1.0.4 | Mouse / keyboard injection (uinput) |

Python stdlib only for sound (`wave`, `struct`, `subprocess`).

---

## Known Issues

### Scroll direction may be inverted
`ydotool mousemove --wheel -y N` — negative N scrolls up in most apps under
XWayland. If your app scrolls the wrong direction, negate `scroll_speed` in
`kontrol.conf` (e.g. `-8.0`).

### Wrist flick sensitivity
At `flick_min_velocity = 450` px/s, aggressive cursor sweeps can occasionally
trigger a tiling flick. Raise to `600`–`800` in `kontrol.conf` if you get
false positives. Tiling is suppressed while in scroll pose.

### ydotool key sends raw keycodes — not keysym strings
This version of ydotool (1.0.4) does **not** accept X11 keysym names like
`super+Right`. Tiling shortcuts are hardcoded as raw Linux input keycodes
from `input-event-codes.h`:

```
Super+Right: 125:1 106:1 106:0 125:0
Super+Left:  125:1 105:1 105:0 125:0
Super+Up:    125:1 103:1 103:0 125:0
Super+Down:  125:1 108:1 108:0 125:0
```

If your KDE tiling shortcuts differ from `Meta+Arrow`, modify `_TILING_KEYS`
in `kontrol.py` with the appropriate keycodes.

### KDE Polonium tiling layout
Tiling keys work with stock KDE window tiling. Polonium uses the same
`Meta+Arrow` bindings by default. If Polonium overrides these, check
KDE Settings → Shortcuts → KWin.

### Preview window requires XWayland
`cv2.imshow` bundles a Qt backend that has no Wayland plugin. `run.sh`
forces `DISPLAY=:0 QT_QPA_PLATFORM=xcb` (XCB/XWayland). The window
will appear on your primary Wayland output via XWayland.

### No dedicated GPU
All inference runs on Intel UHD 620 CPU. Tested at 20–25 FPS sustained
on i5-8250U with the float16 `hand_landmarker.task` model. If FPS drops
below ~15, reduce camera resolution in the `cap.set()` calls (try 320×240).

---

## Project Context

**Hyperspace** — ASUS X510UAR, Fedora 43, KDE Plasma 6 Wayland  
i5-8250U (8 threads, 3.4 GHz boost), 16 GB DDR4, Intel UHD 620  
CPU-only inference — no dedicated GPU.

Part of **Project Akira + Kontrol** — a personal AI-integrated computing
environment. Kontrol is the gesture input layer; Akira (separate repo) will
provide the voice + LLM orchestration layer.
