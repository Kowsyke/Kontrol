# Kontrol — Change Log

## v0.4 — 2026-04-12 (Session 4)

### Camera: brightness / flicker fix
- **Root cause was `power_line_frequency = 60 Hz` on UK 50 Hz mains.**
  Anti-flicker algorithm was fighting the wrong frequency, causing periodic
  brightness pulsing under artificial light. Fixed to `50 Hz`.
- `exposure_time_absolute` raised 150 → 250 (brighter, still holds 30 fps).
- `gain` raised 64 → 80.
- `backlight_compensation` enabled (1) — evens out bright backgrounds / windows behind hand.

### Camera: settings now locked after OpenCV opens device
Previously `cam-tune.sh` ran in `run.sh` before Python started. `cv2.VideoCapture()`
resets UVC controls to driver defaults when it opens the device, silently undoing all
tuning. `cam-tune.sh` is now called from inside `kontrol.py` immediately after
`VideoCapture` opens and configures the capture — settings can no longer be clobbered.

### Cursor: smoother movement
| Parameter | Before | After | Effect |
|---|---|---|---|
| `min_smooth` | 0.08 | 0.05 | Heavier smoothing at rest, less jitter |
| `max_smooth` | 0.35 | 0.22 | Less snappy at peak speed |
| `velocity_scale` | 2.5 | 1.5 | Gentler ramp, stays near min longer |
| `cursor_deadzone_px` | — | 2 | Drops ≤1 px moves — kills landmark jitter when still |

### Drag gesture (left pinch hold)
Left pinch (index + thumb) redesigned from atomic click to mousedown/mouseup split:
- `ydotool click 0x40` on pinch entry (button held).
- Cursor moves normally while button is held → drags windows.
- `ydotool click 0x80` on pinch release.
- Quick tap still acts as a left click; no separate mode needed.
- Drag is force-released on scroll/rotate entry, lock toggle, right-click, and hand-lost.

### Flick false-fire fix
Three compounding bugs fixed:

| Bug | Fix |
|---|---|
| Velocity threshold 450 px/s was hit during normal cursor sweeps | Switched to normalized units; threshold 2.0 norm/s — real flick: 2.5–5.0 |
| No axis-purity check — diagonal sweeps triggered tiling | Added `flick_axis_ratio = 3.0` (dominant/minor axis ratio must exceed this) |
| wrist_history stored in screen pixels (4480:1440 aspect) — 45° diagonals looked axis-pure | History now in normalized 0–1 coords |
| Drag momentum could trigger flick after drop | `wrist_history.clear()` on drag-end |

---

## v0.3 — 2026-04-12 (Session 3)

### Hardware: Logitech C920 + dual monitor
- Built-in IMC cam (13d3:5a07) permanently disabled via udev rule
  (`/etc/udev/rules.d/99-disable-builtin-cam.rules`); runtime deauthorize via sysfs.
- `cam-tune.sh` created: manual exposure, fixed focus 30 (≈60 cm), gain 64.
- `kontrol.conf`: `camera.id = 2`, `screen.width = 4480` (1920 built-in + 2560 QHZ), `height = 1440`.

### Cursor tracking fix: ydotool EV_ABS non-functional
`ydotoold` virtual device (`/dev/input/event26`) has no `EV_ABS` capability
(`ev = 0x07` = SYN+KEY+REL only). Every `ydotool mousemove --absolute` was silently
dropped at the kernel level. Switched to `ydotool mousemove -x dx -y dy` (relative).
`prev_sent_x/y` state tracks last sent integer position; EMA smoothing unchanged.

### New gestures
| Gesture | Action |
|---|---|
| Wrist rotate — fingers pointing down (LM 12 y > LM 0 y + 0.10), hold 8 frames | KDE task overview (Meta+PgUp) |
| Fist medium hold (6–11 frames, released) | Minimize window (Meta+Down) |
| Fist long hold (≥ 12 frames, sustained) | Toggle tracking lock (was: ≥ 6 frames) |

### Config additions
```ini
[gestures]
fist_minimize_frames = 6
fist_lock_frames     = 12
taskview_hold_frames = 8
taskview_cooldown    = 1.5
```

### HUD additions
- Fist hold progress bar: `FIST [████░░░░░░░░] 4/12`
- `TASK VIEW` flash overlay (1 s) on task-view trigger.
- `MINIMIZE` flash overlay (0.5 s) on minimize trigger.

---

## v0.2 — 2026-04-11 (Session 2)

### Externalized config (`kontrol.conf`)
All tuning knobs moved from hard-coded constants to `kontrol.conf` (INI, configparser).
File is auto-created with defaults if missing.

### Velocity-adaptive EMA smoothing
Replaced fixed `SMOOTH = 0.2` with a velocity-adaptive factor:
```
smooth = clamp(vel_norm * velocity_scale, min_smooth, max_smooth)
vel_norm = euclidean_delta / screen_diagonal
```
Cursor is snappier when moving fast, smoother when near-still.

### New gestures
| Gesture | Action |
|---|---|
| Index + middle extended, ring + pinky curled | Scroll (velocity-scaled) |
| Wrist flick (LM 0 velocity > threshold) | KDE window tiling (Meta+Arrow) |
| Ring + thumb pinch (LM 16+4) | Right click |
| Fist (all 4 tips below MCP joints), hold ≥ 6 frames | Toggle tracking lock |

### HUD overlay
Live preview window now shows: FPS, mode (TRACKING / LOCKED), active gesture,
left/right pinch distances. Border turns blue when locked.

### Startup sound
880 Hz double-beep via `aplay` on launch. Pure stdlib (wave + struct), no deps.

---

## v0.1 — 2026-04-11 (Session 1)

### What was built
Hand-gesture mouse controller for Hyperspace (ASUS X510UAR, Fedora 43, KDE Wayland).

### Stack
- **Input**: MediaPipe Tasks API `HandLandmarker` (v0.10.33) — model: `hand_landmarker.task` (float16, 7.8 MB)
- **Output**: `ydotool` subprocess calls — no pynput (fails on Python 3.14)
- **Display**: `cv2.imshow` via XWayland (`DISPLAY=:0`, `QT_QPA_PLATFORM=xcb`)
- **Socket**: ydotoold user service at `/run/user/1000/.ydotool_socket`

### Key decisions & issues resolved

| Issue | Resolution |
|-------|-----------|
| `pynput` fails on Python 3.14 | Dropped entirely; use `ydotool` subprocess only |
| `mp.solutions` AttributeError | MediaPipe 0.10+ removed legacy API; rewrote to use `mp.tasks.vision.HandLandmarker` |
| `hand_landmarker.task` missing | Downloaded from Google storage: `mediapipe-models/.../float16/1/hand_landmarker.task` |
| `cv2.imshow` — no Wayland Qt plugin | Force XCB: `DISPLAY=:0 QT_QPA_PLATFORM=xcb` |
| ydotoold socket path | Must run as user K, socket at `/run/user/1000/.ydotool_socket` |

### Gesture map
| Gesture | Action |
|---------|--------|
| Index fingertip (landmark 8) position | Cursor moves |
| Pinch index ↔ thumb (landmarks 8 ↔ 4, dist < 0.05) | Left click |

### Launch
```bash
cd /home/K/Storage/Projects/Kontrol
./run.sh
```

### Known warnings (harmless)
- `inference_feedback_manager.cc` — MediaPipe internal, ignored
- `landmark_projection_calculator.cc` — cosmetic, tracking still works
- `QFontDatabase: Cannot find font directory` — cv2 Qt font path issue, display works fine
