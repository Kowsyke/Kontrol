# Kontrol — Change Log

## v0.5.1 — 2026-04-12 (Session 5 patch)

Four regressions diagnosed and fixed.

### Regression 1 — Cursor too slow to cover both monitors
**Root cause:** `velocity_scale=2.0` + `max_smooth=0.18` left over from pre-zone-mapping
tuning.  Zone mapping compresses input range to ~70% of camera frame, which reduces
per-frame pixel deltas → lower `vel_norm` → EMA factor capped at 0.18 even during fast
sweeps → heavily damped motion.

**Fix:** Config only.
| Key | Before | After |
|---|---|---|
| `landmark_smooth` | 0.4 | 0.3 |
| `max_smooth` | 0.18 | 0.35 |
| `velocity_scale` | 2.0 | 3.5 |

### Regression 2 — Scroll command silently failing
**Root cause:** `ydotool mousemove --wheel -y <N>` always exits 1 on this build.
ydotool's `mousemove` parser requires `-x` to be provided when using `--wheel`; without
it the argument set is considered invalid and the command prints usage and exits.
Both `scroll_up` and `scroll_down` were silently dead — every scroll tick was dropped.
Diagnosed via `CHECK 2`: `--wheel -y -3` → exit 1, `--wheel -x 0 -y -3` → exit 0.

**Fix:** Code — added `-x 0` to both functions.
```python
# before:  ydocall("mousemove", "--wheel", "-y", str(-ticks))
# after:   ydocall("mousemove", "--wheel", "-x", "0", "-y", str(-ticks))
```

### Regression 3 — Scroll pose too strict
**Root cause:** `is_scroll_pose` checked ring and pinky curl against their **PIP** joints
(landmarks 14 and 18).  A natural two-finger scroll hold only curls fingers to roughly
mid-range — tip often stays above PIP, so the pose never registered.

**Fix:** Code — relaxed ring and pinky curl check to **MCP** joints (landmarks 13 and 17).
```python
# before:  not is_finger_extended(lm, 16, 14)   # ring tip vs ring PIP
#          not is_finger_extended(lm, 20, 18)   # pinky tip vs pinky PIP
# after:   not is_finger_extended(lm, 16, 13)   # ring tip vs ring MCP (more lenient)
#          not is_finger_extended(lm, 20, 17)   # pinky tip vs pinky MCP
```
Also lowered `scroll_deadzone 0.012 → 0.010` and `scroll_speed 8.0 → 6.0` for smoother
per-tick feel.

### Regression 4 — Flick detection unreachable at 20 fps
**Root cause:** `flick_min_velocity=2.0` norm/s with `flick_window_ms=120` at 20 fps =
~2.4 frames of data.  Maximum realistic wrist displacement in 120 ms ≈ 0.08 normalized
units → peak velocity ≈ 0.67 norm/s, well below the 2.0 threshold.  Combined with
`flick_axis_ratio=3.0`, no deliberate flick could ever pass both gates.

**Fix:** Config only.
| Key | Before | After |
|---|---|---|
| `flick_min_velocity` | 2.0 | 1.2 |
| `flick_window_ms` | 120 | 150 |
| `flick_axis_ratio` | 3.0 | 2.0 |

---

## v0.5 — 2026-04-12 (Session 5) — "Stark Pass"

### Zone-based hand mapping
Active tracking zone: `[0.15–0.85 x, 0.10–0.90 y]` in camera-normalized space.
The hand no longer needs to travel to the physical frame edge to reach the screen edge —
the zone maps linearly to the full screen, so 70% of lateral travel covers 100% of width.
New `[mapping]` config section:
```ini
zone_x_min = 0.15  |  zone_x_max = 0.85
zone_y_min = 0.10  |  zone_y_max = 0.90
edge_boost  = 1.8
```
`edge_boost` multiplies the velocity-scale factor when the mapped cursor is within 10% of
a screen edge, allowing snappy corner-reaching without leaving the zone.

### Double EMA smoothing
Two-stage pipeline replacing the single velocity-adaptive EMA:

| Stage | What | Alpha |
|---|---|---|
| 1 — Landmark pre-filter | Fixed EMA on raw LM 8 (x,y) before zone mapping | `landmark_smooth = 0.4` |
| 2 — Screen-space EMA | Velocity-adaptive factor after zone mapping | `clamp(vel_norm × scale × boost, min, max)` |

Updated Stage 2 parameters: `max_smooth 0.22 → 0.18`, `velocity_scale 1.5 → 2.0`.

### Strict gesture priority chain
Dispatcher rebuilt as a clean `if/elif` ladder — no nested conditional tangles:

```
fist → lock → wrist_rotate → scroll → flick → right_click → left_pinch → cursor
```

Wrist-rotate is now checked before scroll (previously scroll had higher priority).

### In-frame ✕ / — buttons
Two buttons drawn top-right of the cv2 preview window; mouse-clickable via `setMouseCallback`:
- **✕** sets `_quit_flag[0] = True` → main loop exits cleanly.
- **—** calls `wmctrl -r :ACTIVE: -b add,hidden` → iconifies the active KDE window.

### Single-hand confidence raised
`min_hand_presence_confidence`: 0.60 → **0.70** (detection stays 0.70, tracking stays 0.60).
Reduces ghost-hand detections from reflective surfaces and out-of-frame partial hands.

### Performance
| Change | Detail |
|---|---|
| `ydocall` fire-and-forget | Cursor/click calls use `subprocess.Popen`; key sequences use `subprocess.run` (blocking) |
| 20 fps cap | `time.sleep(remaining)` at end of each frame loop — prevents CPU saturation at idle |
| Rolling FPS EMA | Per-frame `fps = fps × 0.9 + instant_fps × 0.1` — display no longer resets every second |

### Hand visibility warning
When `lm[8]` (index tip) enters the outer 15% of the camera frame, a directional arrow
is drawn from the fingertip toward the frame centre, labelled **MOVE IN**.
Cues the user to reposition before they exit the active tracking zone.

### Smooth cursor re-entry
New `hand_was_present` bool. On the first frame after the hand re-enters the camera view:
- `raw_x_smooth` / `raw_y_smooth` are re-initialized to the current landmark position.
- `cx`, `cy`, `prev_tx/ty`, `prev_sent_x/y` are all reset to the mapped position.
- No cursor delta is sent on that frame — eliminates the jump-to-corner artifact on re-entry.

### Startup sound
Triple ascending tone — 880 → 1047 → 1319 Hz — replaces v0.4 double-beep.

---

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
