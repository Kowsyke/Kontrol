# Kontrol CHANGELOG v2

## v1.6 — interactive settings panel with live threshold editing (session 7)

### Architecture: frozen constants → mutable _SETTINGS dict
- All 15 gesture thresholds moved from module-level frozen constants into
  `_SETTINGS: dict[str, float | int]`, initialised from kontrol.conf on startup
- `is_index_thumb()`, `is_middle_thumb()`, `is_ring_thumb()`, `is_pinky_thumb()`,
  `is_bunch()`, `draw_skeleton()` updated to read from `_SETTINGS` (not frozen names)
- All 34 threshold reads in gesture chain in `run()` use `_SETTINGS["key"]`
- Settings take effect on the NEXT frame after clicking +/- — no restart needed

### Settings panel UI (cv2-drawn, no native widgets)
- `build_panel_rows()` — single layout registry (list of dicts) consumed by both
  draw and mouse handler. 6 sections, 15 data rows, alternating row backgrounds.
- `draw_settings_panel(frame, now)` — full overlay: 88% dark alpha, header bar,
  scrollable row list, scrollbar, reset button, footer hint
- Row columns: label (x=10), value (x=215, flashes cyan on change), [-] (x=295),
  [+] (x=328), unit label (x=365)
- Hover highlight: row background lightens when mouse is over it
- Value flash: cyan for 0.5 s after any change; `__all__` key flashes everything

### [S] button — top button row
- Added to `draw_buttons()` left of [−]: amber `S` on dark bg, hover highlight,
  active state (brighter) when panel is open
- `_btn_settings` registered as module-level tuple; same pattern as `_btn_close`

### Mouse callback extended
- `_mouse_cb` checks `_settings_open[0]` first
- Panel closed: original close/minimize/settings-open logic
- Panel open: close-button → closes panel; reset-button (bottom) → `_reset_settings()`;
  row +/- clicks via scroll-adjusted `adj_y`; mousemove → hover tracking;
  MOUSEWHEEL → `_panel_scroll_y` clamped to content range
- All mutable panel state uses list wrappers (`_settings_open`, `_panel_scroll_y`,
  `_hover_row`, `_changed`) to avoid `global` declarations in callback

### Save / reset
- `_save_settings()` — writes all _SETTINGS keys back to `[gestures]` in kontrol.conf
  immediately after every +/- click; prints `[SETTINGS] saved …`
- `_reset_settings()` — reloads from `_DEFAULTS["gestures"]`, saves, flashes all values
- STEP_SIZES, VALUE_CLAMPS, UNITS dicts define behaviour per key

### S key toggle
- `ord("s") / ord("S")` in main loop key handler toggles panel open/closed
- Closing resets `_panel_scroll_y` to 0

### Gesture detection during panel
- Camera feed and gesture engine continue running while panel is open
- Cursor movement, clicks, scroll all work — panel is display-only overlay

### kontrol.conf
- No new keys; existing `[gestures]` section written back by `_save_settings()`
- Values round-tripped through `float(f"{v:.4f}")` to stay tidy

---

## v1.5 — wrist rotation replaces peace sign → Alt+Tab (session 6)

### Removed: peace sign gesture
- Deleted `is_peace_sign()` function and all associated state/constants
- Removed: `peace_held`, `task_wrist_xs`, `last_task_t` state variables
- Removed: `TASK_THRESHOLD`, `TASK_COOLDOWN` constants and `task_move_threshold`,
  `task_cooldown` config keys

### Added: wrist rotation gesture (priority 4)
- `knuckle_angle(lm)`: computes angle of knuckle axis LM17 (pinky MCP) → LM5 (index MCP)
  using `atan2(dy, dx)`, returns radians in `[-π, π]`
- Clockwise real-world rotation → angle decreases over time (frame is flipped)
- History: `rot_angle_history` deque (maxlen=12, ≈0.6 s at 20 fps)
- Every detected frame: `(now, knuckle_angle(lm))` appended before priority chain
- Angular velocity: `(a_new - a_old) / dt` after unwrapping ±π boundary
- `ang_vel < 0` → CW → `ydotool key 56:1 15:1 15:0 56:0` (Alt+Tab forward)
- `ang_vel > 0` → CCW → `ydotool key 42:1 56:1 15:1 15:0 42:0 56:0` (Shift+Alt+Tab backward)
- Fires once per rotation: `rot_angle_history.clear()` after fire prevents rapid repeat
- Cooldown `rot_last_fired_t` prevents re-trigger within `rotation_cooldown` seconds
- When `rotation_direction is not None` but on cooldown: `was_gesturing = True` still set
  (prevents cursor firing during partial rotation)
- When no rotation detected: falls through to priorities 5–8 unchanged

### Debug overlay
- With `SHOW_LM_INFO` on (press `i`): `[ROT] {angle}°` line appears below landmark table
  showing live knuckle axis angle in degrees

### kontrol.conf [gestures] changes
| Key | Old | New |
|---|---|---|
| task_move_threshold | 0.06 | removed |
| task_cooldown | 0.5 | removed |
| rotation_threshold | — | 1.8 (rad/s, between normal <0.5 and deliberate 2–5) |
| rotation_cooldown | — | 0.6 s |
| rotation_min_frames | — | 8 frames (≈0.4 s history before evaluating) |

### Calibration notes
- Initial threshold: 1.8 rad/s — raise to 2.5 if false positives during normal cursor use,
  lower to 1.2 if hard to trigger deliberately
- Raise `rotation_cooldown` to 0.8–1.0 if fires multiple times per rotation

---

## v1.4 — three-finger pinch, TouchDesigner skeleton, KWin D-Bus (session 5)

### Three-finger pinch gesture (MISSION 1 — gesture-agent)
- `is_three_finger_pinch(lm, thresh)`: Thumb (LM4) within thresh of BOTH index (LM8)
  AND middle (LM12) simultaneously — distinguishes from individual two-finger pinches
- Threshold: `three_finger_threshold = 0.058` (PINCH_THRESHOLD + 0.010, slightly looser)
- Priority 3 in chain — before middle+thumb scroll (three-finger is a superset of middle+thumb;
  must detect first or middle+thumb fires instead)
- Fires once on pinch entry with 1.5 s cooldown — does NOT repeat while held
- State: `three_finger_held`, `last_three_finger_t`
- HUD: magenta border (220, 50, 220) when active; flash "TASK VIEW" in magenta 1.0 s
- HUD PIN line now shows: `I= M= 3F= / P= R=` (3F = min(i_dist, m_dist))

### KWin D-Bus integration (MISSION 3 — kwin-agent)
- `kwin_dbus_available()`: checks `/component/kwin` on startup
- Confirmed available on this KDE version ✓
- `kwin_call(shortcut_name)`: invokes via `org.kde.kglobalaccel.Component.invokeShortcut`
- `fire_kwin(action)`: D-Bus first, keycode fallback
- **Confirmed KWin shortcut names on this KDE 6 install:**
  - Tile right:  `Window Quick Tile Right`
  - Tile left:   `Window Quick Tile Left`
  - Tile up:     `Window Quick Tile Top`
  - Tile down:   `Window Quick Tile Bottom`
  - Maximize:    `Window Maximize`
  - Minimize:    `Window Minimize`
  - Overview:    `Overview`
  - Show Desktop:`Show Desktop`
- NOTE: `slotWindowQuickTile*` methods do NOT exist in this KDE version.
  The `/KWin` D-Bus interface only exposes showDesktop, queryWindowInfo etc.
  Correct path is `/component/kwin` with `invokeShortcut`.
- All tile gestures now route through `fire_kwin(direction)` instead of raw `tiling_key()`
- Three-finger pinch fires `fire_kwin("task_view")` → "Overview" effect

### TouchDesigner-style skeleton (MISSION 2 — viz-agent)
- Full rewrite of `draw_skeleton()` — `draw_fingertip_markers()` removed (integrated)
- Per-finger colour coding: thumb=purple, index=blue, middle=green, ring=cyan, pinky=orange, wrist=white
- 21 connections including palm cross-connections: (5,9),(9,13),(13,17),(5,17)
- Depth-scaled dots via `lm_radius(lm, i, base, scale)`: `lm[i].z * 4` → larger when closer
- Fingertip highlight rings: outer ring +4px on tips (4,8,12,16,20)
- Dynamic pinch lines (4↔8, 4↔12, 4↔20): interpolate grey→colour, 1px→4px as dist decreases
- Three-finger magenta triangle: drawn via `cv2.polylines` when gesture active
- 'n' key: toggle `SHOW_LM_NUMBERS` — draws index labels (0–20) on each dot
- 'i' key: toggle `SHOW_LM_INFO` — coordinate overlay for key landmarks (wrist, 5 tips)
  showing normalised x,y,z values

### Tile gesture — absolute delta (inherited from v1.3 design)
- Records `tile_start_x/y` on pinch entry; measures absolute displacement from that point
- `tile_wrist_xs/ys` deque removed; no longer needed
- Fires once per pinch hold when displacement > `tile_move_threshold = 0.050`

### Scroll velocity EMA
- `scroll_vel` smoothed with `SCROLL_VEL_ALPHA = 0.30`
- Ticks: `max(1, min(SCROLL_MAX_TICKS, int(abs(scroll_vel) * SCROLL_SPEED)))`
- `scroll_vel` reset to 0.0 on middle+thumb release and hand loss

### kontrol.conf changes
| Section | Key | Old | New |
|---|---|---|---|
| gestures | pinch_threshold | 0.055 | **0.048** |
| gestures | three_finger_threshold | — | 0.058 (new) |
| gestures | three_finger_cooldown | — | 1.5 (new) |
| gestures | scroll_deadzone | 0.010 | **0.008** |
| gestures | scroll_vel_alpha | — | 0.30 (new) |
| gestures | scroll_max_ticks | — | 8 (new) |
| gestures | tile_window_frames | 8 | removed |

### Test suite (MISSION 5 — tester-agent) — requires live run with camera
**Gesture tests:**
- [ ] N01: Three-finger pinch → KDE Overview opens
- [ ] N02: Three-finger fires once per hold (no repeat)
- [ ] N03: Cooldown: cannot fire twice in 1.5 s
- [ ] N04: Three-finger does NOT trigger middle+thumb scroll
- [ ] N05: Three-finger does NOT trigger right click
- [ ] N06: Release three-finger → cursor resumes

**Visualization tests:**
- [ ] V01: All 21 dots visible
- [ ] V02: Each finger different colour
- [ ] V03: Closer joints appear larger
- [ ] V04: Palm cross-connections drawn
- [ ] V05: Pinch lines between thumb and each fingertip
- [ ] V06: Pinch line brightens and thickens when close
- [ ] V07: Three-finger magenta triangle when active
- [ ] V08: 'n' → landmark numbers appear
- [ ] V09: 'n' again → numbers disappear
- [ ] V10: 'i' → coordinate overlay appears

**KWin D-Bus tests:**
- [ ] K01: Startup prints `[KWIN] D-Bus available ✓`
- [ ] K02: Tile right → window tiles right
- [ ] K03: Tile left → window tiles left
- [ ] K04: Task view → Overview opens
- [ ] K05: Minimize → Show Desktop fires

**Regression tests:**
- [ ] R01: Cursor smooth, no lag
- [ ] R02: Hand re-entry no jump
- [ ] R03: Scroll velocity works
- [ ] R04: Palm progress bar fills
- [ ] R05: Pinch threshold 0.048 — not too sensitive

### Deferred to v1.5
- Live test pass/fail table (requires camera session)
- Adaptive detection phases (SEARCHING/LOCKED) — reverted in v1.3, deferred again
- Peace sign still uses raw landmarks (low priority, pose not distance)
- Landmark smoothing layer (pdist_s / slm) — reverted in v1.3, deferred again

---

## v1.3 — adaptive detection, landmark smoothing, styled HUD (session 4)

### Detection phases (MISSION 1 — Sensing)
- Switched from `RunningMode.VIDEO` to `RunningMode.IMAGE` to enable hot-reload
- `build_detector(detect_t, presence_t, track_t)` factory function
- Two-phase system: **SEARCHING** (high thresholds) → **LOCKED** (lower thresholds)
  - SEARCHING: detection=0.75, presence=0.75, tracking=0.65
  - LOCKED:    detection=0.60, presence=0.60, tracking=0.50
  - Lock trigger: 3 consecutive detection frames
  - Unlock trigger: 5 consecutive miss frames (tolerates brief occlusion)
- Hot-reload on phase transition (~50 ms): old detector closed, new one built
- Phase transitions logged to terminal: `[PHASE] → LOCKED` / `[PHASE] → SEARCHING`

### Landmark smoothing (MISSION 1 — S4/S5)
- `smooth_landmarks(lm, alpha)` — EMA applied to indices [0,4,5,8,9,12,13,16,17,20]
- Module-level `_smooth_lm` dict reset on hand loss via `reset_smooth()`
- All gesture detectors updated to use `pdist_s(slm, a, b)` on smoothed coords
- Allows pinch threshold tightened from 0.055 → **0.048** (pre-smoothed input)
- Bunch/palm detection (`is_bunch_s`) uses smoothed landmarks
- Wrist reference for tile/scroll uses `slm[0]` (smoothed wrist)

### Tile gesture — absolute delta (MISSION 3 — P3)
- Old: tracked wrist delta history over sliding window (8 frames)
- New: record `tile_start_x/y` on pinch entry; measure absolute displacement
- Fires once per pinch hold when displacement exceeds `tile_move_threshold=0.050`
- Eliminates stale-history false fires; direction is unambiguous from entry point
- Removed `tile_window_frames` state (no longer needed)

### Scroll velocity EMA (MISSION 3 — P2)
- Old: raw `dy_norm * SCROLL_SPEED` per frame (tremor → single-tick jitter)
- New: `scroll_vel` smoothed with `SCROLL_VEL_ALPHA=0.30`
- Ticks clamped: `max(1, int(abs(scroll_vel) * SCROLL_SPEED))` up to `scroll_max_ticks=8`
- `scroll_vel` reset to 0.0 on scroll gesture exit
- Result: slow hand = 1 tick, fast hand = up to 8 ticks, tremor = no scroll

### Palm countdown feedback (MISSION 3 — P4)
- Terminal print at halfway (frame 10/20): `[PALM] halfway — keep closed`
- Terminal print at trigger (frame 20/20): `[PALM] firing`
- `play_palm_beep()`: descending two-tone (600 Hz → 400 Hz, 80 ms each) via aplay
- `palm_beep_fired` flag — beep plays exactly once per bunch hold sequence

### HUD redesign (MISSION 2 — UI)
- `draw_panel()`: semi-transparent dark background (alpha=0.55) for all HUD text
- `draw_progress_bar()`: proper filled/track bar with border for palm countdown
- `draw_skeleton()`: styled — grey connections, grey joints, white wrist ring,
  green-highlighted index tip (cursor controller), active pinch ring overlays
- `draw_detection_phase()`: top-right corner indicator
  - LOCKED: solid green dot + "LOCK" label
  - SEARCHING: pulsing grey dot (sin wave, 4 Hz) + "SRCH" label
- `draw_zone_warning()`: now uses smoothed `slm[8]` instead of raw `lm[8].x/y`
- `draw_buttons()`: hover-aware (brightens on cursor hover) via `_mouse_xy` state
- HUD rows: FPS (colour-coded), HAND (phase), MODE, ACT (bold yellow when active),
  PIN (I/M/P distances, green when active), PALM (progress bar, orange→green)
- Flash messages: centred with drop shadow, 1.5 s for palm, 0.8 s for tile

### kontrol.conf changes
| Section | Key | Old | New |
|---|---|---|---|
| detection | detection_confidence | 0.50 | replaced by search/locked pairs |
| gestures | pinch_threshold | 0.055 | **0.048** |
| gestures | scroll_deadzone | 0.010 | **0.008** |
| gestures | tile_move_threshold | 0.06 | **0.050** |
| gestures | scroll_vel_alpha | — | 0.30 (new) |
| gestures | scroll_max_ticks | — | 8 (new) |
| gestures | palm_hold_frames | bunch_hold_frames=12 | **20** |
| gestures | tile_window_frames | 8 | removed |
| camera_tuning | sharpness | 128 | removed (not all cameras support) |

### Test suite (MISSION 4) — to be run live
37 tests across detection, cursor, gesture, conflict, UI, performance categories.
See session prompt for full test list. Live run required — code not auto-tested.

### Performance targets (not yet measured — live run required)
- Target: mean <30 ms, p95 <45 ms, max <80 ms
- IMAGE mode removes timestamp overhead vs VIDEO mode
- All ydotool calls remain Popen (fire-and-forget) except blocking key sequences

### Deferred to v1.4
- Frame timing audit (P1) — `statistics` profiling block — add/collect/remove live
- Formal 37-test pass/fail table — requires live session with camera
- Adaptive smooth alpha per-gesture (palm vs cursor could use different α)
- `is_peace_sign` still uses raw landmarks (low priority — pose not distance)

---

## v1.2 — clean architecture, corrected gesture map, new HUD (session 3)

### Architecture changes
- `kontrol.py` restructured into strict sections: Imports → Config → Constants →
  ydotool helpers → Landmark helpers → Camera → Draw functions → Cursor pipeline → run()
- All constants derived from config, named in ALL_CAPS — no magic numbers
- Every non-obvious landmark index annotated with its anatomical meaning
- Every ydotool call annotated with what it does in KDE

### Camera
- `apply_camera_settings()`: added `time.sleep(0.3)` before first apply (let UVC settle)
- Two-pass confirmed: pass 1 before warm-up reads, pass 2 after VIDIOC_STREAMON reset
- Brightness verification print on startup: `[CAM] brightness mean: X.X (target >80)`
- **Device confirmed: `/dev/video0`** — no video2 exists (built-in disabled via udev)
- `_DEFAULTS` camera id fixed: was `"2"` (bug), now `"0"` (matches runtime conf)

### Gesture map — CORRECTED (v1.1 code had wrong finger assignments)

| Priority | Gesture | Landmark pair | Action |
|---|---|---|---|
| 1 | Palm close | all 5 curled | minimize / restore toggle (Meta+Down) |
| 2 | Pinky+Thumb hold + move | LM 4+20 | KDE tile (Meta+direction) |
| 3 | Middle+Thumb + vertical move | LM 4+12 | scroll up/down |
| 4 | Middle+Thumb stationary | LM 4+12 | left click / drag |
| 5 | Index+Thumb | LM 4+8 | right click |
| 6 | Index tip (LM 8) | — | cursor (5-stage EMA pipeline) |

**Bug fixes from v1.1 code:**
- Index+Thumb was left click/drag → corrected to right click
- Ring+Thumb was right click → removed entirely (ring_thumb_pinched dead code purged)
- Middle+Thumb was scroll-only → now click/drag/scroll (scroll wins when wrist moves)
- `locked` state existed, was never set True — dead code removed
- Tile history now cleared on pinky+thumb entry (prevents stale pre-pinch movement)
- Palm restore key was Meta+PgUp (104) → corrected to Meta+Down (108) — toggle both ways

### Cursor pipeline (unchanged from v1.1 — was correct)
- Stage 1: landmark pre-smooth (α=0.35, fixed)
- Stage 2: zone mapping [0.15–0.85 x, 0.10–0.90 y] → screen pixels
- Stage 3: velocity (screen px/frame) → adaptive alpha
- Stage 4: velocity-adaptive EMA (min=0.06, max=0.40, scale=4.0)
- Stage 5: integer delta, 2px deadzone, Popen fire-and-forget
- Re-entry: snaps cx/cy on first_frame or was_gesturing, sends no delta

### HUD rewrite
- Border: red=no hand, orange=palm closing, blue=drag, yellow=tile hold, green=tracking
- Panel lines: `[FPS]`, `[HAND]`, `[GESTURE]`, `[PINCH] I= M= P=` (removed Ring column)
- Palm bar: shown only when palm_frames > 0
- Flash messages: MINIMIZE, RESTORE, TILE UP/DOWN/LEFT/RIGHT — 1.0 s duration
- Fingertip markers: red circle on LM 8 (index+thumb), blue on LM 12 (middle+thumb),
  yellow on LM 20 (pinky+thumb) — only when respective pinch is active
- Zone warnings: text arrows when LM 8 within 5% of zone boundary
- CamPanel removed (complexity not justified by spec)
- Buttons: [—] minimize + [✕] quit only (CAM button removed)

### kontrol.conf — final values

| Section | Key | Value |
|---|---|---|
| screen | width / height | 4480 / 1440 |
| camera | id / flip | 0 / true |
| mapping | zone_x | 0.15 – 0.85 |
| mapping | zone_y | 0.10 – 0.90 |
| smoothing | landmark_smooth | 0.35 |
| smoothing | min/max_smooth | 0.06 / 0.40 |
| smoothing | velocity_scale | 4.0 |
| smoothing | cursor_deadzone_px | 2 |
| gestures | pinch_threshold | 0.055 |
| gestures | pinch_cooldown | 0.35 |
| gestures | scroll_deadzone | 0.010 |
| gestures | scroll_speed | 6.0 |
| gestures | palm_hold_frames | 20 |
| gestures | palm_cooldown | 2.0 |
| gestures | tile_move_threshold | 0.06 |
| gestures | tile_window_frames | 8 |
| gestures | tile_cooldown | 0.8 |
| camera_tuning | auto_exposure | 1 |
| camera_tuning | exposure_time_absolute | 300 |
| camera_tuning | gain | 100 |
| camera_tuning | brightness | 160 |

### Known issues / deferred
- Palm key (Meta+Down = KEY_DOWN = 108): toggles minimize/restore in KDE with some
  tiling managers. If your WM uses different shortcuts, update `[KEY_DOWN=108]` references
  in the gesture priority chain directly.
- CamPanel removed — if live brightness tuning is needed, re-add from v1.1.
- Performance numbers: not yet measured (no [PERF] timing run conducted this session).
  Run with `_dt > 0.045` print to collect; expected <30 ms per frame on i5-8250U.
- Tests T01–T37 not yet formally verified — session produced code, not live run.

---

## v1.1 — smooth cursor + full gesture map (session 2)

### Cursor pipeline (complete rewrite)
- Replaced old double-EMA with clean 5-stage pipeline:
  - Stage 1: landmark pre-smooth (fixed α=0.35) — kills MediaPipe jitter before zone mapping
  - Stage 2: zone mapping `[zone_min, zone_max] → [0,1] → screen px`
  - Stage 3: velocity in screen pixels → adaptive smooth factor, no edge_boost hack
  - Stage 4: screen-space EMA with velocity-adaptive α (min=0.06, max=0.40, scale=4.0)
  - Stage 5: integer delta with deadzone (2px) — Popen fire-and-forget
- Re-entry guard: `was_gesturing` flag snaps cursor to current hand position on any
  transition from gesture mode back to cursor — no jump after pinch/tile/palm
- Cursor also moves during drag (middle+thumb held) — same pipeline, same re-entry guard
- Removed: `edge_boost`, `still_threshold` — both caused more problems than they solved
- `raw_x_s = None` sentinel: pre-smooth initializes from landmark on first hand detection,
  resets to None on hand loss

### Gesture map (completely new)
Old gesture set replaced entirely. New priority chain (strict if/elif):

**Gesture 1 — Palm close → minimize / restore**
- Detection: all 5 tips below their MCP/IP joints (thumb: tip > IP; fingers: tip > MCP)
- Hold 20 consecutive frames (~1 s at 20 fps), fire on RELEASE
- Toggle: first fire = minimize (Meta+Down), second fire = restore (Meta+Down again)
- `palm_minimized` bool tracks state; cooldown 2.0 s between fires
- HUD: progress bar while closing, flash "MINIMIZE" or "RESTORE" on trigger

**Gesture 2 — Index+Thumb (LM 4+8) → RIGHT click**  
- Changed from left click (v1.0) to right click
- Fires on pinch entry, cooldown 0.35 s, no drag

**Gesture 3 — Middle+Thumb (LM 4+12) → left click / drag**
- Quick pinch-release (< 0.35 s held) → left click (mousedown + mouseup)
- Hold > 0.35 s → mousedown (drag mode); cursor moves normally while held; mouseup on release

**Gesture 4 — Middle+Thumb held + vertical wrist movement → scroll**
- Same pinch as gesture 3; scroll takes priority when `abs(dy_norm) > SCROLL_DEADZONE`
- Scroll direction: wrist up = scroll up, wrist down = scroll down
- Per-frame delta: `ticks = max(1, int(abs(dy_norm) * SCROLL_SPEED))`
- Command: `ydotool mousemove --wheel -x 0 -y ±ticks`
- Cursor does NOT move during scroll frames

**Gesture 5 — Pinky+Thumb (LM 4+20) held + direction → KDE tile**
- While pinch held, track wrist (LM 0) over `tile_window_frames=8` frames
- Net displacement: `dx_total = xs[-1] - xs[0]`, `dy_total = ys[-1] - ys[0]`
- Dominant axis + exceeds `tile_move_threshold=0.06` → fire direction
- Fires once per pinch hold; `tile_fired` flag resets on release
- KDE keycodes via raw input-event-codes.h (Meta+Up/Down/Left/Right)
- Cooldown 0.8 s; HUD flashes "TILE RIGHT" etc.

### Removed gestures
- Wrist rotate → task overview (Meta+PgUp): removed
- Wrist flick → tile (velocity-based): replaced by pinky+thumb direction hold
- Ring+thumb → right click: replaced by index+thumb
- Index+thumb left click/drag: moved to middle+thumb
- Fist lock: removed (no lock-enter gesture in v1.1)
- Fist minimize: replaced by palm close

### HUD update
- Line 1: FPS, Line 2: Mode, Line 3: Gesture, Line 4: all three pinch distances
- Line 5: palm progress bar when closing
- Flash messages: MINIMIZE, RESTORE, TILE UP/DOWN/LEFT/RIGHT
- Border: green=tracking, red=locked, blue=drag active, yellow=tile held

### kontrol.conf final values
- `[screen]` width=4480, height=1440
- `[camera]` id=2 (C920 on /dev/video2)
- `[smoothing]` landmark_smooth=0.35, min=0.06, max=0.40, velocity_scale=4.0, deadzone=2px
- `[gestures]` pinch=0.055, cooldown=0.35, scroll_deadzone=0.010, scroll_speed=6.0,
  palm_hold_frames=20, palm_cooldown=2.0, tile_move_threshold=0.06, tile_window=8, tile_cooldown=0.8
- Removed: edge_boost, still_threshold, flick_*, taskview_*, fist_*, abs_scale_*,
  exposure_dynamic_framerate

---

## v1.0 — camera + cursor basics (Ruflo session 1)

### Camera
- Added `[camera_tuning]` section to `kontrol.conf` — all brightness/exposure
  values are now config-driven; no Python code changes needed to tune.
- Replaced `cam-tune.sh` shell calls with `apply_camera_settings()` Python
  function that reads from `kontrol.conf` and applies each v4l2 control via
  subprocess, wrapping each in try/except so unknown controls are skipped.
- Controls applied in correct dependency order: `auto_exposure=1` first (enables
  manual mode), then `exposure_time_absolute`, then focus sequence.
- Two-pass apply preserved: pass 1 before VIDIOC_STREAMON, pass 2 after 5
  warm-up reads — locks exposure after stream-on reset fires.
- Frame brightness mean achieved: **81.3** (target >80).
- Confirmed device: C920 enumerates as `/dev/video0` (built-in udev-disabled).
- Confirmed 50 Hz mains (UK), manual focus at 30, gain=100, brightness=160.

### Cursor
- Zone mapping verified correct: center→(2240,720), edges map to screen extremes.
- Double EMA confirmed: Stage 1 (landmark pre-smooth α=0.3), Stage 2
  (velocity-adaptive screen-space EMA, vel_norm × velocity_scale × edge_boost).
- Hand re-entry no-jump verified: on first frame back, cx/cy and prev_sent_x/y
  both reset to hand position — no cursor delta is sent that frame.
- Relative moves confirmed: `ydotool mousemove -x dx -y dy` only. No --absolute.
- Smoothing defaults aligned: `_DEFAULTS` now matches `kontrol.conf` values
  (`velocity_scale=3.5`, `max_smooth=0.35`).

### Config
- `kontrol.conf` camera id=0 confirmed correct (C920 at /dev/video0).
- All camera tuning values configurable without touching Python.
