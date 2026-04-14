# Kontrol CHANGELOG v2

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
