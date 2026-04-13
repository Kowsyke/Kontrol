# Kontrol CHANGELOG v2

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
