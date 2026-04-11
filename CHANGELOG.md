# Kontrol — Change Log

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
| ydotool absolute coords | `ydotool mousemove -a -x X -y Y` — pixel-direct, `ABS_SCALE=1.0` |

### Current tuning knobs (in `kontrol.py`)

```python
SCREEN_W        = 1920      # primary monitor width
SCREEN_H        = 1080      # primary monitor height
SMOOTH          = 0.2       # EMA: lower=smoother/laggier, higher=snappier
PINCH_THRESHOLD = 0.05      # normalized distance; smaller=tighter pinch required
PINCH_COOLDOWN  = 0.4       # seconds between clicks
ABS_SCALE_X     = 1.0       # set to 32767/1920 if cursor jumps to corner
ABS_SCALE_Y     = 1.0
```

### Gesture map
| Gesture | Action |
|---------|--------|
| Index fingertip (landmark 8) position | Cursor moves |
| Pinch index ↔ thumb (landmarks 8 ↔ 4, dist < PINCH_THRESHOLD) | Left click |

### Launch
```bash
cd /home/K/Storage/Projects/Kontrol
./run.sh
# or manually:
DISPLAY=:0 QT_QPA_PLATFORM=xcb YDOTOOL_SOCKET=/run/user/1000/.ydotool_socket \
  source venv/bin/activate && python kontrol.py
```

### Known warnings (harmless)
- `inference_feedback_manager.cc: Feedback manager requires model with single signature` — MediaPipe internal, ignored
- `landmark_projection_calculator.cc: Using NORM_RECT without IMAGE_DIMENSIONS` — cosmetic, tracking still works
- `QFontDatabase: Cannot find font directory` — cv2 Qt font path issue, display works fine

### Next tuning targets (Session 2)
- Adjust `SMOOTH` if tracking feels laggy or jittery
- Adjust `PINCH_THRESHOLD` if clicks are too easy or too hard to trigger
- Check `ABS_SCALE_X/Y` if cursor maps to wrong screen in dual-monitor setup
- Consider adding right-click (ring+thumb pinch) and scroll (two-finger drag)
