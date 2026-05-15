# Kontrol — Claude Code Project Brief

## System
- ASUS X510UAR, Fedora 44, KDE Plasma 6.6.4, Wayland
- i5-8250U, 16GB RAM, Intel UHD 620 (Mesa 26.0.6), NO GPU
- Dual monitor: Built-in 1920x1080 (left) + QHZ 2560x1440 (right)
- Total desktop: 4480x1440

## Python
- Version: 3.14.4
- venv: /home/K/Storage/Projects/Kontrol/venv/
- mediapipe: 0.10.33
- opencv-python: 4.13.0.92
- numpy: 2.4.4
- flask: 3.1.3

## Camera
- Logitech C920 on /dev/video0
- Built-in IMC cam PERMANENTLY disabled (udev rule: /etc/udev/rules.d/99-disable-builtin-cam.rules)
- v4l2 settings applied via apply_camera_settings() after VideoCapture opens
- Startup brightness mean: ~170 (target >80, confirmed OK on Fedora 44)

## Input injection
- ydotoold user service, socket: /run/user/1000/.ydotool_socket
- ydotool: installed at /usr/bin/ydotool (Fedora 44 package)
- ONLY ydotool mousemove -x dx -y dy (RELATIVE) — NO --absolute (silently dropped)
- Scroll: ydotool mousemove --wheel -x 0 -y N (still works on Fedora 44)
- No pynput, no evdev, no xdotool

## Stack
- Python 3.14.4, venv at venv/
- MediaPipe Tasks API ONLY (mp.tasks.vision.HandLandmarker)
- NOT mp.solutions — removed in 0.10+
- Single hand only — num_hands=1, always lm[0]

## Launch
- alias: kontrol / kontroloff / kontrolh / kontrolstatus / kontrollog
- direct: cd /home/K/Storage/Projects/Kontrol && ./run.sh
- DISPLAY=:0 QT_QPA_PLATFORM=xcb YDOTOOL_SOCKET=/run/user/1000/.ydotool_socket

## Current state
- v1.8.1 — wrist rotation priority fix, autostart disabled
- CHANGELOG_V2.md tracks progress from this point
- git remote: git@github.com:Kowsyke/Kontrol.git (main)
- SSH key: ~/.ssh/github_ed25519
- Fedora 44 migrated: 2026-05-15

## REST API (v1.8)
- Flask daemon on 127.0.0.1:5555 (api_enabled = true in kontrol.conf)
- GET  /status          — full state snapshot (fps, hand_detected, active_gesture, cursor, uptime)
- GET  /gestures        — 10-gesture list with priorities/thresholds
- POST /profile         — {"name": "precise"}
- POST /setting         — {"key": "pinch_threshold", "value": 0.05}
- POST /headless        — toggle headless mode
- POST /stop            — graceful shutdown
- GET  /log             — last 50 lines of kontrol.log
- GET  /app-profiles    — current app→profile map
- POST /app-profile     — {"app": "firefox", "profile": "default"}
- GET  /diagnostic      — gesture + cursor pipeline values

## Services
- hdd-storage.service: REMOVED in Fedora 44 — storage now mounts via /etc/fstab at boot
- ydotoold.service: user service, active, socket at /run/user/1000/.ydotool_socket
- kontrol.service: DISABLED (intentional per v1.8.1) — start manually via `kontrol` alias

## Display environment (Fedora 44 confirmed)
- XDG_SESSION_TYPE=wayland
- WAYLAND_DISPLAY=wayland-0
- DISPLAY=:0

## Rules (NEVER VIOLATE)
- Never use --absolute with ydotool
- Never use mp.solutions
- Never use pynput
- Always call apply_camera_settings() after VideoCapture opens
- Always commit and push after each working feature
- Always update CHANGELOG_V2.md before committing
