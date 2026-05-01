# Kontrol — Claude Code Project Brief

## System
- ASUS X510UAR, Fedora 43, KDE Plasma 6, Wayland
- i5-8250U, 16GB RAM, Intel UHD 620, NO GPU
- Dual monitor: Built-in 1920x1080 (left) + QHZ 2560x1440 (right)
- Total desktop: 4480x1440

## Camera
- Logitech C920 on /dev/video0
- Built-in IMC cam PERMANENTLY disabled (udev rule)
- v4l2 settings applied via apply_camera_settings() after VideoCapture opens

## Input injection
- ydotoold user service, socket: /run/user/1000/.ydotool_socket
- ONLY ydotool mousemove -x dx -y dy (RELATIVE) — NO --absolute (silently dropped)
- No pynput, no evdev, no xdotool

## Stack
- Python 3.14, venv at venv/
- MediaPipe Tasks API ONLY (mp.tasks.vision.HandLandmarker)
- NOT mp.solutions — removed in 0.10+
- Single hand only — num_hands=1, always lm[0]

## Launch
- alias: kontrol / kontroloff
- direct: cd /home/K/Storage/Projects/Kontrol && ./run.sh
- DISPLAY=:0 QT_QPA_PLATFORM=xcb YDOTOOL_SOCKET=/run/user/1000/.ydotool_socket

## Current state
- v1.8 — Flask REST API, auto profile switching, zoom gesture, diagnostic mode
- CHANGELOG_V2.md tracks progress from this point
- git remote: git@github.com:Kowsyke/Kontrol.git (main)
- SSH key: ~/.ssh/github_ed25519

## REST API (v1.8)
- Flask daemon on 127.0.0.1:5555 (api_enabled = true in kontrol.conf)
- GET  /status          — full state snapshot
- GET  /gestures        — 10-gesture list with priorities/thresholds
- POST /profile         — {"name": "precise"}
- POST /setting         — {"key": "pinch_threshold", "value": 0.05}
- POST /headless        — toggle headless mode
- POST /stop            — graceful shutdown
- GET  /log             — last 50 lines of kontrol.log
- GET  /app-profiles    — current app→profile map
- POST /app-profile     — {"app": "firefox", "profile": "default"}
- GET  /diagnostic      — gesture + cursor pipeline values

## Rules
- Never use --absolute with ydotool
- Never use mp.solutions
- Never use pynput
- Always call apply_camera_settings() after VideoCapture opens
- Always commit and push after each working feature
- Always update CHANGELOG_V2.md before committing
