#!/usr/bin/env bash
# Kontrol v0.1 — launcher
# Run from any directory.

set -e
cd "$(dirname "$0")"

# XWayland display (cv2 bundled Qt has no Wayland plugin)
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb

# ydotool daemon socket (user K, NOT root)
export YDOTOOL_SOCKET=/run/user/1000/.ydotool_socket

# Ensure ydotoold is up
if ! systemctl --user is-active --quiet ydotoold.service; then
    echo "Starting ydotoold..."
    systemctl --user start ydotoold.service
fi

source venv/bin/activate

# Apply C920 v4l2 tuning on every launch
bash cam-tune.sh

exec python kontrol.py "$@"
