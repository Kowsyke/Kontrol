#!/usr/bin/env bash
# C920 v4l2 tuning for close-range hand tracking (~60 cm desk distance)
# All controls verified against: v4l2-ctl -d /dev/video2 --list-ctrls
set -e

DEV=/dev/video2

echo "[cam-tune] Tuning $DEV for hand tracking..."

# Manual exposure (value 1 = Manual Mode; 3 = Aperture Priority auto)
v4l2-ctl -d "$DEV" --set-ctrl=auto_exposure=1

# Exposure time — 150 is good for indoor desk lighting
# (range 3–2047; must set auto_exposure=1 first or this is ignored)
v4l2-ctl -d "$DEV" --set-ctrl=exposure_time_absolute=150

# Disable dynamic framerate adjustment (interferes with manual exposure)
v4l2-ctl -d "$DEV" --set-ctrl=exposure_dynamic_framerate=0

# Disable continuous autofocus, then set fixed focus for ~60 cm
# (range 0–250 step 5; 30 = ~60 cm working distance)
v4l2-ctl -d "$DEV" --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d "$DEV" --set-ctrl=focus_absolute=30

# Gain — 64 balances noise vs brightness at manual exposure 150
# (range 0–255)
v4l2-ctl -d "$DEV" --set-ctrl=gain=64

echo "[cam-tune] Done."
