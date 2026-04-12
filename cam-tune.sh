#!/usr/bin/env bash
# C920 v4l2 tuning for close-range hand tracking (~60 cm desk distance)
# All controls verified against: v4l2-ctl -d /dev/video2 --list-ctrls
set -e

DEV=/dev/video2

echo "[cam-tune] Tuning $DEV for hand tracking..."

# 50 Hz mains (UK) — without this the camera anti-flicker fights the wrong
# frequency and you get periodic brightness pulsing under artificial light.
# value 1 = 50 Hz, value 2 = 60 Hz (US default)
v4l2-ctl -d "$DEV" --set-ctrl=power_line_frequency=1

# Manual exposure (value 1 = Manual Mode; 3 = Aperture Priority auto)
v4l2-ctl -d "$DEV" --set-ctrl=auto_exposure=1

# Exposure time 250 — brighter than the previous 150, still holds 30 fps
# (range 3–2047; must set auto_exposure=1 first or this is inactive)
v4l2-ctl -d "$DEV" --set-ctrl=exposure_time_absolute=250

# Disable dynamic framerate (interferes with manual exposure)
v4l2-ctl -d "$DEV" --set-ctrl=exposure_dynamic_framerate=0

# Backlight compensation — evens out brightness when there's a bright
# background (window, lamp behind the hand)
v4l2-ctl -d "$DEV" --set-ctrl=backlight_compensation=1

# Gain 80 — slightly higher than before for better sensitivity at manual exposure
# (range 0–255)
v4l2-ctl -d "$DEV" --set-ctrl=gain=80

# Disable continuous autofocus, fix focus for ~60 cm desk distance
# (range 0–250 step 5)
v4l2-ctl -d "$DEV" --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d "$DEV" --set-ctrl=focus_absolute=30

echo "[cam-tune] Done."
