#!/usr/bin/env bash

# LARIC System - Real Robot Shutdown Script
# Copyright (c) 2026 LARIC. All rights reserved.
#
# Safely terminates the LAPTOP-side Python processes (HMI and agent). The
# robot's Nav2/SLAM/perception stack must be stopped from the robot itself
# ('just stop' inside ~/rosbot-autonomy on the robot's onboard computer).
#
# Note: 'set -e' intentionally omitted — pkill returns non-zero when no
# process matches, and that should not abort the rest of the cleanup.
# (Same convention as stop_simulation.sh.)

echo "Shutting down LARIC laptop systems..."

# Terminate Python scripts
pkill -9 -f "agent_logic.py" || true
pkill -9 -f "laric_interface.py" || true

# Stop the ROS 2 daemon (matches stop_simulation.sh)
ros2 daemon stop || true

echo "Cleanup complete. Laptop systems offline."
echo "NOTE: the ROSbot's navigation stack is still running. To stop it, run"
echo "'just stop' inside ~/rosbot-autonomy on the robot."