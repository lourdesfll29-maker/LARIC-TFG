#!/usr/bin/env bash
# LARIC System - Real Robot Shutdown Script
# Copyright (c) 2026 LARIC. All rights reserved.
#
# Laptop side: kills only the HMI and agent. Stop the robot's own stack with
# 'just stop' on the ROSbot.

echo "Shutting down LARIC laptop systems..."

pkill -9 -f "agent_logic.py" || true
pkill -9 -f "laric_interface.py" || true
ros2 daemon stop || true

echo "Cleanup complete. Laptop systems offline."
echo "NOTE: the ROSbot's navigation stack is still running. To stop it, run"
echo "'just stop' inside ~/rosbot-autonomy on the robot."