#!/usr/bin/env bash
# LARIC System - Shutdown Script
# Copyright (c) 2026 LARIC. All rights reserved.
#
# Simulation shutdown: kills the ROS 2 / Python / Gazebo processes.

echo "Shutting down LARIC systems..."

pkill -9 -f "agent_logic.py" || true
pkill -9 -f "laric_interface.py" || true
pkill -9 -f "ros2 launch" || true
pkill -9 -f "component_container" || true
pkill -9 -f "nav2" || true
pkill -9 -f "robot_state_publisher" || true
killall -9 gzserver gzclient rviz2 2>/dev/null || true
ros2 daemon stop || true

echo "Cleanup complete. System offline."