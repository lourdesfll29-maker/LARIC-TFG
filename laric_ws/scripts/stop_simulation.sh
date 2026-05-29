#!/usr/bin/env bash

# LARIC System - Shutdown Script
# Copyright (c) 2026 LARIC. All rights reserved.
#
# Safely terminates all ROS 2 nodes, Python agents, and Gazebo processes.

echo "Shutting down LARIC systems..."

# Terminate ROS 2 launches and Python scripts
# Using -f with specific patterns to avoid killing unrelated processes
pkill -9 -f "agent_logic.py" || true
pkill -9 -f "laric_interface.py" || true

pkill -9 -f "ros2 launch" || true

pkill -9 -f "component_container" || true
pkill -9 -f "nav2" || true
pkill -9 -f "robot_state_publisher" || true

# Force kill simulators and visualizers
killall -9 gzserver gzclient rviz2 2>/dev/null || true

ros2 daemon stop || true

echo "Cleanup complete. System offline."

