#!/usr/bin/env bash

# LARIC System - Startup Script
# Copyright (c) 2026 LARIC. All rights reserved.
#
# Orchestrates the launch of Gazebo, Nav2, the RAI Agent, and the HMI.


# Exit immediately if a command exits with a non-zero status
set -e

# Configuration variables
WS_PATH="$HOME/laric_ws"
VENV_PATH="$WS_PATH/laric_env/bin/activate"
LOG_DIR="$WS_PATH/laric_logs/$(date +'%Y%m%d_%H%M%S')"

mkdir -p "$LOG_DIR"

echo "--- Starting LARIC Simulation Environment ---"

# 1. Launch Gazebo (Physical World)
echo "[1/4] Starting Gazebo (World)..."
bash -c "source $VENV_PATH && ros2 launch turtlebot3_gazebo \
    turtlebot3_house.launch.py > $LOG_DIR/gazebo.log 2>&1" &

sleep 10

# 2. Launch Navigation Stack (Mapping & Localization)
# Note: The tb3_house_map (.yaml and .pgm) was obtained from the Anis Koubaa's 
# open-source repository: https://github.com/aniskoubaa/ros_course_part2
echo "[2/4] Starting Nav2 and RViz..."
bash -c "source $VENV_PATH && ros2 launch turtlebot3_navigation2 \
    navigation2.launch.py use_sim_time:=true \
    map:=$WS_PATH/maps/tb3_house_map.yaml > $LOG_DIR/nav2.log 2>&1" &

sleep 5

# 3. Launch HMI (The Voice/Text Interface)
echo "[3/4] Starting Human-Machine Interface..."
bash -c "source $VENV_PATH && python3 \
    $WS_PATH/src/laric_core/laric_core/laric_interface.py > \
    $LOG_DIR/hmi.log 2>&1" &

sleep 3

# 4. Launch RAI Agent (The Brain)
echo "[4/4] Starting RAI Agent Logic..."
gnome-terminal --title="LARIC_Agent_Log" -- bash -c "source $VENV_PATH && \
    python3 $WS_PATH/src/laric_core/laric_core/agent_logic.py; exec \
    bash"

echo "--- All systems are online ---"
