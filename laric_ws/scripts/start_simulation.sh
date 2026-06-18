#!/usr/bin/env bash
# LARIC System - Startup Script
# Copyright (c) 2026 LARIC. All rights reserved.
#
# Simulation startup: launches Gazebo, Nav2 + RViz, the HMI and the agent.

set -e

WS_PATH="$HOME/laric_ws"
VENV_PATH="$WS_PATH/laric_env/bin/activate"
LOG_DIR="$WS_PATH/laric_logs/$(date +'%Y%m%d_%H%M%S')"
MAP_PATH="${MAP_PATH:-$WS_PATH/maps/tb3_house_map.yaml}"
mkdir -p "$LOG_DIR"

echo "--- Starting LARIC Simulation Environment ---"

# The 'sleep's let each stage register its ROS 2 topics before the next starts.
echo "[1/4] Starting Gazebo (World)..."
bash -c "source $VENV_PATH && ros2 launch turtlebot3_gazebo \
    turtlebot3_house.launch.py x_pose:=-2.0 y_pose:=-3.0 \
    > $LOG_DIR/gazebo.log 2>&1" &
sleep 10

echo "[2/4] Starting Nav2 and RViz..."
bash -c "source $VENV_PATH && ros2 launch turtlebot3_navigation2 \
    navigation2.launch.py use_sim_time:=true \
    map:=$MAP_PATH \
    params_file:=$WS_PATH/src/laric_core/config/laric_nav2_params.yaml \
    > $LOG_DIR/nav2.log 2>&1" &
sleep 5

# HMI before the agent, so its /language_changed publisher exists first.
echo "[3/4] Starting Human-Machine Interface..."
bash -c "source $VENV_PATH && python3 \
    $WS_PATH/src/laric_core/laric_core/laric_interface.py > \
    $LOG_DIR/hmi.log 2>&1" &
sleep 3

echo "[4/4] Starting RAI Agent Logic..."
bash -c "source $VENV_PATH && python3 \
    $WS_PATH/src/laric_core/laric_core/agent_logic.py > \
    $LOG_DIR/agent.log 2>&1" &

echo "--- All systems are online ---"