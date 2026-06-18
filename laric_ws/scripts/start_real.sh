#!/usr/bin/env bash
# LARIC System - Real Robot Startup Script
# Copyright (c) 2026 LARIC. All rights reserved.
#
# Laptop side: launches only the HMI and the agent. Nav2 and the sensor stack
# run on the ROSbot ('just start-navigation', SLAM off, against the saved map);
# visualisation / initial pose via Foxglove.

set -e

WS_PATH="$HOME/laric_ws"
VENV_PATH="$WS_PATH/laric_env/bin/activate"
LOG_DIR="$WS_PATH/laric_logs/$(date +'%Y%m%d_%H%M%S')_real"
ROBOT_IP="192.168.1.46"

# ROS 2 network: must match the robot (domain 0, DDS discovery on the subnet).
export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET

mkdir -p "$LOG_DIR"

echo "--- Starting LARIC on Real Robot ---"
echo "Expecting ROSbot at $ROBOT_IP (MERCUSYS_ai2), ROS_DOMAIN_ID=$ROS_DOMAIN_ID."

# Sanity check — is the robot on the network?
if ping -c 1 -W 2 "$ROBOT_IP" > /dev/null 2>&1; then
    echo "Robot reachable."
else
    echo "WARNING: $ROBOT_IP not reachable."
    echo "Verify the laptop has connection to MERCUSYS_ai2"
    echo "and that someone has run 'just start-navigation' on the robot."
fi

# HMI before the agent, so its /language_changed publisher exists first.
echo "[1/2] Starting Human-Machine Interface..."
bash -c "source $VENV_PATH && python3 \
    $WS_PATH/src/laric_core/laric_core/laric_interface.py > \
    $LOG_DIR/hmi.log 2>&1" &
sleep 3

echo "[2/2] Starting RAI Agent Logic..."
bash -c "source $VENV_PATH && python3 \
    $WS_PATH/src/laric_core/laric_core/agent_logic.py > \
    $LOG_DIR/agent.log 2>&1" &

echo "--- LARIC client systems online ---"
echo "Foxglove visualization: http://$ROBOT_IP:8080/ui"
echo "Remember to set the initial pose via Foxglove before sending commands."