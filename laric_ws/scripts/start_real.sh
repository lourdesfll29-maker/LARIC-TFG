#!/usr/bin/env bash

# LARIC System - Real Robot Startup Script
# Copyright (c) 2026 LARIC. All rights reserved.
#
# Launches the LAPTOP-side components of the LARIC stack:
#   1. HMI (laric_interface) — PyQt5 GUI: PTT voice, text input, EN/ES toggle,
#                              hardware E-Stop button, status log
#   2. Agent (agent_logic)   — LangChain tool-calling agent that subscribes to
#                              /from_human, /initialpose, /emergency_stop, and
#                              /language_changed
#
# Nav2, slam_toolbox (in localization mode against the saved map), the
# RPLidar driver and the rest of the perception/navigation stack run ON
# THE ROSBOT itself, started with 'just start-navigation' from
# ~/rosbot-autonomy on the robot's onboard computer (see Marcos's setup
# documentation). The laptop is a ROS 2 client only.
#
# Visualization is provided by Foxglove at http://172.18.191.80:8080/ui
# (served by the husarion-webui snap on the robot). RViz is intentionally
# NOT launched here.
#
# Setting the initial pose: use Foxglove's "Publish" tool on the 3D map
# panel (set msg type to PoseWithCovarianceStamped, target /initialpose).

set -e

WS_PATH="$HOME/laric_ws"
VENV_PATH="$WS_PATH/laric_env/bin/activate"
LOG_DIR="$WS_PATH/laric_logs/$(date +'%Y%m%d_%H%M%S')_real"
ROBOT_IP="172.18.191.80"

# ROS 2 network — must match the robot. Marcos's setup uses domain 0.
export ROS_DOMAIN_ID=0
# Allow cross-machine DDS discovery (default is 0; explicit for clarity).
export ROS_LOCALHOST_ONLY=0

mkdir -p "$LOG_DIR"

echo "--- Starting LARIC on Real Robot ---"
echo "Expecting ROSbot at $ROBOT_IP (UPV-PSK), ROS_DOMAIN_ID=$ROS_DOMAIN_ID."

# Sanity check — is the robot on the network?
if ping -c 1 -W 2 "$ROBOT_IP" > /dev/null 2>&1; then
    echo "Robot reachable."
else
    echo "WARNING: $ROBOT_IP not reachable. Verify the laptop is on UPV-PSK"
    echo "and that someone has run 'just start-navigation' on the robot."
fi

# 1. Launch HMI (started before the agent so its /language_changed publisher
#    exists by the time the agent's subscription comes online)
echo "[1/2] Starting Human-Machine Interface..."
bash -c "source $VENV_PATH && python3 \
    $WS_PATH/src/laric_core/laric_core/laric_interface.py > \
    $LOG_DIR/hmi.log 2>&1" &

sleep 3

# 2. Launch RAI Agent (runs in its own gnome-terminal so the verbose
#    AgentExecutor trace stays visible for live debugging)
echo "[2/2] Starting RAI Agent Logic..."
gnome-terminal --title="LARIC_Agent_Log" -- bash -c "source $VENV_PATH && \
    python3 $WS_PATH/src/laric_core/laric_core/agent_logic.py; exec \
    bash"

echo "--- LARIC client systems online ---"
echo "Foxglove visualization: http://$ROBOT_IP:8080/ui"
echo "Remember to set the initial pose via Foxglove before sending commands."