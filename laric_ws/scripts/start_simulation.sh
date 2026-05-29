#!/usr/bin/env bash

# LARIC System - Startup Script
# Copyright (c) 2026 LARIC. All rights reserved.
#
# Orchestrates the launch of the full LARIC stack in a single command:
#   1. Gazebo                — physics + sensors for the simulated TurtleBot3
#   2. Nav2 + RViz           — localisation, planning, /spin, /drive_on_heading
#   3. HMI (laric_interface) — PyQt5 GUI: PTT voice, text input, EN/ES toggle,
#                              hardware E-Stop button, status log
#   4. Agent (agent_logic)   — LangChain tool-calling agent that subscribes to
#                              /from_human, /initialpose, /emergency_stop, and
#                              /language_changed (kept in its own terminal so
#                              the verbose AgentExecutor trace stays visible)
#
# Each step uses a short 'sleep' to give the previous one time to register its
# ROS 2 topics before the next subscriber comes online. The HMI is started
# BEFORE the agent so it can broadcast its initial language selection on
# /language_changed as soon as the agent subscribes. All non-foreground stdout
# / stderr is captured under $LOG_DIR for post-mortem.


# Exit immediately if a command exits with a non-zero status
set -e

# Configuration variables
WS_PATH="$HOME/laric_ws"
VENV_PATH="$WS_PATH/laric_env/bin/activate"
LOG_DIR="$WS_PATH/laric_logs/$(date +'%Y%m%d_%H%M%S')"

mkdir -p "$LOG_DIR"

echo "--- Starting LARIC Simulation Environment ---"

# 1. Launch Gazebo (Physical World)
#    Loads the turtlebot3_house world. Slowest step — needs ~10 s to publish
#    /clock and the robot's TF tree before Nav2 can start.
echo "[1/4] Starting Gazebo (World)..."
bash -c "source $VENV_PATH && ros2 launch turtlebot3_gazebo \
    turtlebot3_house.launch.py > $LOG_DIR/gazebo.log 2>&1" &

sleep 10

# 2. Launch Navigation Stack (Mapping & Localization)
#    Brings up AMCL, the planners, the behavior servers, and RViz with the
#    pre-built map. use_sim_time:=true is critical so all stamps come from
#    Gazebo's /clock rather than wall time.
echo "[2/4] Starting Nav2 and RViz..."
# TODO: Update the map to the new one once it's ready 
bash -c "source $VENV_PATH && ros2 launch turtlebot3_navigation2 \
    navigation2.launch.py use_sim_time:=true \
    map:=$WS_PATH/maps/tb3_house_map.yaml \
    params_file:=$WS_PATH/src/laric_core/config/laric_nav2_params.yaml \
    > $LOG_DIR/nav2.log 2>&1" &

sleep 5

# 3. Launch HMI (The Voice/Text Interface)
#    Started before the agent so its /language_changed publisher exists by
#    the time the agent's subscription comes online.
echo "[3/4] Starting Human-Machine Interface..."
bash -c "source $VENV_PATH && python3 \
    $WS_PATH/src/laric_core/laric_core/laric_interface.py > \
    $LOG_DIR/hmi.log 2>&1" &

sleep 3

# 4. Launch RAI Agent (The Brain)
#    Runs in its own gnome-terminal window so the AgentExecutor verbose trace
#    remains visible for live debugging of tool calls and reasoning.
echo "[4/4] Starting RAI Agent Logic..."
gnome-terminal --title="LARIC_Agent_Log" -- bash -c "source $VENV_PATH && \
    python3 $WS_PATH/src/laric_core/laric_core/agent_logic.py; exec \
    bash"

echo "--- All systems are online ---"

