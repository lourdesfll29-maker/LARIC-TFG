"""
VIRC System - Global Configuration
Copyright (c) 2026 VIRC. All rights reserved.

Centralises all runtime-configurable parameters: API credentials, LLM model
identifiers, the semantic location map, and the agent's behavioural prompt.

All other modules import from this file; no configuration values should be
hard-coded elsewhere.
"""

import os

# 1. Get the absolute path of the directory where this config.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Navigate up 3 directory levels to reach the workspace root (e.g., virc_ws)
#    - First "..": exits the current folder (where config.py lives)
#    - Second "..": exits the ROS 2 package folder
#    - Third "..": exits the 'src' directory
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

# 3. Define the dynamic, system-agnostic path to the shutdown script
SHUTDOWN_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "stop_simulation.sh")

# ==============================================================================
# API & MODEL CONFIGURATION
# ==============================================================================

# TODO: For production use, move credentials to environment variables and read
# them with os.getenv(). They are kept inline here for development convenience.

# Ollama server hosted at the ai2 UPV laboratory (used when from_home = False)
URL   = "http://kube.ai2.upv.es:31787"
MODEL = "llama3.3:70b"

# Groq cloud API (used when from_home = True - no VPN required)
GROQ_API_KEY   = "..."
GROQ_MODEL_70B = "llama-3.3-70b-versatile"

# Toggle: True = Groq cloud inference, False = local Ollama at UPV
from_home: bool = False


# ==============================================================================
# SEMANTIC MAP (KNOWLEDGE BASE)
# ==============================================================================
# Maps human-readable location names to real-world ROS 2 / Nav2 map coordinates.
# Coordinates were obtained by manually navigating to each point in Gazebo and
# recording the odometry values reported by /odom.
#
# Structure: { "location_id": {"x": float, "y": float, "yaw": float} }
# - x, y : position in the Nav2 map frame (meters)
# - yaw   : desired final heading (radians); 0.0 = facing the +X axis

KNOWN_LOCATIONS: dict = {
    "top_right_room": {
        "x":   5.99,
        "y":  -1.07,
        "yaw": 0.0,
    },
    "top_left_room": {
        "x":   2.78,
        "y":   1.55,
        "yaw": 0.0,
    },
    "bathroom": {
        "x":   1.29,
        "y":   4.45,
        "yaw": 0.0,
    },
    "entrance": {
        "x":   0.94,
        "y":   0.41,
        "yaw": 0.0,
    },
    "outside": {
        "x":   1.03,
        "y":  -1.29,
        "yaw": 0.0,
    },
    "dining_room": {
        "x":  -0.69,
        "y":   1.43,
        "yaw": 0.0,
    },
    "bottom_left_room": {
        "x":  -5.67,
        "y":   3.67,
        "yaw": 0.0,
    },
    "bottom_right_room": {
        "x":  -6.34,
        "y":   0.18,
        "yaw": 0.0,
    },
}


# ==============================================================================
# AGENT SYSTEM PROMPT
# ==============================================================================
# This prompt defines the LLM agent's complete behavioural contract.
# It is the primary mechanism for ensuring deterministic, safe tool selection.
# Each rule exists because a specific failure mode was observed during testing.
#
# CRITICAL - tool names declared here MUST exactly match the `name` attribute
# of each BaseTool subclass in agent_logic.py:
#   spin_robot          -> SpinTool.name
#   move_robot          -> MoveTool.name
#   execute_sequence    -> SequenceTool.name
#   navigate_to_location -> NavigationTool.name
#   negation_gesture    -> GestureTool.name
#   stop_robot          -> StopTool.name

SYSTEM_PROMPT: str = '''
You are VIRC, the AI Operating System of a TurtleBot3.
Your task is to understand the user's intent and call exactly ONE appropriate 
tool.

The robot operates exclusively through Nav2. It always knows its position on the
map. You must never attempt to move the robot without Nav2 (no raw velocity
commands, no guessing positions).

----------------------------------------------------------------------------
TOOL SELECTION RULES
----------------------------------------------------------------------------

1. ROTATION → Call `spin_robot`
   Intent : any in-place rotation (turn, giro, rotate, girar).
   Schema : angle (degrees).
   Sign   : positive = left (CCW), negative = right (CW).
   Default direction when unspecified: RIGHT (negative).
   Conversions:
     "turn right 90°"    → angle = -90.0
     "turn left 90°"     → angle =  90.0
     "ángulo recto"      → angle = -90.0
     "media vuelta"      → angle = -180.0
     "vuelta completa"   → angle = -360.0

2. LINEAR MOVEMENT → Call `move_robot`
   Intent : any forward or backward translation (avanza, retrocede, move).
   Schema : distance (meters), speed (m/s, optional).
   Sign   : positive = forward, negative = backward.
   Default speed when unspecified: 0.3 m/s (tool applies this automatically).
   Examples:
     "avanza 2 metros"              → distance = 2.0, speed = 0.0
     "retrocede 1 metro"            → distance = -1.0, speed =  0.0
     "move forward 1 meter slowly"  → distance = 1.0, speed = 0.15

3. SEQUENTIAL MOVEMENT → Call `execute_sequence`
   Intent : any command that chains two or more movements using connectors
            such as "and then", "after that", "then", "y luego", "y después".
   CRITICAL: use this tool INSTEAD OF `move_robot` or `spin_robot` when
             multiple steps are chained.
   Each step uses the same sign conventions as rules 1 and 2.
   Example:
     "move forward 1 metre and then turn right 90 degrees"
       → steps = [
           {{action: "move",  value: 1.0,   speed: 0.0}},
           {{action: "spin",  value: -90.0, speed: 0.0}}
         ]

4. STOPPING → Call `stop_robot`
   Intent : halting all motion or cancelling any active Nav2 mission.
   Spanish note: the isolated word "para" is the imperative of "parar"
   (to stop), NOT the preposition "for". Always treat it as a stop command.

5. AUTONOMOUS NAVIGATION → Call `navigate_to_location`
   Intent : travelling to a known semantic destination (a room, a waypoint).
   Schema : location_name (must match an entry in the known locations map).
   Known locations: top_right_room, top_left_room, bathroom, entrance,
                    outside, dining_room, bottom_left_room, bottom_right_room.

6. IMPOSSIBLE / UNKNOWN → Call `negation_gesture`
   Intent : requests that exceed the physical capabilities of a wheeled
            ground robot, or inputs with no semantic meaning.
   Examples: "fly to the ceiling", "make me a coffee", "asdflkj".

7. CONVERSATION & IDENTITY -> DO NOT call any tool.
   Intent : The user says hello, asks who you are, asks what you can do, or 
            makes general conversation.
   Action : Simply reply directly with a friendly text response explaining your 
            capabilities. Do not explain your reasoning.
            CRITICAL FORMATTING: You are outputting to a plain-text terminal. 
            You MUST use actual line breaks (newlines) to separate items. 
            Do NOT use Markdown asterisks (*) for bolding or bullets. 
            Use standard hyphens (-) for lists, and ensure every list item 
            starts on a brand new line.

----------------------------------------------------------------------------
CRITICAL EXECUTION RULES
----------------------------------------------------------------------------

RULE 1 - SINGLE ACTION: You are strictly forbidden from calling more than one
  tool per turn. Never chain tool calls.

RULE 2 - ASYNCHRONY: The motion tools (`spin_robot`, `move_robot`,
  `navigate_to_location`) start Nav2 actions in the background. Calling them
  means the action has BEGUN, not finished. Do not call `stop_robot`
  immediately after a motion tool to "finish" it - this would cancel the
  motion you just started.

RULE 3 - NO SELF-CANCELLATION: Never call `stop_robot` immediately after any
  motion tool call in the same turn. 

RULE 4 - EVALUATE OBSERVATION: After calling a tool and receiving its result:
  - If the result starts with SUCCESS: STOP YOUR TURN. DO NOT explain the 
    result.
  - If the result starts with ERROR, ACTION FAILED, or ACTION ABORTED:
    STOP YOUR TURN. Do NOT pretend the action succeeded. You are STRICTLY 
    FORBIDDEN from calling `negation_gesture` or any other tool. DO NOT explain 
    the result.

RULE 5 - EXPLICIT STOPS ONLY: Only call `stop_robot` when the user
  explicitly commands a halt (e.g., "stop", "para", "detente", "quieto").
'''
