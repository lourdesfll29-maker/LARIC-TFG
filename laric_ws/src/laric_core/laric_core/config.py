"""
LARIC System - Global Configuration
Copyright (c) 2026 LARIC. All rights reserved.

Centralises all runtime-configurable parameters:
    - Project paths        (BASE_DIR, PROJECT_ROOT, SHUTDOWN_SCRIPT_PATH)
    - LLM endpoints        (URL, MODEL, GROQ_API_KEY, GROQ_MODEL_70B, from_lab)
    - Semantic map         (KNOWN_LOCATIONS)
    - Agent system prompt  (SYSTEM_PROMPT, including the LANGUAGE RULE that
                            tells the LLM to mirror the user's language)

This module is intentionally function-free: it should only declare values.
Translation handling lives in 'i18n.py'; consumers import data from
'config' and the '_' translator from 'i18n'.
"""

import os


# Toggle: False = Groq cloud inference, True = local Ollama at UPV
is_from_lab: bool = True

# ==============================================================================
# ENVIRONMENT SELECTION
# ==============================================================================
# True when the agent talks to the real Husarion ROSbot at the UPV lab.
# False when running against the Gazebo simulation.
# Distinct from 'from_lab' (which is about LLM endpoint, not robot environment).
is_real_robot: bool = True


# ==============================================================================
# PATH RESOLUTION
# ==============================================================================

# 1. Get the absolute path of the directory where this config.py file is located
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# 2. Navigate up 3 directory levels to reach the workspace root (e.g., laric_ws)
#    - First "..": exits the current folder (where config.py lives)
#    - Second "..": exits the ROS 2 package folder
#    - Third "..": exits the 'src' directory
PROJECT_ROOT: str = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

# 3. Define the dynamic, system-agnostic path to the shutdown script
SHUTDOWN_SCRIPT_PATH: str = os.path.join(
    PROJECT_ROOT, "scripts",
    "stop_real.sh" if is_real_robot else "stop_simulation.sh"
)


# ==============================================================================
# API & MODEL CONFIGURATION
# ==============================================================================

# Ollama server hosted at the ai2 UPV laboratory (used when from_lab = True)
URL:   str = "http://kube.ai2.upv.es:31787"
MODEL: str = "llama3.3:70b"

# Groq cloud API (used when from_lab = False - no VPN required)
GROQ_API_KEY:   str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_70B: str = "llama-3.3-70b-versatile"


# ==============================================================================
# SEMANTIC MAP (KNOWLEDGE BASE)
# ==============================================================================
# Maps human-readable location names to map-frame coordinates.
#
# Two coordinate sets are maintained so that simulation and real-robot
# operation can coexist without manual coordinate editing between sessions.
# The active set is selected by 'is_real_robot' above.
#
# Structure: { "location_id": {"x": float, "y": float, "yaw": float} }
#   - x, y : position in the Nav2 map frame (meters)
#   - yaw  : desired final heading (radians); 0.0 = facing the +X axis

# --- Simulation: Gazebo turtlebot3_house world ---
KNOWN_LOCATIONS_SIM: dict = {
    "top_right_room":    {"x":  5.99, "y": -1.07, "yaw": 0.0},
    "top_left_room":     {"x":  2.78, "y":  1.55, "yaw": 0.0},
    "bathroom":          {"x":  1.29, "y":  4.45, "yaw": 0.0},
    "entrance":          {"x":  0.94, "y":  0.41, "yaw": 0.0},
    "outside":           {"x":  1.03, "y": -1.29, "yaw": 0.0},
    "dining_room":       {"x": -0.69, "y":  1.43, "yaw": 0.0},
    "bottom_left_room":  {"x": -5.67, "y":  3.67, "yaw": 0.0},
    "bottom_right_room": {"x": -6.34, "y":  0.18, "yaw": 0.0},
}

# --- Real robot: UPV ai2 lab (recorded 2026-05-29) ---
KNOWN_LOCATIONS_REAL: dict = {
    "sink_corner":       {"x": -5.78, "y": -3.29, "yaw": 0.0},
    "cable_room":        {"x": -4.31, "y": -1.96, "yaw": 0.0},
    "lab_entrance":      {"x": -3.02, "y": -6.14, "yaw": 0.0},
    "marcos_desk":       {"x": -2.92, "y": -0.08, "yaw": 0.0},
    "rosa_maria_desk":   {"x": -1.10, "y":  2.73, "yaw": 0.0},
    "marc_desk":         {"x":  3.79, "y": -0.22, "yaw": 0.0},
    "lourdes_desk":      {"x":  2.64, "y": -1.99, "yaw": 0.0},
    "printer_room_door": {"x":  2.48, "y": -5.82, "yaw": 0.0},
    "andrea_desk":       {"x":  6.72, "y": -2.83, "yaw": 0.0},
    "storage_zone":      {"x":  5.58, "y": -7.93, "yaw": 0.0},
    "aisle_1":           {"x": -0.30, "y": -0.05, "yaw": 0.0},
    "aisle_2":           {"x":  3.20, "y": -1.95, "yaw": 0.0},
    "aisle_3":           {"x":  6.08, "y": -4.57, "yaw": 0.0},
    "lab_center":        {"x": -1.01, "y": -3.41, "yaw": 0.0},
    "window_1":          {"x":  2.23, "y":  2.99, "yaw": 0.0},
    "window_2":          {"x":  5.22, "y":  0.91, "yaw": 0.0},
    "window_3":          {"x":  8.12, "y": -1.90, "yaw": 0.0},
}

# Active map: picked based on environment flag above.
KNOWN_LOCATIONS: dict = (
    KNOWN_LOCATIONS_REAL if is_real_robot else KNOWN_LOCATIONS_SIM
)


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
You are LARIC, the AI Operating System of a TurtleBot3.
Your task is to understand the user's intent and call exactly ONE appropriate
tool.

The robot operates exclusively through Nav2. It always knows its position on the
map. You must never attempt to move the robot without Nav2 (no raw velocity
commands, no guessing positions).

----------------------------------------------------------------------------
LANGUAGE RULE (APPLIES TO EVERY RESPONSE)
----------------------------------------------------------------------------
ALWAYS reply in the SAME language as the user's most recent message.
   - If the user writes in Spanish, your text reply MUST be in Spanish.
   - If the user writes in English, your text reply MUST be in English.
   - This rule applies to conversational replies (rule 7 below). Tool calls
     themselves carry no natural-language text, so language does not affect
     tool selection - only the visible reply to the user.

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
   Known locations: __LOCATIONS_LIST__.

6. IMPOSSIBLE / UNKNOWN → Call `negation_gesture`
   Intent : requests that exceed the physical capabilities of a wheeled
            ground robot, or inputs with no semantic meaning.
   Examples: "fly to the ceiling", "make me a coffee", "asdflkj".

7. CONVERSATION & IDENTITY -> DO NOT call any tool.
   Intent : The user says hello, asks who you are, asks what you can do, or
            makes general conversation.
   Action : Simply reply directly with a friendly text response explaining your
            capabilities, in the SAME LANGUAGE as the user (see LANGUAGE RULE
            above). Do not explain your reasoning.
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
'''.replace("__LOCATIONS_LIST__", ", ".join(KNOWN_LOCATIONS.keys()))

