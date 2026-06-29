"""
LARIC System - Global Configuration
Copyright (c) 2026 LARIC. All rights reserved.

Centralises all runtime-configurable parameters:
    - Project paths        (BASE_DIR, PROJECT_ROOT, SHUTDOWN_SCRIPT_PATH)
    - LLM endpoints        (URL, MODEL, GROQ_API_KEY, GROQ_MODEL_70B, from_lab)
    - Semantic map         (KNOWN_LOCATIONS)
    - Motion tunables      (SPEED_*, *_ALLOWANCE_*, POLL_*, NAV_TIMEOUT_S,
                            INITIALPOSE_GRACE_S)
    - HMI                  (HMI_GEOMETRY)
    - Agent system prompt  (SYSTEM_PROMPT, including the LANGUAGE RULE that
                            tells the LLM to mirror the user's language)

This module is intentionally function-free: it should only declare values.
Translation handling lives in 'i18n.py'; consumers import data from
'config' and the '_' translator from 'i18n'.
"""

import os


# ==============================================================================
# RUNTIME TOGGLES
# ==============================================================================
# is_from_lab   : False = Groq cloud LLM,    True = local Ollama at the UPV lab.
# is_real_robot : False = Gazebo simulation, True = real Husarion ROSbot.
#                 (is_real_robot is independent of is_from_lab: one selects the
#                 robot environment, the other the LLM endpoint.)
is_from_lab:   bool = False
is_real_robot: bool = False


# ==============================================================================
# PATH RESOLUTION
# ==============================================================================

# 1. Get the absolute path of the directory where this config.py file is located
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# 2. Navigate up 3 directory levels to reach the workspace root
#    - First "..": exits the current folder (where config.py lives)
#    - Second "..": exits the ROS 2 package folder
#    - Third "..": exits the 'src' directory
PROJECT_ROOT: str = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

# Shutdown script for the active environment.
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
# MOTION TUNABLES
# ==============================================================================

# Default linear speeds (m/s) used by MoveTool when the user gives none.
SPEED_SLOW:        float = 0.15
SPEED_NORMAL_FWD:  float = 0.3
SPEED_NORMAL_BACK: float = 0.2
SPEED_FAST:        float = 0.6

# Expected rotation speed (rad/s) used to size the Spin time_allowance.
SPIN_SPEED_RAD_S: float = 0.5

# Safety buffers / floor for the Nav2 action 'time_allowance' (seconds).
SPIN_ALLOWANCE_BUFFER_S: float = 5.0
SPIN_ALLOWANCE_MIN_S:    float = 10.0
MOVE_ALLOWANCE_BUFFER_S: float = 10.0

# Polling intervals (seconds) for the primitive and navigation result loops.
POLL_INTERVAL_S:     float = 0.2
NAV_POLL_INTERVAL_S: float = 0.5

# Maximum time to wait for a navigation goal to finish (seconds).
NAV_TIMEOUT_S: float = 180.0

# Dwell time at each waypoint of a multi-location patrol before the robot
# proceeds to the next one (seconds).
PATROL_DWELL_S: float = 2.0

# Seconds after agent startup during which a retained/stale /initialpose is
# ignored, so a restart does not inherit a previous session's localisation.
INITIALPOSE_GRACE_S: float = 5.0

# Direct /cmd_vel oscillation that signals "no" without going through Nav2.
GESTURE_ANGULAR_VEL_RAD_S:      float = 0.8
GESTURE_WARMUP_DURATION_S:      float = 0.25
GESTURE_OSCILLATION_DURATION_S: float = 0.5
GESTURE_PUBLISH_PERIOD_S:       float = 0.05


# ==============================================================================
# HMI
# ==============================================================================
# (x, y, width, height) of the HMI window. Override without editing this file
# via the env var  LARIC_HMI_GEOMETRY="x,y,w,h".
_g = os.getenv("LARIC_HMI_GEOMETRY", "1300,200,500,600").split(",")
try:
    HMI_GEOMETRY: tuple = (int(_g[0]), int(_g[1]), int(_g[2]), int(_g[3]))
except (ValueError, IndexError):
    HMI_GEOMETRY = (1300, 200, 500, 600)


# ==============================================================================
# SEMANTIC MAP (KNOWLEDGE BASE)
# ==============================================================================
# Location name -> map-frame coordinates {id: {x, y, yaw}} (m, m, rad).
# Two sets (sim / real), selected by 'is_real_robot' above.

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

# --- Real robot: UPV ai2 lab + 3rd-floor corridors (clicked 2026-06-09) ---
KNOWN_LOCATIONS_REAL: dict = {
    "ai2_lab_entrance":      {"x":  2.85, "y":  -7.05, "yaw": 0.0},
    "trash_bins":            {"x":  7.98, "y": -14.01, "yaw": 0.0},
    "healthy_plant":         {"x":  4.63, "y": -13.71, "yaw": 0.0},
    "wilting_plant":         {"x": 11.43, "y": -14.09, "yaw": 0.0},
    "wide_area_access_1":    {"x":  5.16, "y": -10.31, "yaw": 0.0},
    "wide_area_access_2":    {"x": 10.46, "y": -11.05, "yaw": 0.0},
    "networks_lab_3e1":      {"x": 14.57, "y": -10.33, "yaw": 0.0},
    "corridor_3n_entrance":  {"x": -1.69, "y":  -9.36, "yaw": 0.0},
    "restrooms_entrance":    {"x":  3.94, "y": -22.55, "yaw": 0.0},
    "hallway_1_entrance":    {"x": -1.73, "y": -22.19, "yaw": 0.0},
    "hallway_2_entrance":    {"x": 16.12, "y": -23.46, "yaw": 0.0},
    "admin_office_entrance": {"x": 17.73, "y": -10.89, "yaw": 0.0},
    "coat_rack":             {"x":  5.25, "y":  -4.46, "yaw": 0.0},
}

# Active map: picked based on environment flag above.
KNOWN_LOCATIONS: dict = (
    KNOWN_LOCATIONS_REAL if is_real_robot else KNOWN_LOCATIONS_SIM
)


# ==============================================================================
# AGENT SYSTEM PROMPT
# ==============================================================================
# Behavioural contract for the LLM (deterministic, safe tool selection).
#
# CRITICAL - tool names declared here MUST exactly match the `name` attribute
# of each BaseTool subclass in agent_logic.py:
#   spin_robot          -> SpinTool.name
#   move_robot          -> MoveTool.name
#   execute_sequence    -> SequenceTool.name
#   navigate_to_location -> NavigationTool.name
#   navigate_sequence   -> NavigationSequenceTool.name
#   negation_gesture    -> GestureTool.name
#   stop_robot          -> StopTool.name

SYSTEM_PROMPT: str = '''
You are LARIC, the AI control system of a mobile robot (a TurtleBot3 in
simulation or a real Husarion ROSbot 3).
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
   Intent : travelling to a SINGLE known semantic destination (a room, a
            waypoint).
   Schema : location_name (must match an entry in the known locations map).
   Known locations: __LOCATIONS_LIST__.
   CRITICAL: use this ONLY when the user names exactly ONE destination. If the
             user names TWO OR MORE locations, use `navigate_sequence` instead.

6. MULTI-LOCATION PATROL → Call `navigate_sequence`
   Intent : visiting TWO OR MORE known semantic locations in order.
   Schema : locations (an ORDERED list; every entry must match a known
            location).
   Known locations: __LOCATIONS_LIST__.
   CRITICAL: this rule chains NAMED LOCATIONS. Do NOT confuse it with rule 3
             (`execute_sequence`), which chains primitive MOTIONS (distances /
             angles, e.g. "advance 1 m and then turn 90 degrees").
   Map each named place to the closest entry in the known locations list, just
   as for `navigate_to_location`.
   Example:
     "go to the entrance and then the bathroom"
       → locations = ["entrance", "bathroom"]

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