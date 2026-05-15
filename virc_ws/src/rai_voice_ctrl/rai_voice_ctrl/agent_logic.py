"""
VIRC System - Agent Logic & ROS 2 Control
Copyright (c) 2026 VIRC. All rights reserved.

Main module implementing ROS 2 robot control via a LangChain tool-calling agent.
All motion is delegated to Nav2 Behavior Plugins (/spin, /drive_on_heading, /backup)
or the Nav2 NavigateToPose action server. The robot must always be localised via
the '2D Pose Estimate' tool in RViz before any motion command is accepted.
"""

import math
import threading
import time
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field
from builtin_interfaces.msg import Duration

# -- ROS 2 -----------------------------------------------------------------------
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point

# -- Nav2 action definitions -----------------------------------------------------
from nav2_msgs.action import BackUp, DriveOnHeading, Spin

# -- LangChain -------------------------------------------------------------------
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# -- RAI -------------------------------------------------------------------------
from rai.communication.ros2 import ROS2Context, ROS2Message
from rai.communication.ros2.connectors.ros2_connector import ROS2Connector
from rai.tools.ros2.navigation.nav2 import (
    CancelNavigateToPoseTool,
    GetNavigateToPoseFeedbackTool,
    GetNavigateToPoseResultTool,
    NavigateToPoseTool
)

# -- Project configuration -------------------------------------------------------
from config import (
    GROQ_API_KEY,
    GROQ_MODEL_70B,
    KNOWN_LOCATIONS,
    MODEL,
    SYSTEM_PROMPT,
    URL,
    from_home,
)


# ==============================================================================
# PYDANTIC INPUT SCHEMAS
# ==============================================================================

class SpinInput(BaseModel):
    """
    Input schema for SpinTool (rotation via the Nav2 /spin action server).

    Attributes:
        angle: Target rotation angle in degrees. Positive values rotate
            counter-clockwise (left); negative values rotate clockwise (right).
    """

    angle: float = Field(
        default=0.0,
        description=(
            "Target rotation angle in degrees. "
            "Positive = left (CCW). Negative = right (CW). "
            "Default direction when unspecified: right (negative). "
            "Convert geometric terms to exact degrees: "
            "'turn 90 degrees' -> -90.0, "
            "'turn 90 degrees to the left' -> 90.0, "
            "'half turn' / 'media vuelta' -> -180.0, "
            "'full turn' / 'vuelta completa' -> -360.0, "
            "'right angle' / 'angulo recto' -> -90.0."
        )
    )
    
    

class MoveInput(BaseModel):
    """
    Input schema for MoveTool (translation via /drive_on_heading or /backup).

    Attributes:
        distance: Signed target distance in meters. Positive is forward;
            negative is backward.
        speed: Desired linear speed in m/s. A value of 0.0 causes the tool
            to apply a safe default (0.3 m/s forward, 0.2 m/s backward).
    """

    distance: float = Field(
        default=0.0,
        description=(
            "Target distance in meters. "
            "Positive = forward. Negative = backward. "
            "Only set this if the user explicitly specifies a distance. "
            "Do NOT infer distance from a duration. "
            "Examples: 'move forward 1 meter' -> 1.0, "
            "'move backwards 2 meters' -> -2.0."
        )
    )
    speed: float = Field(
        default=0.0,
        description=(
            "Desired speed in m/s. "
            "Speed levels: Slow = 0.15, Normal = 0.3, Fast = 0.6. "
            "Use 0.0 to let the tool apply the safe default (0.3 m/s)."
        )
    )
    
    
class SequenceStep(BaseModel):
    """
    A single primitive movement step within a SequenceTool call.

    Attributes:
        action: Type of movement primitive. Must be exactly 'move'
            (linear translation) or 'spin' (in-place rotation).
        value: Numeric target: meters for 'move' steps; degrees for
            'spin' steps. Sign conventions match MoveInput and SpinInput.
        speed: Linear speed in m/s for 'move' steps only. Use 0.0 to
            apply the tool default. Ignored for 'spin' steps.
    """

    action: str = Field(
        description=(
            "Type of primitive action. Must be exactly 'move' or 'spin'."
        )
    )
    value: float = Field(
        description=(
            "Numeric target: meters for 'move', degrees for 'spin'. "
            "Positive distance = forward; negative = backward. "
            "Positive angle = left (CCW); negative = right (CW)."
        )
    )
    speed: float = Field(
        default=0.0,
        description=(
            "Linear speed in m/s. Applies to 'move' steps only. "
            "Use 0.0 to apply the tool default."
        )
    )


class SequenceInput(BaseModel):
    """
    Input schema for SequenceTool (ordered series of primitive movements).

    Attributes:
        steps: Ordered list of movement steps. Maximum 5 steps. Each step
            executes only after the previous one has completed successfully.
    """

    steps: List[SequenceStep] = Field(
        description=(
            "Ordered list of movement steps to execute sequentially. "
            "Maximum 5 steps. "
            "Use for commands that chain movements with connectors such as "
            "'and then', 'after that', 'then', 'y luego', 'y despues'."
        )
    )
    

class NavigationInput(BaseModel):
    """
    Input schema for NavigationTool (goal-based autonomous Nav2 navigation).

    Attributes:
        location_name: Semantic name of the destination room or waypoint.
            Must match a key in 'KNOWN_LOCATIONS'.
        target_x: Explicit X coordinate in meters. Use only when the user
            provides raw map coordinates.
        target_y: Explicit Y coordinate in meters. Use only with 'target_x'.
        target_yaw: Desired final heading in degrees. Defaults to 0.0.
    """

    location_name: str = Field(
        default="",
        description=(
            "Semantic destination name. "
            f"Must match one of: {list(KNOWN_LOCATIONS.keys())}. "
            "Map the user's words to the closest entry in that list. "
            "Convert to lowercase. Leave empty if using raw coordinates."
        )
    )
    target_x: Optional[float] = Field(
        default=None,
        description=(
            "Explicit X coordinate in meters. "
            "Use ONLY if the user provides raw numeric map coordinates."
        )
    )
    target_y: Optional[float] = Field(
        default=None,
        description=(
            "Explicit Y coordinate in meters. Use ONLY together with target_x."
        )
    )
    target_yaw: float = Field(
        default=0.0,
        description=(
            "Desired final heading in degrees. "
            "Use ONLY if the user explicitly specifies a final orientation."
        )
    )


class TriggerInput(BaseModel):
    """
    Minimal input schema for parameter-free tools.

    Attributes:
        execute: Always 'True'. Exists solely to satisfy LangChain's
            requirement that every tool schema declares at least one field.
    """

    execute: bool = Field(
        default=True,
        description="Always True. Triggers the action."
    )


class GestureInput(BaseModel):
    """
    Input schema for GestureTool (physical negation gesture).

    Attributes:
        reason: Brief natural-language explanation of why the user's
            request cannot be fulfilled.
    """

    reason: str = Field(
        default="",
        description="Brief reason why the user's request cannot be fulfilled."
    )


# ==============================================================================
# TOOLS
# ==============================================================================

class SpinTool(BaseTool):
    """
    Rotates the robot in place using the Nav2 '/spin' Behavior Plugin.

    All rotation is handled by the Nav2 action server, which ensures the
    robot's pose estimate is maintained throughout the manoeuvre. Direct
    '/cmd_vel' publishing is intentionally avoided to keep Nav2 costmaps
    consistent.

    Attributes:
        name: LangChain tool identifier. Must match the name used in
            'SYSTEM_PROMPT'.
        description: Human-readable description read by the LLM to decide
            when to invoke this tool.
        args_schema: Pydantic model that validates and documents input.
        connector: Active 'ROS2Connector' instance for action I/O.
        shared_state: Shared mutable dictionary containing 'abort_flag'
            and 'is_busy' keys used for cross-tool coordination.
    """

    name: str = "spin_robot"
    description: str = (
        "Rotates the robot in place by a specific angle in degrees using Nav2. "
        "Use for ALL rotation commands ('gira 90 grados', etc.). "
        "Nav2 MUST be localised via '2D Pose Estimate' in RViz before calling. "
        "Do NOT use for forward/backward movement - use move_robot for that."
    )
    args_schema: Type[BaseModel] = SpinInput
    return_direct: bool = True

    connector: Any = None
    shared_state: dict = None

    def __init__(
        self, 
        connector: ROS2Connector, 
        shared_state: dict, 
        **kwargs
    ) -> None:
        """
        Initialises SpinTool with the ROS 2 connector and shared state.

        Args:
            connector: Active 'ROS2Connector' used to communicate with the
                Nav2 '/spin' action server.
            shared_state: Shared mutable dictionary with 'abort_flag' and
                'is_busy' keys.
            **kwargs: Additional keyword arguments forwarded to 'BaseTool'.
        """
        super().__init__(**kwargs)
        self.connector = connector
        self.shared_state = shared_state

    def _run(self, angle: float) -> str:
        """
        Executes a Nav2-controlled in-place rotation.

        Sends a 'Spin.Goal' to the Nav2'/spin' action server and blocks
        the calling thread until the action completes, times out, or is
        aborted via 'abort_flag'.

        Args:
            angle: Target rotation in degrees. Positive = CCW (left);
                negative = CW (right).

        Returns:
            A status string prefixed with 'SUCCESS', 'ACTION ABORTED',
            or 'ERROR', which the LLM evaluates to formulate its response.
        """
        # 0. Blabla
        if not self.shared_state.get("is_localised", False):
            _publish_feedback(
                self.connector, 
                "[VIRC]: Error - Ensure the robot is localised: use '2D Pose "
                "Estimate' in RViz, then retry."
            )
            return (
                "ACTION FAILED. The robot is not localised. "
                "DO NOT CALL ANY OTHER TOOLS."
            )
        if self.shared_state.get("is_busy", False):
            _publish_feedback(
                self.connector, 
                "[VIRC]: Error - Another action is currently in progress. "
                "Please wait for it to complete or use the 'Stop' tool to "
                "abort it before issuing a new command."
            )
            return (
                "ACTION FAILED. Another action is currently in progress. "
                "DO NOT CALL ANY OTHER TOOLS."
            )

        # 1. Announce the action and reset shared state flags
        _publish_feedback(
            self.connector, 
            f"[VIRC]: Starting rotation of {angle}°..."
        )
        self.shared_state["abort_flag"] = False
        self.shared_state["is_busy"] = True

        # 2. Build the Nav2 Spin goal
        target_yaw_rad = math.radians(angle)
        # time_allowance caps how long Nav2 may attempt the spin.
        # Computed from the expected kinematic duration plus a safety buffer
        # to avoid false timeouts on large rotations (e.g., 360°).
        timeout_float = _compute_spin_allowance(target_yaw_rad)

        goal_dict = {
            "target_yaw": float(target_yaw_rad),
            "time_allowance": {
                "sec": int(timeout_float),
                "nanosec": int((timeout_float - int(timeout_float)) * 1e9)
            }
        }

        # 3. Set up threading synchronisation primitives
        # threading.Event provides atomic set/wait semantics without an 
        # explicit lock, making it preferable to a raw boolean flag for 
        # cross-thread signalling.
        action_done = threading.Event()
        result_container: dict = {"text": ""}

        def _on_done(result: Spin.Result) -> None:
            """
            Signals completion when the Nav2 action server responds.
            """
            result_container["text"] = (
                f"SUCCESS. Rotation of {angle}° completed. "
                "DO NOT CALL ANY OTHER TOOLS."
            )
            action_done.set()

        def _on_feedback(feedback: Any) -> None:
            """
            Incremental feedback handler (unused; reserved for future logging).
            """
            pass

        # 4. Send the goal to the Nav2 /spin action server
        try:
            handle = self.connector.start_action(
                action_data=ROS2Message(payload=goal_dict),
                target="/spin",
                msg_type="nav2_msgs/action/Spin",
                on_feedback=_on_feedback,
                on_done=_on_done,
            )

            # 5. Poll for completion, abort signal, or timeout
            timeout = timeout_float
            elapsed = 0.0
            poll_interval = 0.2  # seconds; balances responsiveness vs CPU load

            while not action_done.is_set() and elapsed < timeout:
                if self.shared_state.get("abort_flag", False):
                    # StopTool has set the flag; 
                    # cancel the Nav2 goal immediately.
                    self.connector.terminate_action(handle)
                    self.shared_state["is_busy"] = False
                    _publish_feedback(
                        self.connector, 
                        "[VIRC]: Rotation cancelled."
                    )
                    return (
                        "ACTION ABORTED. The user stopped the robot. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    )

                time.sleep(poll_interval)
                elapsed += poll_interval

            self.shared_state["is_busy"] = False

            # 6. Evaluate final outcome
            if action_done.is_set():
                _publish_feedback(
                    self.connector, 
                    "[VIRC]: Rotation completed successfully."
                )
                return result_container["text"]
            else:
                # Timeout: terminate the stale action before returning.
                self.connector.terminate_action(handle)
                _publish_feedback(
                    self.connector, 
                    "[VIRC]: Error - Spin timed out."
                )
                return (
                    f"ERROR: Timeout of {timeout:.1f}s reached for spin. "
                    "DO NOT CALL ANY OTHER TOOLS."
                )

        except Exception as exc:
            self.shared_state["is_busy"] = False
            _publish_feedback(
                self.connector, 
                "[VIRC]: Error - Ensure the robot is localised: use '2D Pose "
                "Estimate' in RViz, then retry."
            )
            return (
                f"ERROR: Critical Nav2 exception in SpinTool: {exc}. "
                "DO NOT CALL ANY OTHER TOOLS."
            )


# ------------------------------------------------------------------------------

class MoveTool(BaseTool):
    """
    Moves the robot forward or backward via Nav2 Behavior Plugins.

    Positive distances are handled by the '/drive_on_heading' action server;
    negative distances use '/backup'. Both keep the robot within the Nav2
    localisation framework.

    Attributes:
        name: LangChain tool identifier. Must match the name in 'SYSTEM_PROMPT'.
        description: Read by the LLM to decide when to invoke this tool.
        args_schema: Pydantic model validating and documenting input.
        connector: Active 'ROS2Connector' instance.
        shared_state: Shared mutable dictionary ('abort_flag', 'is_busy').
    """

    name: str = "move_robot"
    description: str = (
        "Moves the robot forward or backward a specific distance in meters "
        "using Nav2. "
        "Positive distance = forward (/drive_on_heading). "
        "Negative distance = backward (/backup). "
        "Nav2 MUST be localised via '2D Pose Estimate' in RViz before calling. "
        "Do NOT use for rotations - use spin_robot for that."
    )
    args_schema: Type[BaseModel] = MoveInput
    return_direct: bool = True

    connector: Any = None
    shared_state: dict = None

    def __init__(self, connector: ROS2Connector, shared_state: dict, **kwargs) -> None:
        """
        Initialises MoveTool with the ROS 2 connector and shared state.

        Args:
            connector: Active 'ROS2Connector' used to communicate with
                '/drive_on_heading' or '/backup' action servers.
            shared_state: Shared mutable dictionary with 'abort_flag' and
                'is_busy' keys.
            **kwargs: Additional keyword arguments forwarded to 'BaseTool'.
        """
        super().__init__(**kwargs)
        self.connector = connector
        self.shared_state = shared_state

    def _run(self, distance: float, speed: float = 0.0) -> str:
        """
        Executes a Nav2-controlled linear translation.

        Dispatches to '/drive_on_heading' for forward motion and '/backup'
        for rearward motion. Blocks the calling thread until the action
        completes, times out, or is aborted.

        Args:
            distance: Signed target distance in meters. Positive = forward;
                negative = backward.
            speed: Desired speed in m/s. If '0.0', a safe default is applied:
                0.3 m/s forward, 0.2 m/s backward.

        Returns:
            A status string prefixed with 'SUCCESS', 'ACTION ABORTED',
            or 'ERROR'.

        Raises:
            No exceptions are raised; all errors are caught and returned as
            string messages so the LLM can report them to the user.
        """
        # 0. Blabla
        if not self.shared_state.get("is_localised", False):
            _publish_feedback(
                self.connector, 
                "[VIRC]: Error - Ensure the robot is localised: use '2D Pose "
                "Estimate' in RViz, then retry."
            )
            return (
                "ACTION FAILED. The robot is not localised. "
                "DO NOT CALL ANY OTHER TOOLS."
            )
        if self.shared_state.get("is_busy", False):
            _publish_feedback(
                self.connector, 
                "[VIRC]: Error - Another action is currently in progress. "
                "Please wait for it to complete or use the 'Stop' tool to "
                "abort it before issuing a new command."
            )
            return (
                "ACTION FAILED. Another action is currently in progress. "
                "DO NOT CALL ANY OTHER TOOLS."
            )

        # 1. Apply speed defaults and determine travel direction
        is_forward = distance >= 0.0
        dist_abs = abs(distance)
        direction_label = "forward" if is_forward else "backward"

        if speed <= 0.0:
            speed = 0.3 if is_forward else 0.2

        # 2. Announce action and reset shared state flags
        _publish_feedback(
            self.connector,
            f"[VIRC]: Moving {dist_abs:.2f} m {direction_label} "
            f"at {speed:.2f} m/s..."
        )
        self.shared_state["abort_flag"] = False
        self.shared_state["is_busy"] = True

        timeout_float = _compute_move_allowance(dist_abs, speed)
        goal_dict = {
            "target": {"x": float(dist_abs), "y": 0.0, "z": 0.0},
            "speed": float(speed),
            "time_allowance": {
                "sec": int(timeout_float),
                "nanosec": int((timeout_float - int(timeout_float)) * 1e9)
            }
        }

        # 3. Build the appropriate Nav2 goal based on direction
        if is_forward:
            target_action = "/drive_on_heading"
            action_type= "nav2_msgs/action/DriveOnHeading"
        else:
            target_action = "/backup"
            action_type= "nav2_msgs/action/BackUp"

        # 4. Set up threading synchronisation
        action_done = threading.Event()
        result_container: dict = {"text": ""}

        def _on_done(result: Union[DriveOnHeading.Result, BackUp.Result]) -> None:
            """
            Signals completion when the Nav2 action server responds.
            """
            result_container["text"] = (
                f"SUCCESS. Moved {dist_abs:.2f} m {direction_label}. "
                "DO NOT CALL ANY OTHER TOOLS."
            )
            action_done.set()

        def _on_feedback(feedback: Any) -> None:
            """
            Incremental feedback handler (reserved for future use).
            """
            pass

        # 5. Send the goal to the selected Nav2 action server
        try:
            handle = self.connector.start_action(
                action_data=ROS2Message(payload=goal_dict),
                target=target_action,
                msg_type=action_type,
                on_feedback=_on_feedback,
                on_done=_on_done,
            )

            # 6. Poll for completion, abort, or timeout
            timeout = timeout_float
            elapsed = 0.0
            poll_interval = 0.2

            while not action_done.is_set() and elapsed < timeout:
                if self.shared_state.get("abort_flag", False):
                    self.connector.terminate_action(handle)
                    self.shared_state["is_busy"] = False
                    _publish_feedback(
                        self.connector, 
                        "[VIRC]: Movement cancelled."
                    )
                    return (
                        "ACTION ABORTED. The user stopped the robot. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    ) 

                time.sleep(poll_interval)
                elapsed += poll_interval

            self.shared_state["is_busy"] = False

            # 7. Evaluate outcome
            if action_done.is_set():
                _publish_feedback(
                    self.connector, 
                    "[VIRC]: Movement completed successfully."
                )
                return result_container["text"]
            else:
                self.connector.terminate_action(handle)
                _publish_feedback(
                    self.connector, 
                    "[VIRC]: Error - Movement timed out."
                )
                return (
                    f"ERROR: Timeout of {timeout:.1f}s reached for movement. "
                    "DO NOT CALL ANY OTHER TOOLS."
                )

        except Exception as exc:
            self.shared_state["is_busy"] = False
            _publish_feedback(
                self.connector, 
                "[VIRC]: Error - Ensure the robot is localised: use '2D Pose "
                "Estimate' in RViz, then retry."
            )
            return (
                "ERROR: Critical Nav2 exception in MoveTool: {exc} "
                "DO NOT CALL ANY OTHER TOOLS."
            )


# ------------------------------------------------------------------------------

class SequenceTool(BaseTool):
    """
    Executes an ordered series of primitive Nav2 movements sequentially.

    This tool exists to work around LangChain's one-tool-per-turn constraint:
    multi-step commands (e.g., "move forward 2 meters and then turn right 90°")
    are packaged into a single tool call and executed step-by-step. Each step
    delegates to the internal '_move_tool' or '_spin_tool' instances, which
    block until Nav2 confirms completion before the next step begins.

    Attributes:
        name: LangChain tool identifier. Must match the name in 'SYSTEM_PROMPT'.
        description: Read by the LLM to decide when to invoke this tool.
        args_schema: Pydantic model validating input.
        connector: Active 'ROS2Connector' instance.
        shared_state: Shared mutable dictionary ('abort_flag', 'is_busy').
    """

    name: str = "execute_sequence"
    description: str = (
        "Executes multiple movements in order using Nav2. "
        "Use INSTEAD of move_robot or spin_robot when the user chains two or "
        "more movements with connectors: 'and then', 'after that', 'then', "
        "'y luego', 'y despues'. "
        "Each step executes only after the previous one completes. "
        "Maximum 5 steps. "
        "Nav2 MUST be localised before calling."
    )
    args_schema: Type[BaseModel] = SequenceInput
    return_direct: bool = True

    connector: Any = None
    shared_state: dict = None
    _spin_tool: Any = None
    _move_tool: Any = None

    def __init__(self, connector: ROS2Connector, shared_state: dict, **kwargs) -> None:
        """
        Initialises SequenceTool and its internal primitive sub-tools.

        Args:
            connector: Active 'ROS2Connector'.
            shared_state: Shared mutable dictionary. Passed to sub-tools so
                that 'abort_flag' propagation works across step boundaries:
                a 'StopTool' call interrupts the currently-running step.
            **kwargs: Additional keyword arguments forwarded to 'BaseTool'.
        """
        super().__init__(**kwargs)
        self.connector = connector
        self.shared_state = shared_state
        # Sub-tools share the same connector and state so abort_flag propagates
        # correctly - stopping mid-sequence halts the active step immediately.
        self._spin_tool = SpinTool(
            connector=connector, 
            shared_state=shared_state
        )
        self._move_tool = MoveTool(
            connector=connector, 
            shared_state=shared_state
        )

    def _run(self, steps: List[SequenceStep]) -> str:
        """
        Executes the sequence steps one by one.

        Performs defensive coercion of LangChain's serialised arguments
        (dicts) to 'SequenceStep' objects before processing.

        Args:
            steps: Ordered list of 'SequenceStep' objects or equivalent
                dicts describing each movement primitive.

        Returns:
            A status string prefixed with 'SUCCESS', 'ACTION ABORTED',
            or 'ERROR' (or an error message naming the failed step).
        """
        # 0. Blabla
        if not self.shared_state.get("is_localised", False):
            _publish_feedback(
                self.connector, 
                "[VIRC]: Error - Ensure the robot is localised: use '2D Pose "
                "Estimate' in RViz, then retry."
            )
            return (
                "ACTION FAILED. The robot is not localised. "
                "DO NOT CALL ANY OTHER TOOLS."
            )

        # 1. Defensive parsing: LangChain serialises tool arguments as plain
        # dicts when the schema contains a List of nested models. Coerce them
        # back to SequenceStep instances before processing.
        parsed_steps: List[SequenceStep] = [
            SequenceStep(**s) if isinstance(s, dict) else s for s in steps
        ]

        _publish_feedback(
            self.connector,
            f"[VIRC]: Starting sequence of {len(parsed_steps)} steps..."
        )

        # 2. Execute each step in order
        for index, step in enumerate(parsed_steps):

            # Check abort flag before starting the next step
            if self.shared_state.get("abort_flag", False):
                _publish_feedback(
                    self.connector, 
                    "[VIRC]: Sequence aborted between steps."
                )
                return (
                        "ACTION ABORTED. Sequence halted by user. "
                        "DO NOT CALL ANY OTHER TOOLS."
                )

            unit = "meters" if step.action.lower() == "move" else "degrees"
            _publish_feedback(
                self.connector,
                f"[VIRC]: Step {index + 1}/{len(parsed_steps)}: "
                f"{step.action} -> {step.value} {unit}."
            )

            # 3. Dispatch to the appropriate primitive tool
            if step.action.lower() == "move":
                result = self._move_tool._run(distance=step.value, speed=step.speed)

            elif step.action.lower() == "spin":
                result = self._spin_tool._run(angle=step.value)

            else:
                _publish_feedback(
                    self.connector,
                    f"[VIRC]: Error - Unknown action type '{step.action}' "
                    f"at step {index + 1}. Valid types are 'move' and 'spin'. "
                )
                return (
                    f"ERROR: Unknown action type '{step.action}' "
                    f"at step {index + 1}. Valid types are 'move' and 'spin'. "
                    "DO NOT CALL ANY OTHER TOOLS."
                )

            # 4. Propagate errors upward; do not continue a failed sequence
            if "ERROR" in result or "ABORTED" in result:
                _publish_feedback(
                    self.connector,
                    f"[VIRC]: Sequence interrupted at step {index + 1}. "
                )
                return (
                    f"Sequence halted at step {index + 1}: {result}. "
                    "DO NOT CALL ANY OTHER TOOLS."
                )

            # Brief settling pause for Nav2 internal state stabilisation
            time.sleep(0.5)

        _publish_feedback(
            self.connector, 
            "[VIRC]: Sequence completed successfully."
        )
        return (
            "SUCCESS. All sequence steps executed successfully. "
            "DO NOT CALL ANY OTHER TOOLS."
        )


# ------------------------------------------------------------------------------

class NavigationTool(BaseTool):
    """
    Sends autonomous navigation goals to Nav2 via the NavigateToPose action 
    server.

    Resolves semantic location names to map coordinates via 'KNOWN_LOCATIONS',
    then delegates to RAI's 'NavigateToPoseTool'. Monitors the result in a
    polling loop on the agent thread so the 'abort_flag' set by 'StopTool'
    remains effective during long navigation missions.

    Attributes:
        name: LangChain tool identifier. Must match the name in 'SYSTEM_PROMPT'.
        description: Read by the LLM to decide when to invoke this tool.
        args_schema: Pydantic model validating input.
        connector: Active 'ROS2Connector' instance.
        shared_state: Shared mutable dictionary ('abort_flag', 'is_busy').
    """

    name: str = "navigate_to_location"
    description: str = (
        "Sends an autonomous navigation goal to Nav2. "
        f"Available semantic locations: {list(KNOWN_LOCATIONS.keys())}. "
        "Can also accept explicit X/Y map coordinates. "
        "Nav2 MUST be localised via '2D Pose Estimate' in RViz before calling. "
        "Use for destination-based commands: 've al baño', etc."
    )
    args_schema: Type[BaseModel] = NavigationInput
    return_direct: bool = True

    connector: Any = None
    shared_state: dict = None
    _rai_nav_tool: Any = None
    _rai_nav_feedback_tool: Any = None
    _rai_nav_result_tool: Any = None

    def __init__(self, connector: ROS2Connector, shared_state: dict, **kwargs) -> None:
        """
        Initialises NavigationTool and its internal RAI sub-tools.

        Args:
            connector: Active 'ROS2Connector' used to communicate with the
                Nav2 '/navigate_to_pose' action server.
            shared_state: Shared mutable dictionary with 'abort_flag' and
                'is_busy' keys.
            **kwargs: Additional keyword arguments forwarded to 'BaseTool'.
        """
        super().__init__(**kwargs)
        self.connector = connector
        self.shared_state = shared_state
        self._rai_nav_tool = NavigateToPoseTool(connector=connector)
        self._rai_nav_feedback_tool = GetNavigateToPoseFeedbackTool(
            connector=connector
        )
        self._rai_nav_result_tool = GetNavigateToPoseResultTool(
            connector=connector
        )

    def _run(
        self,
        location_name: str = "",
        target_x: Optional[float] = None,
        target_y: Optional[float] = None,
        target_yaw: float = 0.0
    ) -> str:
        """
        Resolves the destination and commands Nav2 to navigate there.

        Args:
            location_name: Semantic room or waypoint name. Takes precedence
                over raw coordinates when both are supplied.
            target_x: Explicit X coordinate in meters.
            target_y: Explicit Y coordinate in meters.
            target_yaw: Desired final heading in degrees.

        Returns:
            A status string prefixed with 'SUCCESS', 'ACTION ABORTED',
            'ACTION FAILED', or 'ERROR'.
        """
        # 0. Blabla
        if not self.shared_state.get("is_localised", False):
            _publish_feedback(
                self.connector, 
                "[VIRC]: Error - Ensure the robot is localised: use '2D Pose "
                "Estimate' in RViz, then retry."
            )
            return (
                "ACTION FAILED. The robot is not localised. "
                "DO NOT CALL ANY OTHER TOOLS."
            )
        if self.shared_state.get("is_busy", False):
            _publish_feedback(
                self.connector, 
                "[VIRC]: Error - Another action is currently in progress. "
                "Please wait for it to complete or use the 'Stop' tool to "
                "abort it before issuing a new command."
            )
            return (
                "ACTION FAILED. Another action is currently in progress. "
                "DO NOT CALL ANY OTHER TOOLS."
            )

        # 1. Resolve destination: semantic name takes priority over raw coords
        if target_x is not None and target_y is not None:
            x, y = target_x, target_y
            yaw_rad = math.radians(target_yaw)
            dest_label = f"coordinates ({x:.2f}, {y:.2f})"

        elif location_name:
            loc = location_name.lower().strip()
            if loc not in KNOWN_LOCATIONS:
                _publish_feedback(
                    self.connector,
                    f"[VIRC] - Error: The Location '{loc}' was not found in " 
                    "the semantic map of known locations. You can ask for the " 
                    "known locations."
                )
                return (
                    f"ERROR. The Location '{loc}' was not found in " 
                    "the semantic map of known locations. "
                    "DO NOT CALL ANY OTHER TOOLS. "
                )
            entry = KNOWN_LOCATIONS[loc]
            x, y = entry["x"], entry["y"]
            yaw_rad = entry["yaw"]
            dest_label = f"'{loc.replace('_', ' ')}'"

        else:
            _publish_feedback(
                self.connector,
                "[VIRC] - Error: No valid destination provided. You can ask for "
                "the known locations."
            )
            return (
                "ERROR. No valid destination provided. "
                "DO NOT CALL ANY OTHER TOOLS."
            )

        # 2. Announce intent and reset shared state
        _publish_feedback(
            self.connector, 
            f"[VIRC]: Navigating to {dest_label}..."
        )
        self.shared_state["abort_flag"] = False
        self.shared_state["is_busy"] = True

        # 3. Send the NavigateToPose goal via RAI
        try:
            start_result = str(
                self._rai_nav_tool._run(x=x, y=y, z=0.0, yaw=yaw_rad)
            ).strip()

            # TODO: Comprobar la palabra que emplea RAI para los errores de Nav2
            # -- DEBUG (ELIMINATE) --------------------------------------------
            print(f"\n[DEBUG 952] {start_result}\n") # navigating to pose
            # -----------------------------------------------------------------

        except Exception as exc:
            # -- DEBUG (ELIMINATE) --------------------------------------------
            print(f"\n[DEBUG 957] {exc}\n") # Action goal was not accepted
            # -----------------------------------------------------------------
            self.shared_state["is_busy"] = False
            _publish_feedback(
                self.connector, 
                "[VIRC]: Error - Ensure the robot is localised: use '2D Pose "
                "Estimate' in RViz, then retry."
            )
            return (
                f"ERROR. Failed to send navigation goal: {exc}. "
                "DO NOT CALL ANY OTHER TOOLS." 
            )

        # 4. Poll for the navigation result (maximum 5 minutes)
        # Polling runs on the agent thread to keep abort_flag responsive.
        # The agent cannot accept new tool calls while this loop runs; StopTool
        # sets abort_flag from a concurrent think() thread to break out early.
        timeout = 300.0
        elapsed = 0.0
        poll_interval = 0.5

        while elapsed < timeout:

            # Check whether StopTool has requested an abort
            if self.shared_state.get("abort_flag", False):
                self.shared_state["is_busy"] = False
                _publish_feedback(
                    self.connector, 
                    f"[VIRC]: Navigation to {dest_label} cancelled."
                )
                return (
                    "ACTION ABORTED. User stopped the navigation. "
                    "DO NOT CALL ANY OTHER TOOLS." 
                )

            # Query Nav2 for the current goal status
            try:
                nav_result = str(self._rai_nav_result_tool._run()).strip()

                # -- DEBUG (ELIMINATE) ----------------------------------------
                print(f"\n[DEBUG 997] Navigation result: {nav_result}\n") # Action is not done yet, 6 (ABORTED), 4 (SUCCEED)
                # -------------------------------------------------------------

                if nav_result == str(GoalStatus.STATUS_SUCCEEDED):
                    self.shared_state["is_busy"] = False
                    _publish_feedback(
                        self.connector, 
                        f"[VIRC]: Arrived at {dest_label}.")
                    return (
                        f"SUCCESS. Arrived at {dest_label}. "
                        "DO NOT CALL ANY OTHER TOOLS." 
                    )

                elif nav_result in [
                    str(GoalStatus.STATUS_ABORTED), 
                    str(GoalStatus.STATUS_CANCELED)
                ]:
                    self.shared_state["is_busy"] = False
                    _publish_feedback(
                        self.connector,
                        f"[VIRC]: Error - Could not reach {dest_label}. "
                        "Nav2 aborted."
                    )
                    return (
                        "ACTION FAILED. Nav2 could not find or execute a valid "
                        "path. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    ) 

            except Exception:
                # Transient polling errors are non-fatal; continue waiting.
                pass

            time.sleep(poll_interval)
            elapsed += poll_interval

        # 5. Timeout path
        self.shared_state["is_busy"] = False
        _publish_feedback(
            self.connector,
            f"[VIRC]: Error - Navigation to {dest_label} timed out (5 min)."
        )
        return (
            "ERROR: Navigation timeout (5 minutes). "
            "DO NOT CALL ANY OTHER TOOLS."
        )


# ------------------------------------------------------------------------------

class GestureTool(BaseTool):
    """
    Performs a physical negation gesture via direct '/cmd_vel' publication.

    Produces a brief left-right angular oscillation to communicate "no" to a
    human observer. Direct '/cmd_vel' use is intentional here: the gesture is
    a non-navigational social signal, executes even when Nav2 is idle, and its
    short duration (< 2 s) does not meaningfully perturb AMCL localisation.

    Attributes:
        name: LangChain tool identifier. Must match the name in 'SYSTEM_PROMPT'.
        description: Read by the LLM to decide when to invoke this tool.
        args_schema: Pydantic model validating input.
        connector: Active 'ROS2Connector' for '/cmd_vel' publication.
        shared_state: Shared mutable state dictionary.
    """

    name: str = "negation_gesture"
    description: str = (
        "Makes the robot perform a brief left-right oscillation "
        "(physical 'no' gesture). "
        "Use when the user's request exceeds the robot's physical capabilities "
        "or has no semantic meaning (e.g., 'make me a coffee', 'fly up')."
    )
    args_schema: Type[BaseModel] = GestureInput
    return_direct: bool = True

    # Pydantic/LangChain requires all non-default attributes to be declared
    # as class-level fields; they are assigned their values in __init__.
    connector: Any = None
    shared_state: dict = None

    def __init__(self, connector: ROS2Connector, shared_state: dict, **kwargs) -> None:
        """
        Initialises GestureTool with the ROS 2 connector and shared state.

        Args:
            connector: Active 'ROS2Connector' for '/cmd_vel' publication.
            shared_state: Shared mutable state dictionary.
            **kwargs: Additional keyword arguments forwarded to 'BaseTool'.
        """
        super().__init__(**kwargs)
        self.connector = connector
        self.shared_state = shared_state

    def _run(self, reason: str = "") -> str:
        """
        Executes the oscillation gesture synchronously.

        Args:
            reason: Human-readable explanation of why the request was rejected.

        Returns:
            A status string prefixed with 'SUCCESS' or 'ACTION ABORTED'.
        """
        # 1. Announce the gesture with the rejection reason
        _publish_feedback(
            self.connector, 
            "[VIRC]: (Negation gesture) Request not executable. "
            f"Reason: {reason}"
        )

        self.shared_state["abort_flag"] = False
        self.shared_state["is_busy"] = True

        # 2. Oscillation pattern: alternating left/right angular velocity
        angular_steps = [1.0, -1.0, 1.0, -1.0]

        try:
            for angular_z in angular_steps:
                if self.shared_state.get("abort_flag", False):
                    break

                self.connector.send_message(
                    ROS2Message(payload={
                        "linear":  {"x": 0.0, "y": 0.0, "z": 0.0},
                        "angular": {"x": 0.0, "y": 0.0, "z": angular_z},
                    }),
                    target="/cmd_vel",
                    msg_type="geometry_msgs/msg/Twist",
                )
                time.sleep(0.35)

            # 3. Publish zero-velocity to ensure the robot is fully stationary
            self.connector.send_message(
                ROS2Message(payload={
                    "linear":  {"x": 0.0, "y": 0.0, "z": 0.0},
                    "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
                }),
                target="/cmd_vel",
                msg_type="geometry_msgs/msg/Twist",
            )

            self.shared_state["is_busy"] = False

            if self.shared_state.get("abort_flag", False):
                _publish_feedback(self.connector, "[VIRC]: Gesture interrupted.")
                return (
                    "ACTION ABORTED.  "
                    "DO NOT CALL ANY OTHER TOOLS."
                )

            return (
                f"SUCCESS. Negation gesture executed. Reason: {reason}. "
                "DO NOT CALL ANY OTHER TOOLS." 
            )

        except Exception as exc:
            self.shared_state["is_busy"] = False
            _publish_feedback(
                self.connector, 
                f"[VIRC]: Error - Couldn't execute the negation gesture."
            )
            return (
                f"ERROR executing gesture: {exc}. "
                "DO NOT CALL ANY OTHER TOOLS." 
            )


# ------------------------------------------------------------------------------

class StopTool(BaseTool):
    """
    Issues an immediate emergency stop for all active robot motion.

    Sets 'abort_flag' to interrupt any currently-polling tool (SpinTool,
    MoveTool, NavigationTool), publishes a zero-velocity 'Twist' to
    '/cmd_vel' as an immediate fallback, and cancels any active Nav2 goal
    via RAI's 'CancelNavigateToPoseTool'.

    Attributes:
        name: LangChain tool identifier. Must match the name in 'SYSTEM_PROMPT'.
        description: Read by the LLM to decide when to invoke this tool.
        args_schema: Pydantic model validating input.
        connector: Active 'ROS2Connector' instance.
        shared_state: Shared mutable dictionary ('abort_flag', 'is_busy').
    """

    name: str = "stop_robot"
    description: str = (
        "IMMEDIATELY stops all robot motion (movement, sequences, navigation) "
        "and cancels active Nav2 goals. "
        "Use ONLY when the user explicitly commands a halt: "
        "'stop', 'para', 'detente', 'quieto'. "
        "IMPORTANT - Spanish: isolated 'para' = imperative of 'parar' "
        "(to stop), NOT the preposition 'for'."
    )
    args_schema: Type[BaseModel] = TriggerInput
    return_direct: bool = True

    connector: Any = None
    shared_state: dict = None
    _cancel_nav_tool: Any = None

    def __init__(self, connector: ROS2Connector, shared_state: dict, **kwargs) -> None:
        """
        Initialises StopTool and its internal Nav2 cancellation helper.

        Args:
            connector: Active 'ROS2Connector'.
            shared_state: Shared mutable state dictionary.
            **kwargs: Additional keyword arguments forwarded to 'BaseTool'.
        """
        super().__init__(**kwargs)
        self.connector = connector
        self.shared_state = shared_state
        self._cancel_nav_tool = CancelNavigateToPoseTool(connector=connector)

    def _run(self, **kwargs: Any) -> str:
        """
        Executes the emergency stop sequence.

        Args:
            **kwargs: Ignored; required to satisfy the LangChain tool interface.

        Returns:
            A confirmation string indicating the stop was activated.
        """
        # 1. Signal all polling threads to terminate their current action
        self.shared_state["abort_flag"] = True
        self.shared_state["is_busy"] = False

        # 2. Cancel any active Nav2 goal.
        try:
            self._cancel_nav_tool._run()
        except Exception:
            pass  # Non-fatal if no active goal exists

        # 3. Publish zero velocity to /cmd_vel as an immediate fallback.
        # Note: when Nav2 is active, its controller will override this on the
        # next control cycle. The CancelNavigateToPoseTool call below is the
        # authoritative stop for all Nav2-driven motion.
        try:
            self.connector.send_message(
                ROS2Message(payload={
                    "linear":  {"x": 0.0, "y": 0.0, "z": 0.0},
                    "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
                }),
                target="/cmd_vel",
                msg_type="geometry_msgs/msg/Twist",
            )
        except Exception:
            # Non-fatal: abort_flag and Nav2 cancel are the primary mechanisms
            pass

        _publish_feedback(
            self.connector, 
            "[VIRC]: Stop activated. All motion halted."
        )
        return (
            "SUCCESS. Stop activated. All Nav2 goals cancelled. "
            "DO NOT CALL ANY OTHER TOOLS." 
        )


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def _publish_feedback(connector: ROS2Connector, message: str) -> None:
    """
    Publishes a status message to the '/robot_feedback' ROS 2 topic.

    Safe to call from any thread. Failures are silently swallowed: feedback
    is best-effort and must never propagate into a tool's return value.

    Args:
        connector: Active 'ROS2Connector' instance.
        message: Plaintext status message to publish.
    """
    try:
        connector.send_message(
            ROS2Message(payload={"data": message}),
            target="/robot_feedback",
            msg_type="std_msgs/msg/String"
        )
    except Exception:
        pass


def _compute_spin_allowance(
    target_yaw_rad: float, 
    speed_rad_s: float = 0.5
) -> float:
    """
    Computes a conservative 'time_allowance' for the Nav2 Spin action.

    Args:
        target_yaw_rad: Target rotation magnitude in radians (sign ignored).
        speed_rad_s: Expected rotation speed in rad/s. Defaults to 0.5.

    Returns:
        Estimated duration in seconds, with a 5-second safety buffer and a
        minimum floor of 10 seconds.
    """
    kinematic_estimate = abs(target_yaw_rad) / max(speed_rad_s, 0.01)
    return max(kinematic_estimate + 5.0, 10.0)


def _compute_move_allowance(distance_m: float, speed_m_s: float) -> float:
    """
    Computes a conservative 'time_allowance' for DriveOnHeading / BackUp.

    Args:
        distance_m: Target distance magnitude in meters.
        speed_m_s: Commanded speed in m/s. Clamped to a minimum of 0.01.

    Returns:
        Estimated duration in seconds with a 10-second safety buffer.
    """
    kinematic_estimate = distance_m / max(speed_m_s, 0.01)
    return kinematic_estimate + 10.0


# ==============================================================================
# ENTRY POINT
# ==============================================================================

@ROS2Context()
def main() -> None:
    """
    Main entry point for the VIRC robot agent system.

    Initialises the ROS 2 infrastructure, instantiates LangChain tools,
    connects to the LLM backend, and registers the human-input callback.
    Runs until interrupted with Ctrl-C.

    The '@ROS2Context()' decorator initialises 'rclpy' and guarantees
    proper cleanup on exit.
    """
    # 1. Initialise ROS 2 connector
    ros_connector = ROS2Connector()
    # Brief pause to allow internal publishers to register before the first
    # feedback message is sent.
    time.sleep(3.0)

    _publish_feedback(ros_connector, "[SYSTEM]: Starting VIRC Agent...")

    # 2. Create the shared state dictionary
    # This dict is accessed by multiple threads via the abort_flag and is_busy
    # keys. 
    # TODO:
    # A plain dict is used here to preserve the current architecture;
    # see the Architectural Critique for recommended improvements using
    # threading.Lock or threading.Event.
    robot_state: dict = {
        "abort_flag":   False,  # Set by StopTool; polled by all motion tools
        "is_busy":      False,  # Set by motion tools at start/end of actions
        "is_localised": False   # Blabla
    }

    # 3. Instantiate LangChain tools (Dependency Injection)
    # Each tool receives the connector and shared_state explicitly at
    # construction time, making dependencies visible and testable.
    _publish_feedback(ros_connector, "[SYSTEM]: Loading Nav2 tools...")
    tools = [
        SpinTool(connector=ros_connector, shared_state=robot_state),
        MoveTool(connector=ros_connector, shared_state=robot_state),
        SequenceTool(connector=ros_connector, shared_state=robot_state),
        NavigationTool(connector=ros_connector, shared_state=robot_state),
        GestureTool(connector=ros_connector, shared_state=robot_state),
        StopTool(connector=ros_connector, shared_state=robot_state),
    ]

    # 4. Connect to the LLM backend
    _publish_feedback(ros_connector, "[SYSTEM]: Connecting to LLM...")
    if from_home:
        llm = ChatGroq(
            temperature=0, 
            model=GROQ_MODEL_70B, 
            api_key=GROQ_API_KEY
        )
    else:
        llm = ChatOllama(temperature=0, model=MODEL, base_url=URL)

    # 5. Build the LangChain agent and executor
    prompt = ChatPromptTemplate.from_messages([
        ("system",      SYSTEM_PROMPT),
        ("human",       "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True
    )

    # 6. Define and register the human-input callback
    def on_human_input(msg: ROS2Message) -> None:
        """
        Processes incoming user commands from the '/from_human' topic.

        Extracts the user's text, publishes an acknowledgement, and launches
        the LangChain reasoning cycle in a daemon thread so that the ROS 2
        spin loop is never blocked.

        Args:
            msg: Incoming 'ROS2Message' wrapping a 'std_msgs/String' payload.
        """
        try:
            user_text = msg.payload.data.strip()
        except AttributeError:
            return  # Malformed message; discard silently

        if not user_text:
            return

        _publish_feedback(ros_connector, f"[VIRC]: Heard: '{user_text}'")
        _publish_feedback(ros_connector, "[VIRC]: Thinking...")

        def think() -> None:
            """
            Runs the AgentExecutor in a daemon thread.
            """
            try:
                result = agent_executor.invoke({"input": user_text})
                response = result.get("output", "").strip()

                #Blabla
                is_internal_return = (
                    "DO NOT CALL ANY OTHER TOOLS" in response or
                    response.startswith((
                        "SUCCESS.", "ACTION FAILED.", "ACTION ABORTED.", 
                        "ERROR:", "ERROR."
                    ))
                )

                if response and not is_internal_return:
                    formatted_response = response.replace("\n", "<br>")
                    _publish_feedback(
                        ros_connector, 
                        f"[VIRC]: {formatted_response}"
                    )
            except Exception as exc:
                _publish_feedback(ros_connector, f"[VIRC]: ERROR - {exc}")

        # Daemon threads are automatically joined when the main process exits.
        threading.Thread(target=think, daemon=True).start()

    try:
        ros_connector.register_callback(
            source    = "/from_human",
            topic_name= "/from_human",
            msg_type  = "std_msgs/msg/String",
            callback  = on_human_input
        )
    except Exception as exc:
        _publish_feedback(
            ros_connector, 
            f"[SYSTEM]: Warning - Could not subscribe to /from_human: {exc}"
        )

    # 7. Blabla
    def on_initial_pose(msg: ROS2Message) -> None:
        """
        Detects when the user sets the initial pose in RViz.

        Blabla
        """
        if not robot_state.get("is_localised", False):
            robot_state["is_localised"] = True
            _publish_feedback(
                ros_connector,
                "[VIRC]: Robot localised via '2D Pose Estimate'. "
                "Ready for commands."
            )

    try:
        ros_connector.register_callback(
            source="/initialpose",
            topic_name="/initialpose",
            msg_type="geometry_msgs/msg/PoseWithCovarianceStamped",
            callback=on_initial_pose
        )
    except Exception as exc:
        _publish_feedback(
            ros_connector, 
            f"[SYSTEM]: Warning - Could not subscribe to /initialpose: {exc}"
        )

    _publish_feedback(
        ros_connector,
        "[VIRC]: Agent ready. Listening on /from_human."
    )
    _publish_feedback(
        ros_connector,
        "[VIRC]: IMPORTANT - Use '2D Pose Estimate' in RViz to localise the "
        "robot before sending commands."
    )

    # 8. Hardware Emergency Stop Listener 
    def on_emergency_stop(msg: ROS2Message) -> None:
        """
        Triggers instantly upon receiving the E-Stop signal from the UI.
        Bypasses the LLM entirely to guarantee deterministic braking.

        Blabla
        """
        if msg.payload.data:
            _publish_feedback(
                ros_connector, 
                "[VIRC]: EMERGENCY STOP ACTIVATED."
            )
            
            # Retrieve the StopTool instance from the tools list
            stop_tool = next((t for t in tools if t.name == "stop_robot"), None)
            
            if stop_tool:
                # Execute the halt maneuver directly
                stop_tool._run()

    try:
        ros_connector.register_callback(
            source="/emergency_stop",
            topic_name="/emergency_stop",
            msg_type="std_msgs/msg/Bool",
            callback=on_emergency_stop
        )
    except Exception as exc:
        _publish_feedback(
            ros_connector, 
            f"[SYSTEM]: Warning - Could not subscribe to /emergency_stop: {exc}"
        )

    # 9. Keep-alive loop
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        _publish_feedback(
            ros_connector, 
            "[SYSTEM]: Shutting down VIRC Agent..."
        )


if __name__ == "__main__":
    main()

