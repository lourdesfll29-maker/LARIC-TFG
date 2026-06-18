"""
LARIC System - Agent Logic & ROS 2 Control
Copyright (c) 2026 LARIC. All rights reserved.

Main module implementing ROS 2 robot control via a LangChain tool-calling agent.
All motion is delegated to Nav2 Behavior Plugins (/spin, /drive_on_heading,
/backup) or the Nav2 NavigateToPose action server. The robot must always be
localised before any motion command is accepted. Localisation is signalled by
a pose published on /initialpose: via RViz's '2D Pose Estimate' in simulation,
and via the Foxglove UI on the real Husarion ROSbot.

ROS 2 topics consumed:
  /from_human         (std_msgs/String)  - user commands routed to the LLM.
  /initialpose        (geometry_msgs/PoseWithCovarianceStamped) - localisation.
  /emergency_stop     (std_msgs/Bool)    - hardware E-Stop from the HMI.
  /language_changed   (std_msgs/String)  - new language code from the HMI,
                                           propagated to the i18n catalog so
                                           operator-facing feedback matches
                                           the language the user selected.

ROS 2 topics produced:
  /robot_feedback     (std_msgs/String)  - status messages shown in the HMI
                                           log; every message is wrapped with
                                           i18n._() so the language follows
                                           the active catalog.

Internationalisation:
  All operator-facing strings flow through 'i18n._()'. The translator is
  resolved at call time, so a runtime 'set_language(...)' (driven by the
  /language_changed subscription below) takes effect immediately for every
  subsequent _publish_feedback. Internal LLM-facing return strings
  (SUCCESS / ACTION FAILED / ACTION ABORTED / ERROR) are intentionally NOT
  translated: SYSTEM_PROMPT's rule 4 matches on those English prefixes.
"""

import math
import threading
import time
from typing import Any, List, Optional, Type, Union

# ------------------------------------------------------------------------------
# Noise suppression — must run BEFORE third-party imports. Silences three
# harmless sources: pydub's SyntaxWarning (Py3.12 non-raw regex), langgraph's
# PendingDeprecationWarning, and rai's 'rai_interfaces is not installed'
# WARNINGs (dropped via the logging.Filter below).
# ------------------------------------------------------------------------------
import logging
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


class _RAIInterfacesNoise(logging.Filter):
    """Drops the 'rai_interfaces is not installed' root-logger WARNINGs."""

    _NEEDLES = ("rai_interfaces is not installed", "based on rai_interfaces")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(needle in msg for needle in self._NEEDLES)


logging.getLogger().addFilter(_RAIInterfacesNoise())

from pydantic import BaseModel, Field

# -- ROS 2 ---------------------------------------------------------------------
from action_msgs.msg import GoalStatus

# -- Nav2 action definitions ---------------------------------------------------
from nav2_msgs.action import BackUp, DriveOnHeading, Spin

# -- LangChain -----------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.tools import BaseTool
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
    from langchain_ollama import ChatOllama

# -- RAI -----------------------------------------------------------------------
from rai.communication.ros2 import ROS2Context, ROS2Message
from rai.communication.ros2.connectors.ros2_connector import ROS2Connector
from rai.tools.ros2.navigation.nav2 import (
    CancelNavigateToPoseTool,
    GetNavigateToPoseFeedbackTool,
    GetNavigateToPoseResultTool,
    NavigateToPoseTool
)

# -- Project configuration & i18n ----------------------------------------------
from config import (
    GROQ_API_KEY,
    GROQ_MODEL_70B,
    KNOWN_LOCATIONS,
    MODEL,
    SYSTEM_PROMPT,
    URL,
    is_from_lab,
    is_real_robot,
    SPEED_NORMAL_FWD,
    SPEED_NORMAL_BACK,
    SPIN_SPEED_RAD_S,
    SPIN_ALLOWANCE_BUFFER_S,
    SPIN_ALLOWANCE_MIN_S,
    MOVE_ALLOWANCE_BUFFER_S,
    POLL_INTERVAL_S,
    NAV_POLL_INTERVAL_S,
    NAV_TIMEOUT_S,
    INITIALPOSE_GRACE_S,
)
from i18n import _, set_language


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
        steps: Ordered list of movement steps. Each step executes only after the
            previous one has completed successfully.
    """

    steps: List[SequenceStep] = Field(
        description=(
            "Ordered list of movement steps to execute sequentially. "
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

    Attributes:
        name: LangChain tool identifier. Must match the name used in
            'SYSTEM_PROMPT'.
        description: Human-readable description read by the LLM to decide
            when to invoke this tool.
        args_schema: Pydantic model that validates and documents input.
        connector: Active 'ROS2Connector' instance for action I/O.
        abort_event: Shared 'threading.Event'; set by 'StopTool' to cancel
            an in-flight motion.
        motion_lock: Shared 'threading.RLock'; acquired non-blockingly at
            entry to guarantee at most one motion runs at a time.
        localised_event: Shared 'threading.Event'; must be set (robot
            localised) before any motion runs.
    """

    name: str = "spin_robot"
    description: str = (
        "Rotates the robot in place by a specific angle in degrees using Nav2. "
        "Use for ALL rotation commands ('gira 90 grados', etc.). "
        "Nav2 MUST be localised before calling. "
        "Do NOT use for forward/backward movement - use move_robot for that."
    )
    args_schema: Type[BaseModel] = SpinInput
    return_direct: bool = True

    connector:       Any = None
    abort_event:     Any = None
    motion_lock:     Any = None
    localised_event: Any = None

    def __init__(
        self, 
        connector: ROS2Connector, 
        abort_event: threading.Event,
        motion_lock: threading.Lock,
        localised_event: threading.Event,
        **kwargs
    ) -> None:
        """
        Initialises SpinTool with the ROS 2 connector and shared state.

        Args:
            connector: Active 'ROS2Connector' used to communicate with the
                Nav2 '/spin' action server.
            abort_event: Shared Event; set by StopTool to cancel motion.
            motion_lock: Shared Lock; prevents concurrent motion commands.
            localised_event: Shared Event; must be set (robot localised) before
                any motion is permitted.
            **kwargs: Additional keyword arguments forwarded to 'BaseTool'.
        """
        super().__init__(**kwargs)
        self.connector       = connector
        self.abort_event     = abort_event
        self.motion_lock     = motion_lock
        self.localised_event = localised_event

    def _run(self, angle: float, standalone: bool = True) -> str:
        """
        Executes a Nav2-controlled in-place rotation.

        Args:
            angle: Target rotation in degrees. Positive = CCW (left);
                negative = CW (right).
            standalone: True for a normal single command. False when invoked by
                SequenceTool, which owns motion_lock and the abort_event reset
                for the whole sequence.

        Returns:
            A status string prefixed with 'SUCCESS', 'ACTION ABORTED',
            or 'ERROR', which the LLM evaluates to formulate its response.
        """
        # 0. Check preconditions
        if not self.localised_event.is_set():
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - The robot is not localised. Set its "
                  "initial pose on the map, then retry.")
            )
            return (
                "ACTION FAILED. The robot is not localised. "
                "DO NOT CALL ANY OTHER TOOLS."
            )
        if not self.motion_lock.acquire(blocking=False):
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - Another action is currently in progress. "
                  "Please wait for it to complete or use 'stop_robot' to "
                  "abort it before issuing a new command.")
            )
            return (
                "ACTION FAILED. Another action is currently in progress. "
                "DO NOT CALL ANY OTHER TOOLS."
            )

        try:
            if standalone:
                self.abort_event.clear()

            # 1. Announce the action
            _publish_feedback(
                self.connector,
                _("[LARIC]: Starting rotation of {angle}°...")
                .format(angle=angle)
            )

            # 2. Build the Nav2 Spin goal
            target_yaw_rad = math.radians(angle)
            timeout_float = _compute_spin_allowance(target_yaw_rad)

            goal_dict = {
                "target_yaw": float(target_yaw_rad),
                "time_allowance": {
                    "sec": int(timeout_float),
                    "nanosec": int((timeout_float - int(timeout_float)) * 1e9)
                }
            }

            # 3. Set up threading synchronisation primitives
            action_done      = threading.Event()
            result_container = {"text": ""}

            def _on_done(result: Spin.Result) -> None:
                """
                Signals completion when the Nav2 action server responds.
                """
                status = result.result().status
                if status == GoalStatus.STATUS_SUCCEEDED:
                    result_container["text"] = (
                        f"SUCCESS. Rotation of {angle}° completed. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    )
                else:
                    result_container["text"] = (
                        f"ACTION FAILED. Nav2 returned status {status}. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    )
                action_done.set()

            def _on_feedback(feedback: Any) -> None:
                """
                Incremental feedback handler (currently unused).
                """
                pass

            # 4. Send the goal to the Nav2 /spin action server
            handle = self.connector.start_action(
                action_data=ROS2Message(payload=goal_dict),
                target="/spin",
                msg_type="nav2_msgs/action/Spin",
                on_feedback=_on_feedback,
                on_done=_on_done,
            )

            # 5. Poll for completion, abort signal, or timeout
            timeout       = timeout_float
            elapsed       = 0.0
            poll_interval = POLL_INTERVAL_S

            while not action_done.is_set() and elapsed < timeout:
                if self.abort_event.is_set():
                    # StopTool sets the flag; cancels the Nav2 goal inmediately.
                    self.connector.terminate_action(handle)
                    _publish_feedback(
                        self.connector,
                        _("[LARIC]: Rotation cancelled.")
                    )
                    return (
                        "ACTION ABORTED. The user stopped the robot. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    )

                time.sleep(poll_interval)
                elapsed += poll_interval

            # 6. Evaluate final outcome
            if action_done.is_set():
                if "SUCCESS" in result_container["text"]:
                    _publish_feedback(
                        self.connector,
                        _("[LARIC]: Rotation completed successfully.")
                    )
                else:
                    _publish_feedback(
                        self.connector,
                        _("[LARIC]: Rotation could not be completed. "
                          "Obstacle detected.")
                    )
                return result_container["text"]
            else:
                # Timeout: terminate the stale action before returning.
                self.connector.terminate_action(handle)
                _publish_feedback(
                    self.connector,
                    _("[LARIC]: Error - Spin timed out.")
                )
                return (
                    f"ERROR: Timeout of {timeout:.1f}s reached for spin. "
                    "DO NOT CALL ANY OTHER TOOLS."
                )

        except Exception as exc:
            # No need to clear any busy flag here: the lock release is handled
            # unconditionally by the 'finally' below.
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - The robot is not localised. Set its "
                  "initial pose on the map, then retry.")
            )
            return (
                f"ERROR: Critical Nav2 exception in SpinTool: {exc}. "
                "DO NOT CALL ANY OTHER TOOLS."
            )
        
        finally:
            self.motion_lock.release()


# ------------------------------------------------------------------------------

class MoveTool(BaseTool):
    """
    Moves the robot forward or backward via Nav2 Behavior Plugins.

    Attributes:
        name: LangChain tool identifier. Must match the name in 'SYSTEM_PROMPT'.
        description: Read by the LLM to decide when to invoke this tool.
        args_schema: Pydantic model validating and documenting input.
        connector: Active 'ROS2Connector' instance.
        abort_event: Shared 'threading.Event'; set by 'StopTool' to cancel
            an in-flight motion.
        motion_lock: Shared 'threading.RLock'; acquired non-blockingly at
            entry to serialise motion commands.
        localised_event: Shared 'threading.Event'; must be set before any
            motion is attempted.
    """

    name: str = "move_robot"
    description: str = (
        "Moves the robot forward or backward a specific distance in meters "
        "using Nav2. "
        "Positive distance = forward (/drive_on_heading). "
        "Negative distance = backward (/backup). "
        "Nav2 MUST be localised before calling. "
        "Do NOT use for rotations - use spin_robot for that."
    )
    args_schema: Type[BaseModel] = MoveInput
    return_direct: bool = True

    connector:       Any = None
    abort_event:     Any = None
    motion_lock:     Any = None
    localised_event: Any = None

    def __init__(
        self,
        connector: ROS2Connector,
        abort_event: threading.Event,
        motion_lock: threading.Lock,
        localised_event: threading.Event,
        **kwargs,
    ) -> None:
        """
        Initialises MoveTool with the ROS 2 connector and shared state.

        Args:
            connector: Active 'ROS2Connector' used to communicate with
                '/drive_on_heading' or '/backup' action servers.
            abort_event: Shared Event for stop signalling.
            motion_lock: Shared Lock; prevents concurrent motion.
            localised_event: Shared Event; robot localisation prerequisite.
            **kwargs: Additional keyword arguments forwarded to 'BaseTool'.
        """
        super().__init__(**kwargs)
        self.connector       = connector
        self.abort_event     = abort_event
        self.motion_lock     = motion_lock
        self.localised_event = localised_event

    def _run(self, distance: float, speed: float = 0.0,
             standalone: bool = True) -> str:
        """
        Executes a Nav2-controlled linear translation.

        Args:
            distance: Signed target distance in meters. Positive = forward;
                negative = backward.
            speed: Desired speed in m/s. If '0.0', the configured default is
                applied: SPEED_NORMAL_FWD forward, SPEED_NORMAL_BACK backward.
            standalone: True for a normal single command. False when invoked by
                SequenceTool (which owns motion_lock and the abort reset).

        Returns:
            A status string prefixed with 'SUCCESS', 'ACTION ABORTED',
            or 'ERROR'.

        Raises:
            No exceptions are raised; all errors are caught and returned as
            string messages so the LLM can report them to the user.
        """
        # 0. Check preconditions
        if not self.localised_event.is_set():
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - The robot is not localised. Set its "
                  "initial pose on the map, then retry.")
            )
            return (
                "ACTION FAILED. The robot is not localised. "
                "DO NOT CALL ANY OTHER TOOLS."
            )
        # Non-blocking acquire: True only if the lock was free. A False result
        # means another motion is already running, so the command is rejected.
        if not self.motion_lock.acquire(blocking=False):
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - Another action is currently in progress. "
                  "Please wait for it to complete or use 'stop_robot' to "
                  "abort it before issuing a new command.")
            )
            return (
                "ACTION FAILED. Another action is currently in progress. "
                "DO NOT CALL ANY OTHER TOOLS."
            )

        try:
            # 1. Apply speed defaults and determine travel direction
            is_forward = distance >= 0.0
            dist_abs = abs(distance)
            direction_label = _("forward") if is_forward else _("backward")

            if speed <= 0.0:
                speed = (SPEED_NORMAL_FWD if is_forward
                         else SPEED_NORMAL_BACK)

            if standalone:
                self.abort_event.clear()
            # 2. Announce action
            _publish_feedback(
                self.connector,
                _("[LARIC]: Moving {dist:.2f} m {direction} "
                  "at {speed:.2f} m/s...")
                .format(dist=dist_abs, direction=direction_label, speed=speed)
            )

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
                action_type   = "nav2_msgs/action/DriveOnHeading"
            else:
                target_action = "/backup"
                action_type   = "nav2_msgs/action/BackUp"

            # 4. Set up threading synchronisation
            action_done      = threading.Event()
            result_container = {"text": ""}

            def _on_done(
                    result: Union[DriveOnHeading.Result, BackUp.Result]
            ) -> None:
                """
                Signals completion when the Nav2 action server responds.
                """
                status = result.result().status
                if status == GoalStatus.STATUS_SUCCEEDED:
                    result_container["text"] = (
                        f"SUCCESS. Moved {dist_abs:.2f} m {direction_label}. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    )
                else:
                    result_container["text"] = (
                        f"ACTION FAILED. Nav2 returned status {status}. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    )
                action_done.set()

            def _on_feedback(feedback: Any) -> None:
                """
                Incremental feedback handler (currently unused).
                """
                pass

            # 5. Send the goal to the selected Nav2 action server
            handle = self.connector.start_action(
                action_data=ROS2Message(payload=goal_dict),
                target=target_action,
                msg_type=action_type,
                on_feedback=_on_feedback,
                on_done=_on_done,
            )

            # 6. Poll for completion, abort, or timeout
            timeout       = timeout_float
            elapsed       = 0.0
            poll_interval = POLL_INTERVAL_S

            while not action_done.is_set() and elapsed < timeout:
                if self.abort_event.is_set():
                    self.connector.terminate_action(handle)
                    _publish_feedback(
                        self.connector,
                        _("[LARIC]: Movement cancelled.")
                    )
                    return (
                        "ACTION ABORTED. The user stopped the robot. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    )

                time.sleep(poll_interval)
                elapsed += poll_interval

            # 7. Evaluate outcome
            if action_done.is_set():
                if "SUCCESS" in result_container["text"]:
                    _publish_feedback(
                        self.connector,
                        _("[LARIC]: Movement completed successfully.")
                    )
                else:
                    _publish_feedback(
                        self.connector,
                        _("[LARIC]: Movement could not be completed. "
                          "Obstacle detected.")
                    )
                return result_container["text"]
            else:
                self.connector.terminate_action(handle)
                _publish_feedback(
                    self.connector,
                    _("[LARIC]: Error - Movement timed out.")
                )
                return (
                    f"ERROR: Timeout of {timeout:.1f}s reached for movement. "
                    "DO NOT CALL ANY OTHER TOOLS."
                )

        except Exception as exc:
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - The robot is not localised. Set its "
                  "initial pose on the map, then retry.")
            )
            return (
                f"ERROR: Critical Nav2 exception in MoveTool: {exc}. "
                "DO NOT CALL ANY OTHER TOOLS."
            )
        
        finally:
            self.motion_lock.release()


# ------------------------------------------------------------------------------

class SequenceTool(BaseTool):
    """
    Executes an ordered series of primitive Nav2 movements sequentially.

    Attributes:
        name: LangChain tool identifier. Must match the name in 'SYSTEM_PROMPT'.
        description: Read by the LLM to decide when to invoke this tool.
        args_schema: Pydantic model validating input.
        connector: Active 'ROS2Connector' instance.
        abort_event: Shared 'threading.Event'; propagated to the sub-tools so
            'StopTool' can halt the active step mid-sequence.
        motion_lock: Shared 'threading.RLock'; acquired once for
            the whole sequence so no other command can interleave between steps.
        localised_event: Shared 'threading.Event'; checked once here.
        _spin_tool: Internal 'SpinTool' instance used for spin steps.
        _move_tool: Internal 'MoveTool' instance used for move steps.
    """

    name: str = "execute_sequence"
    description: str = (
        "Executes multiple movements in order using Nav2. "
        "Use INSTEAD of move_robot or spin_robot when the user chains two or "
        "more movements with connectors: 'and then', 'after that', 'then', "
        "'y luego', 'y despues'. "
        "Each step executes only after the previous one completes. "
        "Nav2 MUST be localised before calling."
    )
    args_schema: Type[BaseModel] = SequenceInput
    return_direct: bool = True

    connector:       Any = None
    abort_event:     Any = None
    motion_lock:     Any = None
    localised_event: Any = None
    _spin_tool:      Any = None
    _move_tool:      Any = None

    def __init__(
        self,
        connector: ROS2Connector,
        abort_event: threading.Event,
        motion_lock: threading.Lock,
        localised_event: threading.Event,
        **kwargs,
    ) -> None:
        """
        Initialises SequenceTool and its internal primitive sub-tools.

        Args:
            connector: Active 'ROS2Connector'.
            abort_event: Threading event to signal abort requests.
            motion_lock: Threading lock to synchronize motion commands.
            localised_event: Threading event to signal localisation status.
            **kwargs: Additional keyword arguments forwarded to 'BaseTool'.
        """
        super().__init__(**kwargs)
        self.connector = connector
        self.abort_event = abort_event
        self.motion_lock = motion_lock
        self.localised_event = localised_event
        # Sub-tools share the same connector and state so abort_event propagates
        # correctly - stopping mid-sequence halts the active step immediately.
        self._spin_tool = SpinTool(
            connector=connector, 
            abort_event=abort_event,
            motion_lock=motion_lock,
            localised_event=localised_event
        )
        self._move_tool = MoveTool(
            connector=connector, 
            abort_event=abort_event,
            motion_lock=motion_lock,
            localised_event=localised_event
        )

    def _run(self, steps: List[SequenceStep]) -> str:
        """
        Executes the sequence steps one by one.

        Args:
            steps: Ordered list of 'SequenceStep' objects or equivalent
                dicts describing each movement primitive.

        Returns:
            A status string prefixed with 'SUCCESS', 'ACTION ABORTED',
            or 'ERROR' (or an error message naming the failed step).
        """
        # 0. Preconditions, checked once for the whole sequence.
        if not self.localised_event.is_set():
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - The robot is not localised. Set its "
                  "initial pose on the map, then retry.")
            )
            return (
                "ACTION FAILED. The robot is not localised. "
                "DO NOT CALL ANY OTHER TOOLS."
            )
        if not self.motion_lock.acquire(blocking=False):
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - Another action is currently in progress. "
                  "Please wait for it to complete or use 'stop_robot' to "
                  "abort it before issuing a new command.")
            )
            return (
                "ACTION FAILED. Another action is currently in progress. "
                "DO NOT CALL ANY OTHER TOOLS."
            )

        try:
            self.abort_event.clear()

            # 1. Coerce serialised dicts back to SequenceStep objects.
            parsed_steps: List[SequenceStep] = [
                SequenceStep(**s) if isinstance(s, dict) else s for s in steps
            ]

            _publish_feedback(
                self.connector,
                _("[LARIC]: Starting sequence of {n} steps...")
                .format(n=len(parsed_steps))
            )

            # 2. Execute each step in order
            for index, step in enumerate(parsed_steps):

                # Check abort flag before starting the next step
                if self.abort_event.is_set():
                    _publish_feedback(
                        self.connector,
                        _("[LARIC]: Sequence aborted between steps.")
                    )
                    return (
                        "ACTION ABORTED. Sequence halted by user. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    )

                unit = (_("meters") if step.action.lower() == "move"
                        else _("degrees"))
                _publish_feedback(
                    self.connector,
                    _("[LARIC]: Step {i}/{n}: {action} -> {value} {unit}.")
                    .format(
                        i=index + 1, n=len(parsed_steps),
                        action=step.action, value=step.value, unit=unit,
                    )
                )

                # 3. Dispatch (standalone=False: this class owns the lock and
                #    the abort reset).
                if step.action.lower() == "move":
                    result = self._move_tool._run(
                        distance=step.value, speed=step.speed, standalone=False
                    )

                elif step.action.lower() == "spin":
                    result = self._spin_tool._run(
                        angle=step.value, standalone=False
                    )

                else:
                    _publish_feedback(
                        self.connector,
                        _("[LARIC]: Error - Unknown action type '{action}' at "
                          "step {i}. Valid types are 'move' and 'spin'.")
                        .format(action=step.action, i=index + 1)
                    )
                    return (
                        f"ERROR: Unknown action type '{step.action}' at step "
                        f"{index + 1}. Valid types are 'move' and 'spin'. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    )

                # 4. Propagate errors upward; do not continue a failed sequence
                if "ERROR" in result or "ABORTED" in result:
                    _publish_feedback(
                        self.connector,
                        _("[LARIC]: Sequence interrupted at step {i}.")
                        .format(i=index + 1)
                    )
                    return (
                        f"Sequence halted at step {index + 1}: {result}. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    )

                # Brief settling pause for Nav2 internal state stabilisation
                time.sleep(0.5)

            _publish_feedback(
                self.connector,
                _("[LARIC]: Sequence completed successfully.")
            )
            return (
                "SUCCESS. All sequence steps executed successfully. "
                "DO NOT CALL ANY OTHER TOOLS."
            )
        finally:
            self.motion_lock.release()


# ------------------------------------------------------------------------------

class NavigationTool(BaseTool):
    """
    Sends autonomous navigation goals to Nav2 via the NavigateToPose action 
    server.

    Attributes:
        name: LangChain tool identifier. Must match the name in 'SYSTEM_PROMPT'.
        description: Read by the LLM to decide when to invoke this tool.
        args_schema: Pydantic model validating input.
        connector: Active 'ROS2Connector' instance.
        abort_event: Shared 'threading.Event'; observed by the polling loop
            so 'StopTool' can interrupt a long mission.
        motion_lock: Shared 'threading.RLock' serialising motion commands.
        localised_event: Shared 'threading.Event'; prerequisite for sending
            navigation goals.
        _rai_nav_tool: RAI 'NavigateToPoseTool' that sends the goal.
        _rai_nav_feedback_tool: RAI helper for progress feedback.
        _rai_nav_result_tool: RAI helper for the final result status.
    """

    name: str = "navigate_to_location"
    description: str = (
        "Sends an autonomous navigation goal to Nav2. "
        f"Available semantic locations: {list(KNOWN_LOCATIONS.keys())}. "
        "Can also accept explicit X/Y map coordinates. "
        "Nav2 MUST be localised before calling. "
        "Use for destination-based commands: 've al baño', etc."
    )
    args_schema: Type[BaseModel] = NavigationInput
    return_direct: bool = True

    connector:            Any = None
    abort_event:          Any = None
    motion_lock:          Any = None
    localised_event:      Any = None
    _rai_nav_tool:        Any = None
    _rai_nav_feedback_tool: Any = None
    _rai_nav_result_tool: Any = None

    def __init__(
        self,
        connector: ROS2Connector,
        abort_event: threading.Event,
        motion_lock: threading.Lock,
        localised_event: threading.Event,
        **kwargs,
    ) -> None:
        """
        Initialises NavigationTool and its internal RAI sub-tools.

        Args:
            connector: Active 'ROS2Connector' used to communicate with the
                Nav2 '/navigate_to_pose' action server.
            abort_event: Threading event to signal abort requests.
            motion_lock: Threading lock to synchronize motion commands.
            localised_event: Threading event to signal localisation status.
            **kwargs: Additional keyword arguments forwarded to 'BaseTool'.
        """
        super().__init__(**kwargs)
        self.connector = connector
        self.abort_event = abort_event
        self.motion_lock = motion_lock
        self.localised_event = localised_event
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
        # 0. Check preconditions
        if not self.localised_event.is_set():
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - The robot is not localised. Set its "
                  "initial pose on the map, then retry.")
            )
            return (
                "ACTION FAILED. The robot is not localised. "
                "DO NOT CALL ANY OTHER TOOLS."
            )
        # Non-blocking acquire (see MoveTool): a False result means another
        # motion holds the lock, so the command is rejected.
        if not self.motion_lock.acquire(blocking=False):
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - Another action is currently in progress. "
                  "Please wait for it to complete or use 'stop_robot' to "
                  "abort it before issuing a new command.")
            )
            return (
                "ACTION FAILED. Another action is currently in progress. "
                "DO NOT CALL ANY OTHER TOOLS."
            )

        try:
            # 1. Resolve destination: semantic name beats raw coordinates.
            if target_x is not None and target_y is not None:
                x, y = target_x, target_y
                yaw_rad = math.radians(target_yaw)
                dest_label = (
                    _("coordinates ({x:.2f}, {y:.2f})").format(x=x, y=y)
                )

            elif location_name:
                loc = location_name.lower().strip()
                if loc not in KNOWN_LOCATIONS:
                    _publish_feedback(
                        self.connector,
                        _("[LARIC]: Error - The location '{loc}' was not found "
                          "in the semantic map of known locations. You can ask "
                          "for the known locations.")
                        .format(loc=loc)
                    )
                    return (
                        f"ERROR. The location '{loc}' was not found in "
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
                    _("[LARIC]: Error - No valid destination provided. You can "
                      "ask for the known locations.")
                )
                return (
                    "ERROR. No valid destination provided. "
                    "DO NOT CALL ANY OTHER TOOLS."
                )

            self.abort_event.clear()

            # 2. Announce intent
            _publish_feedback(
                self.connector,
                _("[LARIC]: Navigating to {dest}...").format(dest=dest_label)
            )

            try: 
                # 3. Send the NavigateToPose goal via RAI
                self._rai_nav_tool._run(x=x, y=y, z=0.0, yaw=yaw_rad)


            except Exception as exc:
                _publish_feedback(
                    self.connector,
                    _("[LARIC]: Error - The robot is not localised. Set its "
                      "initial pose on the map, then retry.")
                )
                return (
                    f"ERROR. Failed to send navigation goal: {exc}. "
                    "DO NOT CALL ANY OTHER TOOLS." 
                )

            # 4. Poll for the navigation result (max 3 minutes); StopTool can
            #    set abort_event from another thread to break out early.
            timeout = NAV_TIMEOUT_S
            elapsed = 0.0
            poll_interval = NAV_POLL_INTERVAL_S

            while elapsed < timeout:

                # Check whether StopTool has requested an abort
                if self.abort_event.is_set():
                    _publish_feedback(
                        self.connector,
                        _("[LARIC]: Navigation to {dest} cancelled.")
                        .format(dest=dest_label)
                    )
                    return (
                        "ACTION ABORTED. User stopped the navigation. "
                        "DO NOT CALL ANY OTHER TOOLS."
                    )

                # Query Nav2 for the current goal status
                try:
                    nav_result = str(self._rai_nav_result_tool._run()).strip()


                    if nav_result == str(GoalStatus.STATUS_SUCCEEDED):
                        _publish_feedback(
                            self.connector,
                            _("[LARIC]: Arrived at {dest}.")
                            .format(dest=dest_label)
                        )
                        return (
                            f"SUCCESS. Arrived at {dest_label}. "
                            "DO NOT CALL ANY OTHER TOOLS."
                        )

                    elif nav_result in [
                        str(GoalStatus.STATUS_ABORTED),
                        str(GoalStatus.STATUS_CANCELED)
                    ]:
                        _publish_feedback(
                            self.connector,
                            _("[LARIC]: Error - Could not reach {dest}. "
                              "Nav2 aborted.")
                            .format(dest=dest_label)
                        )
                        return (
                            "ACTION FAILED. Nav2 could not find or execute "
                            "a valid path. "
                            "DO NOT CALL ANY OTHER TOOLS."
                        )

                except Exception:
                    # Transient polling errors are non-fatal; continue waiting.
                    pass

                time.sleep(poll_interval)
                elapsed += poll_interval

            # 5. Timeout path
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - Navigation to {dest} timed out (3 min).")
                .format(dest=dest_label)
            )
            return (
                "ERROR: Navigation timeout (3 minutes). "
                "DO NOT CALL ANY OTHER TOOLS."
            )
        
        finally:
            self.motion_lock.release()


# ------------------------------------------------------------------------------

class GestureTool(BaseTool):
    """
    Performs a physical negation gesture via direct '/cmd_vel' publication.

    Attributes:
        name: LangChain tool identifier. Must match the name in 'SYSTEM_PROMPT'.
        description: Read by the LLM to decide when to invoke this tool.
        args_schema: Pydantic model validating input.
        connector: Active 'ROS2Connector' for '/cmd_vel' publication.
        abort_event: Shared 'threading.Event'; observed between oscillation
            steps so the gesture is interruptible.
        motion_lock: Shared 'threading.RLock'; acquired so a gesture cannot
            overlap a Nav2 motion (both would write '/cmd_vel').
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
    connector:   Any = None
    abort_event: Any = None
    motion_lock: Any = None

    def __init__(
        self,
        connector: ROS2Connector,
        abort_event: threading.Event,
        motion_lock: threading.Lock,
        **kwargs,
    ) -> None:
        """
        Initialises GestureTool with the ROS 2 connector and shared state.

        Args:
            connector: Active 'ROS2Connector' for '/cmd_vel' publication.
            abort_event: Threading event to signal abort requests.
            motion_lock: Threading lock to synchronize motion commands.
            **kwargs: Additional keyword arguments forwarded to 'BaseTool'.
        """
        super().__init__(**kwargs)
        self.connector = connector
        self.abort_event = abort_event
        self.motion_lock = motion_lock

    def _run(self, reason: str = "") -> str:
        """
        Executes the oscillation gesture synchronously.

        Args:
            reason: Human-readable explanation of why the request was rejected.

        Returns:
            A status string prefixed with 'SUCCESS' or 'ACTION ABORTED'.
        """
        if not self.motion_lock.acquire(blocking=False):
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - Another action is currently in progress. "
                  "Please wait for it to complete or use 'stop_robot' to "
                  "abort it before issuing a new command.")
            )
            return (
                "ACTION FAILED. Another action is currently in progress. "
                "DO NOT CALL ANY OTHER TOOLS."
            )

        try:
            self.abort_event.clear()
            # 1. Announce the gesture with the rejection reason
            _publish_feedback(
                self.connector,
                _("[LARIC]: (Negation gesture) Request not executable. "
                  "Reason: {reason}")
                .format(reason=reason)
            )

            # 2. Oscillation pattern: alternating left/right angular velocity
            angular_steps = [1.0, -1.0, 1.0, -1.0]
            for angular_z in angular_steps:
                if self.abort_event.is_set():
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

            if self.abort_event.is_set():
                _publish_feedback(
                    self.connector, _("[LARIC]: Gesture interrupted.")
                )
                return (
                    "ACTION ABORTED.  "
                    "DO NOT CALL ANY OTHER TOOLS."
                )

            return (
                f"SUCCESS. Negation gesture executed. Reason: {reason}. "
                "DO NOT CALL ANY OTHER TOOLS."
                )

        except Exception as exc:
            # The 'finally' block below releases the motion_lock; no extra
            # state cleanup is required here.
            _publish_feedback(
                self.connector,
                _("[LARIC]: Error - Couldn't execute the negation gesture.")
            )
            return (
                f"ERROR executing gesture: {exc}. "
                "DO NOT CALL ANY OTHER TOOLS." 
            )
        finally:
            self.motion_lock.release()


# ------------------------------------------------------------------------------

class StopTool(BaseTool):
    """
    Issues an immediate emergency stop for all active robot motion.

    Attributes:
        name: LangChain tool identifier. Must match the name in 'SYSTEM_PROMPT'.
        description: Read by the LLM to decide when to invoke this tool.
        args_schema: Pydantic model validating input.
        connector: Active 'ROS2Connector' instance.
        abort_event: Shared 'threading.Event'; set here to signal every
            polling motion tool to terminate immediately.
        _cancel_nav_tool: RAI 'CancelNavigateToPoseTool' used to cancel any
            active Nav2 goal.
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

    connector:       Any = None
    abort_event:     Any = None
    _cancel_nav_tool: Any = None

    def __init__(
        self,
        connector: ROS2Connector,
        abort_event: threading.Event,
        **kwargs,
    ) -> None:
        """
        Initialises StopTool and its internal Nav2 cancellation helper.

        Args:
            connector: Active 'ROS2Connector'.
            abort_event: Threading event to signal abort.
            **kwargs: Additional keyword arguments forwarded to 'BaseTool'.
        """
        super().__init__(**kwargs)
        self.connector = connector
        self.abort_event = abort_event
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
        self.abort_event.set()

        # 2. Cancel any active Nav2 goal.
        try:
            self._cancel_nav_tool._run()
        except Exception:
            pass  # Non-fatal if no active goal exists

        # 3. Publish zero velocity to /cmd_vel as an immediate fallback;
        #    Nav2's cancel (above) is the authoritative stop.
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
            # Non-fatal: abort_event and Nav2 cancel are the primary mechanisms
            pass

        _publish_feedback(
            self.connector,
            _("[LARIC]: Stop activated. All motion halted.")
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
    speed_rad_s: float = SPIN_SPEED_RAD_S
) -> float:
    """
    Computes a conservative 'time_allowance' for the Nav2 Spin action.

    Args:
        target_yaw_rad: Target rotation magnitude in radians (sign ignored).
        speed_rad_s: Expected rotation speed in rad/s. Defaults to
            config.SPIN_SPEED_RAD_S.

    Returns:
        Estimated duration in seconds, with a SPIN_ALLOWANCE_BUFFER_S safety
        buffer and a SPIN_ALLOWANCE_MIN_S floor.
    """
    kinematic_estimate = abs(target_yaw_rad) / max(speed_rad_s, 0.01)
    return max(kinematic_estimate + SPIN_ALLOWANCE_BUFFER_S,
               SPIN_ALLOWANCE_MIN_S)


def _compute_move_allowance(distance_m: float, speed_m_s: float) -> float:
    """
    Computes a conservative 'time_allowance' for DriveOnHeading / BackUp.

    Args:
        distance_m: Target distance magnitude in meters.
        speed_m_s: Commanded speed in m/s. Clamped to a minimum of 0.01.

    Returns:
        Estimated duration in seconds with a MOVE_ALLOWANCE_BUFFER_S buffer.
    """
    kinematic_estimate = distance_m / max(speed_m_s, 0.01)
    return kinematic_estimate + MOVE_ALLOWANCE_BUFFER_S


# ==============================================================================
# ENTRY POINT
# ==============================================================================

@ROS2Context()
def main() -> None:
    """
    Main entry point for the LARIC robot agent system.
    """
    # 1. Initialise ROS 2 connector
    ros_connector = ROS2Connector()
    # Brief pause to allow internal publishers to register before the first
    # feedback message is sent.
    time.sleep(3.0)

    _publish_feedback(ros_connector, _("[SYSTEM]: Starting LARIC Agent..."))

    # 2. Initialise the shared concurrency primitives. StopTool never holds
    #    motion_lock so it can always interrupt an in-progress motion.
    abort_event     = threading.Event()
    motion_lock     = threading.RLock()
    localised_event = threading.Event()
    _localised_ready_at = time.time() + INITIALPOSE_GRACE_S

    # 3. Instantiate LangChain tools (dependency injection).
    _publish_feedback(ros_connector, _("[SYSTEM]: Loading Nav2 tools..."))
    tools = [
        SpinTool(
            connector=ros_connector,
            abort_event=abort_event,
            motion_lock=motion_lock,
            localised_event=localised_event,
        ),
        MoveTool(
            connector=ros_connector,
            abort_event=abort_event,
            motion_lock=motion_lock,
            localised_event=localised_event,
        ),
        SequenceTool(
            connector=ros_connector,
            abort_event=abort_event,
            motion_lock=motion_lock,
            localised_event=localised_event,
        ),
        NavigationTool(
            connector=ros_connector,
            abort_event=abort_event,
            motion_lock=motion_lock,
            localised_event=localised_event,
        ),
        GestureTool(
            connector=ros_connector,
            abort_event=abort_event,
            motion_lock=motion_lock,
        ),
        StopTool(
            connector=ros_connector,
            abort_event=abort_event,
        ),
    ]

    # 4. Connect to the LLM backend
    _publish_feedback(
        ros_connector,
        _("[SYSTEM]: Mode - is_from_lab={lab} ({llm}), "
          "is_real_robot={real} ({robot}).").format(
            lab=is_from_lab,
            llm=("Ollama lab" if is_from_lab else "Groq cloud"),
            real=is_real_robot,
            robot=("real ROSbot" if is_real_robot else "simulation"),
        )
    )
    _publish_feedback(ros_connector, _("[SYSTEM]: Connecting to LLM..."))
    if is_from_lab:
        llm = ChatOllama(temperature=0, model=MODEL, base_url=URL)
    else:
        if not GROQ_API_KEY:
            _publish_feedback(
                ros_connector,
                _("[SYSTEM]: ERROR - Groq selected (is_from_lab=False) but "
                  "GROQ_API_KEY is not set. Export it, or switch to the lab "
                  "Ollama (is_from_lab=True).")
            )
        llm = ChatGroq(
            temperature=0, 
            model=GROQ_MODEL_70B, 
            api_key=GROQ_API_KEY
        )

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

        Args:
            msg: Incoming 'ROS2Message' wrapping a 'std_msgs/String' payload.
        """
        try:
            user_text = msg.payload.data.strip()
        except AttributeError:
            return  # Malformed message; discard silently

        if not user_text:
            return

        _publish_feedback(
            ros_connector,
            _("[LARIC]: Heard: '{text}'").format(text=user_text)
        )
        _publish_feedback(ros_connector, _("[LARIC]: Thinking..."))

        def think() -> None:
            """
            Runs the AgentExecutor in a daemon thread.
            """
            try:
                result = agent_executor.invoke({"input": user_text})
                response = result.get("output", "").strip()

                # Check for internal returns that should not be published as 
                # feedback
                is_internal_return = (
                    "DO NOT CALL ANY OTHER TOOLS" in response 
                    or response.startswith((
                        "SUCCESS.", "ACTION FAILED.", "ACTION ABORTED.", 
                        "ERROR:", "ERROR."
                    ))
                )

                if response and not is_internal_return:
                    formatted_response = response.replace("\n", "<br>")
                    _publish_feedback(
                        ros_connector, 
                        f"[LARIC]: {formatted_response}"
                    )
            except Exception as exc:
                _publish_feedback(
                    ros_connector,
                    _("[LARIC]: ERROR - {exc}").format(exc=exc)
                )

        # Daemon threads are automatically joined when the main process exits.
        threading.Thread(target=think, daemon=True).start()

    try:
        ros_connector.register_callback(
            source     = "/from_human",
            topic_name = "/from_human",
            msg_type   = "std_msgs/msg/String",
            callback   = on_human_input
        )
    except Exception as exc:
        _publish_feedback(
            ros_connector,
            _("[SYSTEM]: Warning - Could not subscribe to /from_human: {exc}")
            .format(exc=exc)
        )

    # 7. Initial pose listener for localisation detection
    def on_initial_pose(msg: ROS2Message) -> None:
        """
        Detects when the operator sets the initial pose.

        Args:
            msg: Incoming 'ROS2Message' wrapping a
                 'geometry_msgs/PoseWithCovarianceStamped' payload. The pose
                 comes from RViz '2D Pose Estimate' in simulation or the
                 Foxglove UI on the real robot.
        """
        if time.time() < _localised_ready_at:
            return
        localised_event.set()
        _publish_feedback(
            ros_connector,
            _("[LARIC]: Robot localised. Ready for commands.")
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
            _("[SYSTEM]: Warning - Could not subscribe to /initialpose: {exc}")
            .format(exc=exc)
        )

    # 8. Hardware Emergency Stop Listener 
    def on_emergency_stop(msg: ROS2Message) -> None:
        """
        Triggers instantly upon receiving the E-Stop signal from the UI.
        Bypasses the LLM entirely to guarantee deterministic braking.

        Args:
            msg: Incoming 'ROS2Message' wrapping a 'std_msgs/Bool' payload.
                 If the boolean value is True, the emergency stop is applied.
        """
        if msg.payload.data:
            _publish_feedback(
                ros_connector,
                _("[LARIC]: EMERGENCY STOP ACTIVATED.")
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
            _("[SYSTEM]: Warning - Could not subscribe to "
              "/emergency_stop: {exc}")
            .format(exc=exc)
        )

    # ----- /language_changed listener -----
    # Propagates the HMI's language change to this process's gettext catalog
    # (separate processes, so this topic is the only sync path).
    def on_language_changed(msg: ROS2Message) -> None:
        """
        Updates the agent's translator when the HMI changes language.

        Args:
            msg: Incoming 'ROS2Message' wrapping a 'std_msgs/String' whose
                 'data' field is the new language code (e.g. "en", "es").
        """
        try:
            new_lang = msg.payload.data.strip()
        except AttributeError:
            return
        if not new_lang:
            return
        set_language(new_lang)

    try:
        ros_connector.register_callback(
            source="/language_changed",
            topic_name="/language_changed",
            msg_type="std_msgs/msg/String",
            callback=on_language_changed
        )
    except Exception as exc:
        _publish_feedback(
            ros_connector,
            _("[SYSTEM]: Warning - Could not subscribe to "
              "/language_changed: {exc}")
            .format(exc=exc)
        )

    _publish_feedback(
        ros_connector,
        _("[LARIC]: Agent ready. Listening on /from_human.")
    )
    _publish_feedback(
        ros_connector,
        _("[LARIC]: IMPORTANT - Set the robot's initial pose on the map "
          "before sending commands.")
    )

    # 9. Keep-alive loop
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        _publish_feedback(
            ros_connector,
            _("[SYSTEM]: Shutting down LARIC Agent...")
        )


if __name__ == "__main__":
    main()