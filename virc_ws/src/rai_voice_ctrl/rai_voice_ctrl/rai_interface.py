"""
VIRC System - Human-Machine Interface (HMI)
Copyright (c) 2026 VIRC. All rights reserved.

Implements the graphical user interface for the VIRC system, combining a
ROS 2 node and a PyQt5 widget into a single object. Provides two input
modalities (Push-to-Talk voice and keyboard text), a real-time colour-coded
feedback log, and a direct hardware-level emergency stop button.

Architecture note:
    This module inherits from both 'rclpy.node.Node' and 'PyQt5.QWidget'.
    Because each framework owns a separate event loop, the ROS 2 spin cycle is
    integrated into Qt's event loop via a 'QTimer' that calls
    'rclpy.spin_once()' every 20 ms. This keeps the UI responsive while
    still processing incoming ROS 2 messages.

    All GUI mutations that originate from background threads (audio processing,
    ROS 2 callbacks) are routed through a 'pyqtSignal' to guarantee they
    execute exclusively on the Qt main thread, preventing data races on the
    underlying widget state.
"""

import io
import os
import queue
import sys
import threading
import subprocess
from contextlib import contextmanager
from typing import Generator

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import speech_recognition as sr

# -- ROS 2 -------------------------------------------------------------------
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String

# -- PyQt5 -------------------------------------------------------------------
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QPushButton,
    QShortcut,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# -- Project configuration -------------------------------------------------------
from config import SHUTDOWN_SCRIPT_PATH


# ==============================================================================
# CONSTANTS
# ==============================================================================

SAMPLE_RATE: int = 16000   # Hz — Google STT performs best at 16 kHz
CHANNELS: int    = 1       # Mono capture is sufficient for speech recognition

# Minimum recorded audio duration in seconds. Recordings shorter than this
# threshold are discarded to prevent accidental spacebar taps from triggering
# the Speech-to-Text API and incurring unnecessary latency.
MIN_AUDIO_DURATION_S: float = 0.2

# Debounce interval in milliseconds applied to the Push-to-Talk key release
# event. Prevents rapid key bounces from prematurely stopping an active
# recording session.
PTT_DEBOUNCE_MS: int = 150

# ROS 2 spin timer period in milliseconds. At 20 ms (50 Hz) the UI remains
# responsive while still processing ROS 2 messages promptly.
ROS_SPIN_INTERVAL_MS: int = 20


# ==============================================================================
# UTILITY
# ==============================================================================

@contextmanager
def _suppress_stderr() -> Generator[None, None, None]:
    """
    Context manager that suppresses all output written to 'stderr'.

    Yields:
        None

    Note:
        Used exclusively to silence ALSA / PortAudio driver warnings emitted
        by 'sounddevice' at stream initialisation. These warnings are
        irrelevant to application logic and would pollute the terminal during
        development. The suppression is implemented at the file-descriptor level
        ('os.dup2') rather than via 'sys.stderr' reassignment so that
        low-level C library writes are also captured.
    """
    try:
        with open(os.devnull, "w") as devnull:
            # 1. Save a copy of the original stderr file descriptor
            original_stderr_fd = os.dup(sys.stderr.fileno())
            # 2. Redirect stderr to /dev/null
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            try:
                yield
            finally:
                # 3. Restore the original stderr unconditionally
                os.dup2(original_stderr_fd, sys.stderr.fileno())
    except Exception:
        # If the redirection itself fails (e.g., in certain container
        # environments), fall through silently so the caller is unaffected.
        yield


# ==============================================================================
# HMI NODE
# ==============================================================================

class InterfaceNode(Node, QWidget):
    """
    Combined ROS 2 node and PyQt5 widget implementing the VIRC HMI.

    Handles Push-to-Talk voice capture, Google Speech-to-Text transcription,
    keyboard text input, and real-time display of robot feedback messages.

    Attributes:
        log_signal (pyqtSignal): Qt signal used to safely append HTML text to
            the feedback log from any thread. Accepts '(text: str, color: str)'.
        pub_human: ROS 2 publisher for 'std_msgs/String' on '/from_human'.
        pub_emergency_stop: ROS 2 publisher for 'std_msgs/Bool' on 
            '/emergency_stop'.
        pub_cmd_vel: ROS 2 publisher for 'geometry_msgs/Twist' on '/cmd_vel'.
        sub_feedback: ROS 2 subscription to 'std_msgs/String' on
            '/robot_feedback'.
        audio_queue (queue.Queue): Thread-safe buffer that accumulates raw
            audio chunks from the 'sounddevice' input stream callback.
        is_recording (bool): 'True' while a PTT recording session is active.
        recognizer (sr.Recognizer): Google Speech Recognition engine instance.
        stream: Active 'sounddevice.InputStream', or 'None' when idle.
        stop_timer (QTimer): Single-shot debounce timer for the PTT key release.
        ros_timer (QTimer): Periodic timer that drives the ROS 2 spin loop.
        label (QLabel): Status indicator widget showing the current PTT state.
        log (QTextEdit): Read-only feedback log widget.
        text_input (QLineEdit): Manual text command input field.
        emergency_btn (QPushButton): Hardware-level emergency stop button.
    """

    # Class-level signal declaration required by the Qt meta-object system.
    # Signals must be declared at class scope before __init__ is called.
    log_signal: pyqtSignal = pyqtSignal(str, str)  # (html_text, css_colour)

    def __init__(self) -> None:
        """Initialises the ROS 2 node, Qt widget, audio subsystem, and timers."""
        # 1. Initialise both parent classes explicitly.
        #    Order matters: Node.__init__ must precede QWidget.__init__ to
        #    ensure the ROS 2 node name is registered before Qt sets up the
        #    widget hierarchy.
        Node.__init__(self, "interfaz_hmi_rai")
        QWidget.__init__(self)

        # 2. Connect the thread-safe log signal to its GUI slot.
        #    This connection guarantees that update_log_slot() always executes
        #    on the Qt main thread, regardless of which thread emits the signal.
        self.log_signal.connect(self.update_log_slot)

        # 3. Create ROS 2 publisher and subscriber
        self.pub_human = self.create_publisher(String, "/from_human", 10)

        # Direct /cmd_vel and /emergency_stop publishers for hardware halt
        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_emergency_stop = self.create_publisher(
            Bool, "/emergency_stop", 10
        )

        self.sub_feedback = self.create_subscription(
            String, "/robot_feedback", self._feedback_callback, 10
        )

        # 4. Initialise audio state
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_recording: bool = False
        self.recognizer: sr.Recognizer = sr.Recognizer()
        self.stream = None

        # 5. Configure the PTT debounce timer.
        #    Single-shot: fires once after PTT_DEBOUNCE_MS milliseconds and then
        #    stops. If the spacebar is re-pressed before the timer fires, the
        #    timer is cancelled in keyPressEvent, preventing a false stop.
        self.stop_timer = QTimer()
        self.stop_timer.setSingleShot(True)
        self.stop_timer.setInterval(PTT_DEBOUNCE_MS)
        self.stop_timer.timeout.connect(self._stop_recording)

        # 6. Configure the ROS 2 integration timer.
        #    Calls spin_once() at ROS_SPIN_INTERVAL_MS intervals, integrating
        #    the ROS 2 executor into Qt's event loop without blocking the GUI.
        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self._spin_ros)
        self.ros_timer.start(ROS_SPIN_INTERVAL_MS)

        # 7. Build the graphical layout
        self._init_ui()

    # --------------------------------------------------------------------------
    # UI CONSTRUCTION
    # --------------------------------------------------------------------------

    def _init_ui(self) -> None:
        """
        Constructs and configures the widget layout.

        Builds three stacked components:
        - A status label showing the current PTT state.
        - A read-only HTML log for feedback messages.
        - A single-line text input for manual command entry.
        - A prominent emergency stop button.
        """
        self.setWindowTitle("VIRC - Voice & Text Interface")
        self.setGeometry(1200, 100, 450, 660)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        layout = QVBoxLayout()

        # -- Status label (PTT indicator) ------------------------------------
        self.label = QLabel("HOLD [SPACE] TO SPEAK")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            "font-size: 14px;"
            "background: #333333;"
            "color: white;"
            "padding: 20px;"
            "border-radius: 10px;"
            "font-weight: bold;"
        )
        layout.addWidget(self.label)

        # -- Feedback log (read-only HTML console) ---------------------------
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        # Qt.NoFocus prevents the log widget from stealing keyboard focus
        # away from the main window, which would break spacebar PTT detection.
        self.log.setFocusPolicy(Qt.NoFocus)
        self.log.setStyleSheet(
            "background: #1e1e1e;"
            "color: #d4d4d4;"
            "font-family: 'Consolas', 'Monaco', monospace;"
            "border-radius: 5px;"
            "padding: 10px;"
            "font-size: 12px;"
        )
        layout.addWidget(self.log)

        # -- Text command input ----------------------------------------------
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type a command and press Enter...")
        self.text_input.setStyleSheet("padding: 10px; border-radius: 5px;")
        self.text_input.returnPressed.connect(self._send_text_command)
        layout.addWidget(self.text_input)

        # -- Emergency Stop Button -------------------------------------------
        self.emergency_btn = QPushButton("EMERGENCY STOP (Ctrl+E)")
        self.emergency_btn.setStyleSheet(
            "QPushButton {"
            "  font-size: 13px; background: #D32F2F; color: white;"
            "  padding: 10px; border-radius: 6px; font-weight: bold;"
            "  letter-spacing: 1px; min-height: 40px;"
            "}"
            "QPushButton:hover { background: #B71C1C; }"
            "QPushButton:pressed { background: #FF5252; }"
        )
        self.emergency_btn.clicked.connect(self._emergency_stop)
        layout.addWidget(self.emergency_btn)

        # Register a global shortcut for Ctrl+E to trigger the emergency stop
        self.shortcut_estop = QShortcut(QKeySequence("Ctrl+E"), self)
        self.shortcut_estop.activated.connect(self._emergency_stop)

        # Qt.StrongFocus ensures the main window captures key events even when
        # no child widget holds focus (required for spacebar PTT to work after
        # the application starts).
        self.setFocusPolicy(Qt.StrongFocus)
        self.setLayout(layout)
        self.setFocus()

    # --------------------------------------------------------------------------
    # WINDOW CLOSE EVENT
    # --------------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """
        Intercepts the window close event (the 'X' button).
        Triggers the global shutdown script using a dynamic, portable path
        to gracefully clean up all ROS 2 nodes, simulators, and background processes 
        before the UI terminates.
        """
        self.add_log(
            "[SYSTEM] Shutting down entire VIRC system...", "#FF4444"
        )
        
        try:
            # Verify the script exists at the dynamically resolved path
            if os.path.exists(SHUTDOWN_SCRIPT_PATH):
                # Launch the shutdown script in a detached session
                subprocess.Popen(
                    ["bash", SHUTDOWN_SCRIPT_PATH], start_new_session=True
                )
            else:
                print(f"DEBUG: Script not found at {SHUTDOWN_SCRIPT_PATH}")
                self.add_log("[ERROR] Shutdown script not found.", "red")
        except Exception as e:
            print(f"Error launching shutdown script: {e}")

        # Accept the event to allow the window to close visually
        event.accept()

    # --------------------------------------------------------------------------
    # ROS 2 INTEGRATION
    # --------------------------------------------------------------------------

    def _spin_ros(self) -> None:
        """
        Executes one iteration of the ROS 2 event loop.

        Called periodically by 'ros_timer'. 'timeout_sec=0' means the call
        returns immediately if no callbacks are pending, ensuring the Qt event
        loop is never blocked.
        """
        rclpy.spin_once(self, timeout_sec=0)

    def _feedback_callback(self, msg: String) -> None:
        """
        ROS 2 subscription callback for '/robot_feedback' messages.

        Args:
            msg: Incoming 'std_msgs/String' message containing the agent's
                feedback text.

        Note:
            This callback executes on whichever thread 'rclpy.spin_once()'
            is called from (the Qt main thread, via 'ros_timer'). It is
            therefore safe to call 'add_log' directly, but the signal-based
            route is used for consistency with callbacks that may arrive from
            other threads in future.
        """
        self.add_log(f"{msg.data}", "#4FC3F7")

    # --------------------------------------------------------------------------
    # TEXT INPUT
    # --------------------------------------------------------------------------

    def _send_text_command(self) -> None:
        """
        Publishes the text field content to '/from_human' and resets focus.

        Reads and trims the text input field, publishes a lower-cased version
        as a ROS 2 'String' message, clears the field, and returns keyboard
        focus to the main window so that spacebar PTT resumes immediately.
        """
        text = self.text_input.text().strip()
        if not text:
            return

        # 1. Log the command locally before publishing
        self.add_log(f"[USER - TEXT]: {text}", "#8BC34A")

        # 2. Publish to the agent's input topic
        msg = String()
        msg.data = text.lower()
        self.pub_human.publish(msg)

        # 3. Clear the field and return focus to the main window
        self.text_input.clear()
        self.setFocus()

    # --------------------------------------------------------------------------
    # EMERGENCY STOP
    # --------------------------------------------------------------------------

    def _emergency_stop(self) -> None:
        """
        Immediately halts the robot without routing through the AI agent.

        The execution order is designed to be as fast as possible:
          Step 1 — Publishes a zero-velocity Twist directly to /cmd_vel.
                   This physically stops the motors before anything else.
          Step 2 — Publishes True to /emergency_stop.
                   agent_logic.py can subscribe to this topic and immediately
                   set abort_flag=True, terminating all running motion threads.
          Step 3 — Displays a red confirmation in the HMI log.
        """
        # Step 1: Hard-stop — direct /cmd_vel publish (Twist defaults to 0.0)
        self.pub_cmd_vel.publish(Twist())

        # Step 2: Signal agent_logic to kill all running motion threads
        flag = Bool()
        flag.data = True
        self.pub_emergency_stop.publish(flag)

        # Step 3: Inform the operator
        self.add_log("[EMERGENCY STOP] Robot halted by user.", "#FF4444")
        
        # Return focus to main window so PTT continues to work
        self.setFocus()

    # --------------------------------------------------------------------------
    # THREAD-SAFE LOG
    # --------------------------------------------------------------------------

    def add_log(self, text: str, color: str = "#d4d4d4") -> None:
        """
        Queues a log entry for display on the Qt main thread.

        Safe to call from any thread. Instead of touching the 'QTextEdit'
        widget directly (which would cause a data race), this method emits
        'log_signal', which Qt delivers to 'update_log_slot()' on the
        main thread via the event queue.

        Args:
            text: Plaintext or HTML content to display in the log.
            color: CSS colour string applied to the message text.
                   Defaults to light grey ('#d4d4d4').
        """
        self.log_signal.emit(text, color)

    def update_log_slot(self, text: str, color: str) -> None:
        """Appends a colour-coded HTML entry to the feedback log widget.

        This slot is connected to 'log_signal' and executes exclusively on
        the Qt main thread, satisfying Qt's thread-affinity requirement for
        widget mutations.

        Args:
            text: Message content to display.
            color: CSS colour string for the message text.
        """
        html = f"<div style='color: {color}; margin-bottom: 5px;'>{text}</div>"
        self.log.append(html)

    # --------------------------------------------------------------------------
    # KEYBOARD EVENTS (PTT)
    # --------------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        """
        Handles key press events to start a PTT recording session.

        Ignores spacebar events when the text input field has focus so that
        the user can type spaces in commands without triggering recording.
        Cancels any pending debounce timer if the key is pressed again before
        the timer fires, preventing a false recording-stop.

        Args:
            event (QKeyEvent): The key press event delivered by Qt.
        """
        # Delegate all events to the default handler when the text box is active
        if self.text_input.hasFocus():
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key_Space:
            # isAutoRepeat() is True when the OS generates repeated keydown
            # events for a held key. Ignore these to avoid re-triggering
            # the recording start on every repeat cycle.
            if not event.isAutoRepeat():
                # Cancel a pending stop timer (user pressed space again before
                # the debounce window closed).
                if self.stop_timer.isActive():
                    self.stop_timer.stop()

                if not self.is_recording:
                    self._start_recording()

    def keyReleaseEvent(self, event) -> None:
        """
        Handles key release events to schedule the end of a PTT session.

        Starts the debounce timer on spacebar release. The actual recording
        stop only fires after PTT_DEBOUNCE_MS milliseconds, during which a
        second press cancels the timer (handled in 'keyPressEvent').

        Args:
            event (QKeyEvent): The key release event delivered by Qt.
        """
        if self.text_input.hasFocus():
            return

        if event.key() == Qt.Key_Space:
            if not event.isAutoRepeat() and self.is_recording:
                # Start the debounce timer rather than stopping immediately,
                # giving the user a PTT_DEBOUNCE_MS window to re-press if
                # the key release was accidental.
                self.stop_timer.start()

    # --------------------------------------------------------------------------
    # AUDIO RECORDING
    # --------------------------------------------------------------------------

    def _start_recording(self) -> None:
        """
        Opens the microphone stream and updates the UI to the recording state.

        Clears any residual audio data from previous sessions, opens a
        'sounddevice.InputStream', and sets 'is_recording = True'.
        If the microphone cannot be opened, logs the error and returns to
        the idle UI state without raising an exception.
        """
        # 1. Set recording flag and clear the audio buffer
        self.is_recording = True
        self.audio_queue.queue.clear()

        # 2. Update the status label to the active-recording style
        self.label.setText("[RECORDING IN PROGRESS...]")
        self.label.setStyleSheet(
            "font-size: 14px;"
            "background: #4CAF50;"
            "color: white;"
            "padding: 20px;"
            "border-radius: 10px;"
            "font-weight: bold;"
        )

        # 3. Open the sounddevice input stream.
        #    stderr suppression hides ALSA/PortAudio driver warnings that are
        #    unrelated to application logic but clutter the terminal.
        try:
            with _suppress_stderr():
                self.stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    callback=self._audio_callback,
                )
                self.stream.start()
        except Exception as exc:
            self.add_log(f"[HARDWARE ERROR] Microphone: {exc}", "red")
            self.is_recording = False
            self._reset_ui()

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time, status
    ) -> None:
        """
        Sounddevice stream callback — enqueues raw audio chunks.

        Invoked by the 'sounddevice' library on a dedicated audio thread for
        every captured frame block. The data is copied and placed on
        'audio_queue' for consumption by '_process_audio()'.

        Args:
            indata: NumPy array of shape '(frames, channels)' containing the
                captured PCM samples in float32 format.
            frames: Number of frames in this chunk.
            time: 'sounddevice.CallbackFlags' timing info (unused).
            status: 'sounddevice.CallbackFlags' status flags (unused).

        Note:
            'indata.copy()' is mandatory. The 'indata' buffer is owned by
            'sounddevice' and will be overwritten on the next callback
            invocation. Without the copy, the queue would hold stale references
            to the same memory region.
        """
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def _stop_recording(self) -> None:
        """
        Stops the microphone stream and triggers background audio processing.

        Called by the debounce timer after 'PTT_DEBOUNCE_MS' ms. Closes the
        'sounddevice' stream, updates the UI to the processing state, and
        launches '_process_audio()' on a daemon thread to avoid blocking the
        Qt event loop during the potentially slow Google STT API call.
        """
        if not self.is_recording:
            return

        # 1. Clear the recording flag before touching the stream to prevent
        #    the audio callback from enqueuing more data after we close it.
        self.is_recording = False

        # 2. Close the sounddevice stream
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass  # Stream may already be in an error state; proceed regardless

        # 3. Update UI to the processing state
        self.label.setText("[PROCESSING AUDIO...]")
        self.label.setStyleSheet(
            "font-size: 14px;"
            "background: #FFC107;"
            "color: black;"
            "padding: 20px;"
            "border-radius: 10px;"
            "font-weight: bold;"
        )

        # 4. Offload STT work to a background thread.
        #    The Google STT API call is network-bound and may take 1–3 seconds.
        #    Running it on the Qt main thread would freeze the entire UI.
        threading.Thread(target=self._process_audio, daemon=True).start()

    # --------------------------------------------------------------------------
    # AUDIO PROCESSING & STT
    # --------------------------------------------------------------------------

    def _process_audio(self) -> None:
        """
        Transcribes the recorded audio and publishes the result to ROS 2.

        Runs on a daemon thread started by '_stop_recording()'. Drains
        'audio_queue', validates the recording duration, converts to WAV,
        calls the Google Speech-to-Text API, and publishes the transcription
        to '/from_human'. Resets the UI to idle in the 'finally' block
        regardless of success or failure.

        Raises:
            No exceptions propagate out of this method; all errors are caught
            and reported via 'add_log()'.
        """
        try:
            # 1. Drain the audio queue into a single contiguous array
            chunks = []
            while not self.audio_queue.empty():
                chunks.append(self.audio_queue.get())

            if not chunks:
                self.add_log(
                    "[WARNING] Audio buffer is empty. Nothing to transcribe.", 
                    "orange"
                )
                return

            full_audio = np.concatenate(chunks, axis=0)

            # 2. Discard recordings shorter than MIN_AUDIO_DURATION_S.
            #    Short bursts are almost certainly accidental spacebar taps
            #    rather than genuine voice commands. Discarding them avoids
            #    unnecessary STT API calls and the latency they introduce.
            duration_s = len(full_audio) / SAMPLE_RATE
            if duration_s < MIN_AUDIO_DURATION_S:
                self.add_log(
                    f"[WARNING] Recording too short ({duration_s:.2f}s &lt; "
                    f"{MIN_AUDIO_DURATION_S}s). Discarded.",
                    "orange",
                )
                return

            # 3. Convert float32 PCM to int16 WAV in an in-memory buffer.
            #    The SpeechRecognition library requires WAV format input.
            #    Scaling by 32767 maps the float [-1.0, 1.0] range to int16.
            audio_int16 = (full_audio * 32767).astype(np.int16)
            wav_buffer = io.BytesIO()
            wav.write(wav_buffer, SAMPLE_RATE, audio_int16)
            wav_buffer.seek(0)

            # 4. Feed the WAV buffer to the SpeechRecognition engine
            with sr.AudioFile(wav_buffer) as source:
                audio_data = self.recognizer.record(source)

            # 5. Call the Google STT API.
            #    language="es-ES" targets Castilian Spanish, which gives
            #    significantly better accuracy than the default English model
            #    for this application's target user base.
            text = self.recognizer.recognize_google(
                audio_data, language="es-ES"
            )

            self.add_log(f"[USER - VOICE]: {text}", "#8BC34A")

            # 6. Publish the transcription to the agent's input topic
            msg = String()
            msg.data = text.lower()
            self.pub_human.publish(msg)

        except sr.UnknownValueError:
            # Raised when the audio contains only silence or is unintelligible.
            self.add_log(
                "[WARNING] No speech detected "
                "(silence or unintelligible audio)."
                , "orange")

        except sr.RequestError as exc:
            # Raised when the Google STT API is unreachable or returns an error.
            self.add_log(f"[ERROR] Google STT API request failed: {exc}", "red")

        except Exception as exc:
            self.add_log(
                f"[ERROR] Unexpected error during audio processing: {exc}", 
                "red"
            )

        finally:
            # 7. Always reset the UI to idle, even if an exception occurred.
            #    QTimer.singleShot() schedules the call on the Qt main thread,
            #    which is required because this method runs on a daemon thread.
            QTimer.singleShot(0, self._reset_ui)

    # --------------------------------------------------------------------------
    # UI STATE HELPERS
    # --------------------------------------------------------------------------

    def _reset_ui(self) -> None:
        """
        Resets the status label to the idle (ready-to-record) state.
        """
        self.label.setText("HOLD [SPACE] TO SPEAK")
        self.label.setStyleSheet(
            "font-size: 14px;"
            "background: #333333;"
            "color: white;"
            "padding: 20px;"
            "border-radius: 10px;"
            "font-weight: bold;"
        )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main() -> None:
    """Application entry point.

    Initialises 'rclpy', creates the Qt application and the combined
    ROS 2 / PyQt5 node, runs the Qt event loop, and guarantees clean
    shutdown of both frameworks on exit.
    """
    # 1. Initialise rclpy before creating the Node
    rclpy.init()

    # 2. Create the Qt application object (must exist before any QWidget)
    app = QApplication(sys.argv)

    # 3. Instantiate the combined ROS 2 node / Qt widget
    node = InterfaceNode()
    node.show()

    # 4. Enter the Qt event loop.
    #    This call blocks until the user closes the window.
    try:
        sys.exit(app.exec_())
    finally:
        # 5. Destroy the ROS 2 node and shut down rclpy cleanly.
        #    Placed in finally to guarantee execution even on abnormal exit.
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
