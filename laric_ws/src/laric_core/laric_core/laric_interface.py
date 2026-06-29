"""
LARIC System - Human-Machine Interface (HMI)
Copyright (c) 2026 LARIC. All rights reserved.

Implements the graphical user interface for the LARIC system, combining a
ROS 2 node and a PyQt5 widget into a single object. Provides two input
modalities (Push-to-Talk voice and keyboard text), a real-time colour-coded
feedback log, spoken feedback (text-to-speech of the short phrases published
on '/laric_voice'), a language selector, and a direct hardware-level
emergency stop button.

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

Internationalisation:
    Every operator-visible string goes through 'i18n._()'. The dropdown
    drives 'i18n.set_language(...)' locally AND publishes the new language
    code on the '/language_changed' ROS 2 topic so that 'agent_logic.py'
    (a separate process) can update its translator too.
"""

import io
import os
import queue
import shutil
import sys
import tempfile
import threading
import subprocess
from contextlib import contextmanager
from typing import Generator, Optional

# Force the X11 (xcb) backend before any PyQt5 import: native Wayland ignores
# stay-on-top hints and setGeometry, while XWayland behaves like X11.
os.environ["QT_QPA_PLATFORM"] = "xcb"

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import speech_recognition as sr

# -- ROS 2 -------------------------------------------------------------------
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Bool, String

# -- PyQt5 -------------------------------------------------------------------
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QShortcut,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# -- Project configuration & i18n --------------------------------------------
from config import HMI_GEOMETRY, SHUTDOWN_SCRIPT_PATH, is_real_robot
from i18n import _, set_language


# ==============================================================================
# CONSTANTS
# ==============================================================================

SAMPLE_RATE: int = 16000   # Hz — Google STT performs best at 16 kHz
CHANNELS: int    = 1       # Mono capture is sufficient for speech recognition

# Minimum recording length (s); shorter clips are discarded as accidental taps.
MIN_AUDIO_DURATION_S: float = 0.2

# Debounce (ms) on PTT key release, so key bounce does not cut a recording.
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
# VOICE OUTPUT
# ==============================================================================

class VoiceSpeaker:
    """Text-to-speech playback for spoken system feedback."""

    # MP3-capable command-line players, tried in order of preference.
    _PLAYERS = (
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
        ["mpg123", "-q"],
        ["mpg321", "-q"],
        ["cvlc", "--play-and-exit", "--intf", "dummy"],
    )

    # Neutral voice per language (run 'edge-tts --list-voices to browse more).
    _VOICES = {
        "en": "en-US-JennyNeural",
        "es": "es-ES-ElviraNeural",
    }

    # Speech rate relative to the voice default.
    _RATE = "+25%"

    def __init__(self) -> None:
        """Detects an available player."""
        self._player = self._detect_player()
        # Handle to the currently playing process, so stop() can kill it.
        self._proc = None
        self._proc_lock = threading.Lock()

    @staticmethod
    def _detect_player():
        """Returns the first available player command, or None."""
        for cmd in VoiceSpeaker._PLAYERS:
            if shutil.which(cmd[0]):
                return cmd
        return None
    
    def synthesize(self, text: str, lang: str = "en") -> Optional[str]:
        """Synthesises 'text' to an MP3 file and returns its path (or None)."""
        if not text:
            return None
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ) as handle:
                tmp_path = handle.name
            voice = self._VOICES.get(lang, self._VOICES["en"])
            subprocess.run(
                ["edge-tts", "--voice", voice, "--rate", self._RATE,
                 "--text", text, "--write-media", tmp_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return tmp_path
        except Exception:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            return None

    def play(self, mp3_path: str) -> None:
        """Plays a synthesised MP3 (blocking) and deletes it afterwards."""
        if not self._player or not mp3_path:
            return
        try:
            # Non-blocking launch so stop() can terminate it; wait to keep
            # phrases strictly sequential.
            with self._proc_lock:
                self._proc = subprocess.Popen(
                    self._player + [mp3_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc = self._proc
            proc.wait()
        except Exception:
            pass
        finally:
            with self._proc_lock:
                self._proc = None
            if os.path.exists(mp3_path):
                try:
                    os.unlink(mp3_path)
                except Exception:
                    pass
        

    def stop(self) -> None:
        """Immediately silences any playback currentrly in progress."""
        with self._proc_lock:
            proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass


# ==============================================================================
# HMI NODE
# ==============================================================================

class InterfaceNode(Node, QWidget):
    """
    Combined ROS 2 node and PyQt5 widget implementing the LARIC HMI.

    Attributes:
        log_signal (pyqtSignal): Qt signal used to safely append HTML text to
            the feedback log from any thread. Accepts '(text: str, color: str)'.
        pub_human: ROS 2 publisher for 'std_msgs/String' on '/from_human'.
        pub_cmd_vel: ROS 2 publisher for 'geometry_msgs/TwistStamped' on 
            '/cmd_vel'.
        pub_emergency_stop: ROS 2 publisher for 'std_msgs/Bool' on
            '/emergency_stop'.
        pub_language: ROS 2 publisher for 'std_msgs/String' on
            '/language_changed'. Broadcasts the active language code so the
            agent process can update its gettext catalog in sync with the HMI.
        sub_feedback: ROS 2 subscription to 'std_msgs/String' on
            '/robot_feedback'.
        audio_queue (queue.Queue): Thread-safe buffer that accumulates raw
            audio chunks from the 'sounddevice' input stream callback.
        is_recording (bool): 'True' while a PTT recording session is active.
        recognizer (sr.Recognizer): Google Speech Recognition engine instance.
        stream: Active 'sounddevice.InputStream', or 'None' when idle.
        stt_language (str): Locale code passed to the Google STT API
            (e.g. 'en-US', 'es-ES'); kept in sync with the dropdown.
        stop_timer (QTimer): Single-shot debounce timer for the PTT key release.
        ros_timer (QTimer): Periodic timer that drives the ROS 2 spin loop.
        label (QLabel): Status indicator widget showing the current PTT state.
        log (QTextEdit): Read-only feedback log widget.
        text_input (QLineEdit): Manual text command input field.
        language_selector (QComboBox): EN/ES language selector dropdown.
        emergency_btn (QPushButton): Hardware-level emergency stop button.
    """

    # Class-level signal declaration required by the Qt meta-object system.
    # Signals must be declared at class scope before __init__ is called.
    log_signal: pyqtSignal = pyqtSignal(str, str)  # (html_text, css_colour)

    def __init__(self) -> None:
        """Initialise the ROS 2 node, Qt widget, audio and timers."""
        # 1. Initialise both parents (Node before QWidget — order matters).
        Node.__init__(self, ("laric_interface"))
        QWidget.__init__(self)

        # 2. Connect the thread-safe log signal to its GUI slot (always runs
        #    on the Qt main thread).
        self.log_signal.connect(self.update_log_slot)

        # 3. Create ROS 2 publishers/subscribers (/from_human, /cmd_vel,
        #    /emergency_stop, /language_changed, /robot_feedback).
        self.pub_human = self.create_publisher(String, "/from_human", 10)

        # Direct /cmd_vel and /emergency_stop publishers for hardware halt
        # Real robot uses ros2_control's diff_drive_controller, which subscribes
        # to /cmd_vel as TwistStamped (Jazzy default). Gazebo's simulation 
        # plugin subscribes as plain Twist.
        cmd_vel_type = TwistStamped if is_real_robot else Twist
        self.pub_cmd_vel = self.create_publisher(cmd_vel_type, "/cmd_vel", 10)
        self.pub_emergency_stop = self.create_publisher(
            Bool, "/emergency_stop", 10
        )

        # /language_changed broadcasts the active language code to agent_logic
        # (separate process), keeping its '_()' translator in sync.
        self.pub_language = self.create_publisher(
            String, "/language_changed", 10
        )

        self.sub_feedback = self.create_subscription(
            String, "/robot_feedback", self._feedback_callback, 10
        )
        
        self.sub_voice = self.create_subscription(
            String, "/laric_voice", self._voice_callback, 10
        )

        # 4. Initialise audio state
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_recording: bool = False
        self.recognizer: sr.Recognizer = sr.Recognizer()
        self.stream = None

        # Default STT locale; switched to "es-ES" via the language dropdown.
        self.stt_language = "en-US"
        
        # Default TTS state kept in sync with the language dropdown.
        self.tts_language = "en"
        self._voice_muted = False
        self._voice_speaker = VoiceSpeaker()
        self._voice_queue: queue.Queue = queue.Queue()
        self._play_queue: queue.Queue = queue.Queue()
        self._synth_thread = threading.Thread(
            target=self._synth_worker, daemon=True
        )
        self._play_thread = threading.Thread(
            target=self._play_worker, daemon=True
        )
        self._synth_thread.start()
        self._play_thread.start()

        # 5. Configure the single-shot PTT debounce timer (a re-press within
        #    PTT_DEBOUNCE_MS cancels it in keyPressEvent).
        self.stop_timer = QTimer()
        self.stop_timer.setSingleShot(True)
        self.stop_timer.setInterval(PTT_DEBOUNCE_MS)
        self.stop_timer.timeout.connect(self._stop_recording)

        # 6. Configure the ROS 2 integration timer (spin_once() every
        #    ROS_SPIN_INTERVAL_MS, without blocking Qt's event loop).
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
        """
        self.setWindowTitle(_("LARIC - Voice & Text Interface"))
        self.setGeometry(*HMI_GEOMETRY)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # -- Status label (PTT indicator) ------------------------------------
        self.label = QLabel(_("HOLD [SPACE] TO SPEAK"))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                background: #333333;
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                letter-spacing: 1px;
            }
        """)
        layout.addWidget(self.label)

        # -- Feedback log (read-only HTML console) ---------------------------
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        # Qt.NoFocus prevents the log widget from stealing keyboard focus
        # away from the main window, which would break spacebar PTT detection.
        self.log.setFocusPolicy(Qt.NoFocus)
        self.log.setStyleSheet("""
            QTextEdit {
                background: #252526;
                border: 1px solid #3e3e42;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.log)

        # -- Input Area (Text + Language) -------------------------------------
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText(
            _("Type a command and press Enter...")
        )
        self.text_input.setStyleSheet("""
            QLineEdit {
                background: #333333;
                border: 1px solid #3e3e42;
                padding: 12px 15px;
                border-radius: 8px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #4FC3F7;
            }
        """)
        self.text_input.returnPressed.connect(self._send_text_command)
        input_layout.addWidget(self.text_input)

        # Language dropdown. Qt stylesheets need 'url(<path>)', so the arrow
        # glyph is written to a temp SVG (removed in closeEvent()).
        svg_content = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6" '
            'viewBox="0 0 10 6">'
            '<path d="M0 0 L5 6 L10 0 Z" fill="#d4d4d4"/></svg>'
        )
        temp_svg = tempfile.NamedTemporaryFile(
            delete=False, suffix=".svg", mode='w'
        )
        temp_svg.write(svg_content)
        temp_svg.close()
        self._svg_arrow_path: str = temp_svg.name
        svg_path = self._svg_arrow_path

        # 2. Configure the language ComboBox
        self.language_selector = QComboBox()
        self.language_selector.addItems(["EN", "ES"])
        self.language_selector.setFixedSize(65, 45)
        self.language_selector.setCursor(Qt.PointingHandCursor)
        self.language_selector.setFocusPolicy(Qt.NoFocus)   

        # 3. Apply the stylesheet pointing at the temporary on-disk SVG arrow.
        # 'combobox-popup: 0;' makes the dropdown open below the box; the
        # QAbstractItemView block themes the popup to match the dark HMI.
        self.language_selector.setStyleSheet(f"""
            QComboBox {{
                background: #333333;
                border: 1px solid #3e3e42;
                border-radius: 8px;
                padding-left: 10px;
                color: #d4d4d4;
                font-weight: bold;
                combobox-popup: 0;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 20px;
                border: none;
            }}
            QComboBox::down-arrow {{
                image: url("{svg_path.replace(os.sep, '/')}");
                width: 10px;
                height: 6px;
            }}
            QComboBox::down-arrow:on {{
                image: url("{svg_path.replace(os.sep, '/')}");
                transform: rotate(180deg);
                width: 10px;
                height: 6px;
            }}
            QComboBox QAbstractItemView {{
                background: #333333;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 8px;
                padding: 4px;
                selection-background-color: #4FC3F7;
                selection-color: #1e1e1e;
                outline: 0;
            }}
        """)
        
        self.language_selector.currentIndexChanged.connect(
            self._update_language
        )
        self.language_selector.setFocusPolicy(Qt.NoFocus)
        input_layout.addWidget(self.language_selector)

        layout.addLayout(input_layout)

        # -- Mute / unmute button --------------------------------------------
        # Same footprint as the language selector for visual consistency.
        self.mute_button = QPushButton("🔊")
        self.mute_button.setFixedSize(65, 45)
        self.mute_button.setCursor(Qt.PointingHandCursor)
        self.mute_button.setFocusPolicy(Qt.NoFocus)
        self.mute_button.setCheckable(True)
        self.mute_button.setStyleSheet("""
            QPushButton {
                background: #333333;
                border: 1px solid #3e3e42;
                border-radius: 8px;
                color: #d4d4d4;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover { border: 1px solid #4FC3F7; }
            QPushButton:checked {
                background: #4d2222;
                border: 1px solid #D32F2F;
            }
        """)
        self.mute_button.clicked.connect(self._toggle_mute)
        input_layout.addWidget(self.mute_button)

        # -- Emergency Stop Button -------------------------------------------
        self.emergency_btn = QPushButton(_("EMERGENCY STOP (Ctrl+E)"))
        self.emergency_btn.setCursor(Qt.PointingHandCursor)
        self.emergency_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                background: #D32F2F;
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QPushButton:hover { background: #B71C1C; }
            QPushButton:pressed { background: #FF5252; }
        """)
        self.emergency_btn.clicked.connect(self._emergency_stop)
        layout.addWidget(self.emergency_btn)

        # Register a global shortcut for Ctrl+E to trigger the emergency stop
        self.shortcut_estop = QShortcut(QKeySequence("Ctrl+E"), self)
        self.shortcut_estop.activated.connect(self._emergency_stop)

        # Qt.StrongFocus lets the main window capture keys for spacebar PTT.
        self.setFocusPolicy(Qt.StrongFocus)
        self.setLayout(layout)
        self.setFocus()


    # --------------------------------------------------------------------------
    # MUTE / UNMUTE SELECTION
    # --------------------------------------------------------------------------
    def _toggle_mute(self) -> None:
        """Toggles spoken feedback on/off and reflects state on the button."""
        self._voice_muted = self.mute_button.isChecked()
        if self._voice_muted:
            self.mute_button.setText("🔇")
            self._interrupt_voice()   # cut off anything mid-sentence
            self.add_log(_("[SYSTEM] Voice muted."), "#4FC3F7")
        else:
            self.mute_button.setText("🔊")
            self.add_log(_("[SYSTEM] Voice unmuted."), "#4FC3F7")
        self.setFocus()   # keep spacebar PTT working


    # --------------------------------------------------------------------------
    # LANGUAGE SELECTION
    # --------------------------------------------------------------------------
    def _update_language(self, index: int) -> None:
        """
        Switches the HMI language and notifies the agent process.

        Args:
            index: Selected index in the language dropdown.
                   0 -> English, 1 -> Spanish.
        """
        # 1. Map dropdown index to ISO codes for STT and gettext
        if index == 0:
            self.stt_language = "en-US"
            lang_code = "en"
        elif index == 1:
            self.stt_language = "es-ES"
            lang_code = "es"
        else:
            return
        
        # Keep the TTS language in sync with the dropdown (gTTS code).
        self.tts_language = lang_code

        # 2. Activate the new catalog locally. All subsequent '_("...")' calls
        #    in this process resolve through the freshly loaded translator.
        set_language(lang_code)

        # 3. Refresh the static labels built with the previous catalog (dynamic
        #    strings retranslate naturally next time).
        self.label.setText(_("HOLD [SPACE] TO SPEAK"))
        self.text_input.setPlaceholderText(
            _("Type a command and press Enter...")
        )
        self.emergency_btn.setText(_("EMERGENCY STOP (Ctrl+E)"))

        # 4. Broadcast the change to agent_logic (separate process).
        msg = String()
        msg.data = lang_code
        self.pub_language.publish(msg)

        # 5. Confirm in the log using the now-active catalog.
        self.add_log(_("[SYSTEM] Language changed."), "#4FC3F7")

    # --------------------------------------------------------------------------
    # WINDOW CLOSE EVENT
    # --------------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """
        Intercepts the window close ('X') and launches the shutdown script to
        stop all ROS 2 nodes, simulators and background processes.
        """
        self.add_log(
            (_("[SYSTEM] Shutting down entire LARIC system...")), "#FF4444"
        )
        
        # Silence any speech immediately, then stop the voice worker.
        try:
            self._voice_speaker.stop()
            self._voice_queue.put(None)
        except Exception:
            pass
        
        try:
            # Verify the script exists at the dynamically resolved path
            if os.path.exists(SHUTDOWN_SCRIPT_PATH):
                # Launch the shutdown script in a detached session
                subprocess.Popen(
                    ["bash", SHUTDOWN_SCRIPT_PATH], start_new_session=True
                )
            else:
                self.add_log((_("[ERROR] Shutdown script not found.")), "red")
        except Exception as e:
            print(f"Error launching shutdown script: {e}")

        # Clean up the temporary SVG arrow file created in _init_ui. Wrapped in
        # try/except because cleanup must never block window closure.
        try:
            if getattr(self, "_svg_arrow_path", None) and \
               os.path.exists(self._svg_arrow_path):
                os.unlink(self._svg_arrow_path)
        except Exception:
            pass

        # Accept the event to allow the window to close visually
        event.accept()

    # --------------------------------------------------------------------------
    # ROS 2 INTEGRATION
    # --------------------------------------------------------------------------

    def _spin_ros(self) -> None:
        """
        Executes one iteration of the ROS 2 event loop.
        """
        rclpy.spin_once(self, timeout_sec=0)

    def _feedback_callback(self, msg: String) -> None:
        """
        ROS 2 subscription callback for '/robot_feedback' messages.

        Args:
            msg: Incoming 'std_msgs/String' message containing the agent's
                feedback text.
        """
        self.add_log(f"{msg.data}", "#4FC3F7")


    # --------------------------------------------------------------------------
    # VOICE OUTPUT
    # --------------------------------------------------------------------------
    def _voice_callback(self, msg: String) -> None:
        """
        ROS 2 callback for /laric_voice: queues a phrase for text-to-speech.

        Args:
            msg: Incoming 'std_msgs/String' with the short phrase to speak.
        """
        phrase = msg.data.strip()
        if phrase and not self._voice_muted:
            # Snapshot the current language so the phrase is spoken in whatever
            # language was active when it was issued.
            self._voice_queue.put((phrase, self.tts_language))

    def _synth_worker(self) -> None:
        """Synthesises queued phrases ahead of playback, preserving order."""
        while True:
            item = self._voice_queue.get()
            if item is None:
                self._play_queue.put(None)   # propagate shutdown
                break
            text, lang = item
            mp3_path = self._voice_speaker.synthesize(text, lang)
            if mp3_path:
                self._play_queue.put(mp3_path)

    def _play_worker(self) -> None:
        """Plays synthesised phrases one at a time, in order."""
        while True:
            mp3_path = self._play_queue.get()
            if mp3_path is None:
                break
            self._voice_speaker.play(mp3_path)

    def _interrupt_voice(self) -> None:
        """
        Discards queued/in-flight speech and silences playback in progress.
        """
        # Drop phrases still waiting to be synthesised.
        self._drain_queue(self._voice_queue)
        # Drop already-synthesised phrases waiting to play.
        for mp3_path in self._drain_queue(self._play_queue):
            if mp3_path and os.path.exists(mp3_path):
                try:
                    os.unlink(mp3_path)
                except Exception:
                    pass
        # Silence whatever is playing right now.
        self._voice_speaker.stop()

    @staticmethod
    def _drain_queue(q: queue.Queue) -> list:
        """Removes and returns all items currently in 'q'."""
        items = []
        try:
            while True:
                items.append(q.get_nowait())
        except queue.Empty:
            pass
        return items


    # --------------------------------------------------------------------------
    # TEXT INPUT
    # --------------------------------------------------------------------------

    def _send_text_command(self) -> None:
        """
        Publishes the text field content to '/from_human' and resets focus.
        """
        text = self.text_input.text().strip()
        if not text:
            return
        
        # 0. Silence any current/queued speech so the new command takes over.
        self._interrupt_voice()

        # 1. Log the command locally before publishing
        self.add_log(_("[USER - TEXT]: {}").format(text), "#8BC34A")

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
        """
        # Step 1: Hard-stop — direct /cmd_vel publish (Twist defaults to 0.0)
        # TwistStamped is required on the real robot; the stamp is set to 'now' 
        # so the diff_drive_controller does not reject the message as stale
        if is_real_robot:
            halt_msg = TwistStamped()
            halt_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_cmd_vel.publish(halt_msg)
        else:
            self.pub_cmd_vel.publish(Twist())

        # Step 2: Signal agent_logic to kill all running motion threads
        flag = Bool()
        flag.data = True
        self.pub_emergency_stop.publish(flag)

        # Step 3: Inform the operator
        self.add_log(_("[EMERGENCY STOP] Robot halted by user."), "#FF4444")

        # Step 4: Silence any current/queued speech.
        self._interrupt_voice()
        
        # Return focus to main window so PTT continues to work
        self.setFocus()

    # --------------------------------------------------------------------------
    # THREAD-SAFE LOG
    # --------------------------------------------------------------------------

    def add_log(self, text: str, color: str = "#d4d4d4") -> None:
        """
        Queues a log entry for display on the Qt main thread.

        Args:
            text: Plaintext or HTML content to display in the log.
            color: CSS colour string applied to the message text.
                   Defaults to light grey ('#d4d4d4').
        """
        self.log_signal.emit(text, color)

    def update_log_slot(self, text: str, color: str) -> None:
        """
        Appends a colour-coded HTML entry to the feedback log widget.

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

        Args:
            event (QKeyEvent): The key press event delivered by Qt.
        """
        # Delegate all events to the default handler when the text box is active
        if self.text_input.hasFocus():
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key_Space:
            # Ignore OS auto-repeat keydowns; a held key records once.
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

        Args:
            event (QKeyEvent): The key release event delivered by Qt.
        """
        if self.text_input.hasFocus():
            return

        if event.key() == Qt.Key_Space:
            if not event.isAutoRepeat() and self.is_recording:
                # Start the debounce timer (PTT_DEBOUNCE_MS window to re-press).
                self.stop_timer.start()

    # --------------------------------------------------------------------------
    # AUDIO RECORDING
    # --------------------------------------------------------------------------

    def _start_recording(self) -> None:
        """
        Opens the microphone stream and updates the UI to the recording state.
        """
        # 1. Set recording flag and clear the audio buffer
        self.is_recording = True
        self.audio_queue.queue.clear()

        # 2. Switch the status label to the recording style (geometry matches
        #    _init_ui so the layout does not jitter between states).
        self.label.setText(_("[RECORDING IN PROGRESS...]"))
        self.label.setStyleSheet(
            "font-size: 14px;"
            "background: #4CAF50;"
            "color: white;"
            "padding: 15px;"
            "border-radius: 8px;"
            "font-weight: bold;"
            "letter-spacing: 1px;"
        )

        # 3. Open the sounddevice input stream (stderr suppressed to hide noisy
        #    ALSA/PortAudio driver warnings).
        try:
            with _suppress_stderr():
                self.stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    callback=self._audio_callback,
                )
                self.stream.start()
        except Exception as exc:
            self.add_log(
                _("[HARDWARE ERROR] Microphone: {}").format(exc), "red"
            )
            self.is_recording = False
            self._reset_ui()

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time, status
    ) -> None:
        """
        Sounddevice stream callback — enqueues raw audio chunks.

        Args:
            indata: NumPy array of shape '(frames, channels)' containing the
                captured PCM samples in float32 format.
            frames: Number of frames in this chunk.
            time: 'sounddevice.CallbackFlags' timing info (unused).
            status: 'sounddevice.CallbackFlags' status flags (unused).
        """
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def _stop_recording(self) -> None:
        """
        Stops the microphone stream and triggers background audio processing.
        """
        if not self.is_recording:
            return

        # 1. Clear the recording flag before touching the stream to prevent
        #    the audio callback from enqueuing more data after it is closed.
        self.is_recording = False

        # 2. Close the sounddevice stream
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass  # Stream may already be in an error state; proceed regardless

        # 3. Update UI to the processing state (same geometry as idle/recording
        #    so the label does not resize while STT is in flight).
        self.label.setText(_("[PROCESSING AUDIO...]"))
        self.label.setStyleSheet(
            "font-size: 14px;"
            "background: #FFC107;"
            "color: black;"
            "padding: 15px;"
            "border-radius: 8px;"
            "font-weight: bold;"
            "letter-spacing: 1px;"
        )

        # 4. Offload the (network-bound) STT call to a background thread so the
        #    UI does not freeze.
        threading.Thread(target=self._process_audio, daemon=True).start()

    # --------------------------------------------------------------------------
    # AUDIO PROCESSING & STT
    # --------------------------------------------------------------------------

    def _process_audio(self) -> None:
        """
        Transcribes the recorded audio and publishes the result to ROS 2.

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
                    _("[WARNING] Audio buffer is empty. "
                      "Nothing to transcribe."), 
                    "orange"
                )
                return

            full_audio = np.concatenate(chunks, axis=0)

            # 2. Discard recordings shorter than MIN_AUDIO_DURATION_S
            #    (accidental taps), avoiding needless STT calls.
            duration_s = len(full_audio) / SAMPLE_RATE
            if duration_s < MIN_AUDIO_DURATION_S:
                self.add_log(
                    _("[WARNING] Recording too short "
                      "({:.2f}s &lt; {:.2f}s). Discarded.")
                    .format(duration_s, MIN_AUDIO_DURATION_S),
                    "orange",
                )
                return

            # 3. Convert float32 PCM to an in-memory int16 WAV (what
            #    SpeechRecognition expects); 32767 scales [-1.0, 1.0] to int16.
            audio_int16 = (full_audio * 32767).astype(np.int16)
            wav_buffer = io.BytesIO()
            wav.write(wav_buffer, SAMPLE_RATE, audio_int16)
            wav_buffer.seek(0)

            # 4. Feed the WAV buffer to the SpeechRecognition engine
            with sr.AudioFile(wav_buffer) as source:
                audio_data = self.recognizer.record(source)

            # 5. Call the Google STT API.
            text = self.recognizer.recognize_google(
                audio_data, language=self.stt_language
            )

            self.add_log(_("[USER - VOICE]: {}").format(text), "#8BC34A")

            # 6. Silence any current/queued speech.
            self._interrupt_voice()

            # 7. Publish the transcription to the agent's input topic
            msg = String()
            msg.data = text.lower()
            self.pub_human.publish(msg)

        except sr.UnknownValueError:
            # Raised when the audio contains only silence or is unintelligible.
            self.add_log(
                _("[WARNING] No speech detected ")
                + _("(silence or unintelligible audio)."),
                "orange"
            )

        except sr.RequestError as exc:
            # Raised when the Google STT API is unreachable or returns an error.
            self.add_log(
                _("[ERROR] Google STT API request failed: {}").format(exc),
                "red"
            )

        except Exception as exc:
            self.add_log(
                _("[ERROR] Unexpected error during audio "
                  "processing: {}").format(exc),
                "red"
            )

        finally:
            # 7. Always reset the UI to idle (QTimer.singleShot hops to the Qt
            #    main thread, since this runs on a daemon thread).
            QTimer.singleShot(0, self._reset_ui)

    # --------------------------------------------------------------------------
    # UI STATE HELPERS
    # --------------------------------------------------------------------------

    def _reset_ui(self) -> None:
        """
        Resets the status label to the idle (ready-to-record) state.
        """
        self.label.setText(_("HOLD [SPACE] TO SPEAK"))
        self.label.setStyleSheet(
            "font-size: 14px;"
            "background: #333333;"
            "color: white;"
            "padding: 15px;"
            "border-radius: 8px;"
            "font-weight: bold;"
            "letter-spacing: 1px;"
        )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main() -> None:
    """
    Application entry point.
    """
    # 1. Initialise rclpy before creating the Node
    rclpy.init()

    # 2. Create the Qt application object (must exist before any QWidget)
    app = QApplication(sys.argv)

    # 3. Instantiate the combined ROS 2 node / Qt widget
    node = InterfaceNode()
    node.show()
    # Raise the window and grab focus on launch (advisory under native
    # Wayland; hence the forced xcb backend above).
    node.raise_()
    node.activateWindow()

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