# VIRC — Voice Interactive Robot Control

> Control a TurtleBot3 using natural language (voice or text) in Spanish or English.  
> Say *"ve al baño"* or *"move forward 1 metre and then turn right 90 degrees"* — no robotics knowledge needed.

**Author:** Lourdes Francés Llimerá · ai2, Universitat Politècnica de València  
**Tutor:** Juan Francisco Blanes Noguera  
**Degree:** Grado en Informática Industrial y Robótica — TFG 2025-2026

---

## What is VIRC?

VIRC is an intelligent robot control system that bridges natural language and ROS 2 robotics. A user speaks or types a command, a large language model (Llama 3.3 70B) interprets the intent and selects the appropriate Nav2 action, and the robot executes it safely with continuous position awareness.

The system currently runs on a simulated TurtleBot3 (Gazebo + `turtlebot3_house`) but is architecturally designed to transfer to physical hardware without code changes.

---

## Repository Structure

```
TFG-VIRC/
├── README.md
├── requirements.txt                 # Python dependencies for virc_env
├── docs/                            # Project documentation
│   ├── documentacion.pdf            # Full TFG report
│   ├── glosario.pdf                 # Technical glossary
│   ├── manual_comandos.pdf          # Voice/text command reference
│   ├── manual_configuracion.pdf     # Full installation guide
│   └── manual_programacion.pdf      # Code walkthrough
└── virc_ws/
    ├── start_simulation.sh          # Launch Gazebo + Nav2 + HMI + Agent
    ├── stop_simulation.sh           # Clean shutdown of all processes
    ├── maps/
    │   ├── tb3_house_map.yaml/.pgm  # Simulation environment map
    │   └── laboratory_map.yaml/.pgm # Physical lab map
    └── src/
        └── rai_voice_ctrl/
            ├── package.xml
            ├── setup.py / setup.cfg
            └── rai_voice_ctrl/
                ├── agent_logic.py   # AI agent, tools, ROS 2 control
                ├── rai_interface.py # PyQt5 HMI + Push-to-Talk
                └── config.py        # All configuration (API, map, prompt, UI strings)
```

---

## System Architecture

```
User (voice / text)
        │
        ▼
rai_interface.py  ──[/from_human]──▶  agent_logic.py
 [PyQt5 + STT]                         [LangChain + LLM]
        ▲                                      │
        │                                      ▼
[/robot_feedback] ◀──────────────   Custom Tools (×6)
[/emergency_stop] ──────────────▶          │
                                    Nav2 Action Servers
                    /spin  /drive_on_heading  /backup  /navigate_to_pose
                                            │
                                    TurtleBot3 / Gazebo
```

**Key ROS 2 topics:**

| Topic | Direction | Purpose |
|---|---|---|
| `/from_human` | HMI → Agent | User command text |
| `/robot_feedback` | Agent → HMI | Status messages |
| `/emergency_stop` | HMI → Agent | Hardware stop signal |
| `/initialpose` | RViz → Agent | Localisation initialisation |
| `/cmd_vel` | HMI → Robot | Direct velocity (emergency only) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Robot / Simulation | TurtleBot3 Waffle Pi + Gazebo (`turtlebot3_house`) |
| Middleware | ROS 2 Humble |
| Navigation | Nav2 (AMCL + Behavior Plugins + NavigateToPose) |
| AI Framework | [RAI by Robotec AI](https://github.com/RobotecAI/rai) + LangChain |
| LLM | Llama 3.3 70B via Groq API or local Ollama (ai2 UPV) |
| HMI | PyQt5 + Google Speech Recognition |

---

## Quick Start

> **Full installation guide** (VMware, Ubuntu 22.04, ROS 2, RAI, virc_env): see [`docs/manual_configuracion.pdf`](docs/manual_configuracion.pdf)

### Prerequisites
- Ubuntu 22.04 with ROS 2 Humble installed
- TurtleBot3 packages: `sudo apt install ros-humble-turtlebot3* -y`
- RAI framework installed from [https://github.com/lourdesfll29-maker/rai.git](https://github.com/lourdesfll29-maker/rai.git)
- Python virtual environment `virc_ws/virc_env` with `--system-site-packages`

### 1. Clone the repository
```bash
git clone https://github.com/lourdesfll29-maker/TFG-VIRC.git
cd TFG-VIRC/virc_ws
```

### 2. Set up the Python environment
```bash
python3 -m venv --system-site-packages virc_env
source virc_env/bin/activate
pip install -r ../requirements.txt
```

### 3. Configure the API key
VIRC can run in two modes, toggled by `from_lab` in `config.py`:

| Mode | `from_lab` | Requires |
|---|---|---|
| Local Ollama (ai2 UPV) | `True` | University network |
| Groq cloud API | `False` (default) | `GROQ_API_KEY` environment variable |

For Groq mode:
```bash
export GROQ_API_KEY="gsk_..."
```

### 4. Build the ROS 2 package
```bash
source /opt/ros/humble/setup.bash
source virc_env/bin/activate
colcon build --symlink-install --packages-select rai_voice_ctrl
source install/setup.bash
```

### 5. Launch the system
```bash
./start_simulation.sh
```

This opens four processes in order: **Gazebo** (10 s) → **Nav2 + RViz** (5 s) → **HMI** (3 s) → **Agent** (visible terminal).

### 6. Initialise robot localisation
Before sending any movement command, use the **"2D Pose Estimate"** tool in RViz to click on the robot's approximate position on the map. The HMI will confirm with:
```
[VIRC]: Robot localised via '2D Pose Estimate'. Ready for commands.
```

### 7. Stop the system
```bash
./stop_simulation.sh
```

---

## Available Commands

Full reference: [`docs/manual_comandos.pdf`](docs/manual_comandos.pdf)

| Intent | Example (ES) | Example (EN) |
|---|---|---|
| Rotate | "gira a la derecha 90 grados" | "turn right 90 degrees" |
| Move | "avanza 2 metros" | "move forward 2 metres" |
| Sequence | "avanza 1 metro y luego gira a la derecha" | "move 1m and then turn right" |
| Navigate | "ve al baño" | "go to the bathroom" |
| Stop | "para" / "detente" | "stop" |
| Identity | "¿qué puedes hacer?" | "what can you do?" |

**Known locations:** `bathroom`, `dining_room`, `entrance`, `outside`, `top_right_room`, `top_left_room`, `bottom_left_room`, `bottom_right_room`

### Emergency stop
- **Button** in the HMI interface (always visible)
- **Keyboard shortcut:** `Ctrl+E`

Both bypass the AI agent entirely and halt the robot with minimum latency.

---

## Key Design Decisions

- **All motion through Nav2:** SpinTool uses `/spin`, MoveTool uses `/drive_on_heading` or `/backup`, NavigationTool uses `/navigate_to_pose`. The robot always operates with its position known on the map.
- **Localisation as a hard prerequisite:** All motion tools check `localised_event` before executing. The event is set by a subscriber to `/initialpose`.
- **LangChain one-tool-per-turn solved by SequenceTool:** Multi-step commands are packaged into a single tool call with a Pydantic validator capping sequences at 5 steps.
- **Two-layer emergency stop:** UI button publishes zero-velocity directly to `/cmd_vel` (immediate), then signals the agent via `/emergency_stop` to cancel Nav2 goals.

---

## Credits

- Map `tb3_house_map` by A. Koubaa — [ros_course_part2](https://github.com/aniskoubaa/ros_course_part2)
- RAI Framework by [Robotec AI](https://github.com/RobotecAI/rai)
- Navigation stack: [Nav2](https://nav2.ros.org)
