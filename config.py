"""Configuration values and mappings for the HRC communication system."""

# The LSTM step head's output class names, in output-index order (index == the
# model's step_head column). Mirrors LSTM_HRC/data_proc_3d/src/skeleton_pipeline/
# dataset/labels.py's LABEL_MAP_DICT entries 0-6 -- that repo runs in its own
# separate venv (see that file's docstring) so the names are copied here by hand
# rather than imported; keep in sync if the label taxonomy changes. Entry 7
# ("Mistake") and 8 ("No Related Task") are not part of this 7-class step head
# (best_model/*/config.json's "num_steps": 7).
STEP_NAMES = [
    "Pull Cables",
    "Lift",
    "Place",
    "Align",
    "Screw",
    "Connect Cables",
    "Clamp Coupling",
]

RESPONSE_TIMEOUT_SECONDS = 20.0
DEFER_SECONDS = 5.0
RECOVERY_STOP_DELAY_SECONDS = 0.5

DEFAULT_SPEED = 0.2
SPEED_STEP = 0.1

MIN_SPEED = 0.00001
MAX_SPEED = 0.75

LAST_SPEED = DEFAULT_SPEED

# Keep return-home recovery disabled until the joint ranges are validated on the real robot.
RETURN_HOME_RECOVERY_ENABLED = True

# TEST PLACEHOLDER for a 6-joint robot, centered at 0 rad.
# Replace with ranges validated on the real robot before enabling recovery.
SAFE_RETURN_JOINT_RANGES = [
    (-3, 3),  # joint 1
    (-3, 3),  # joint 2
    (-3, 3),  # joint 3
    (-3, 3),  # joint 4
    (-3, 3),   # joint 5
    (-3, 3),  # joint 6
]

# Human task step ids, as the recognition model's step head numbers them (STEP_NAMES
# order). Looked up by name so a reordered head can't silently shift them -- the old
# literals (3/4/5) were ELAN annotation ids and pointed at the wrong model classes.
HUMAN_PULL_CABLES = STEP_NAMES.index("Pull Cables")
HUMAN_SCREW_DONE = STEP_NAMES.index("Screw")
HUMAN_CONNECT_PIPES = STEP_NAMES.index("Connect Cables")
HUMAN_CLAMP_TOOL = STEP_NAMES.index("Clamp Coupling")

TASK_LIFT_PANEL = 1
TASK_LEAVE = 2
TASK_BRING_CONNECTOR = 3
TASK_BRING_CLAMPING_TOOL = 4
TASK_RETURN_CLAMPING_TOOL = 5

# When robot tasks are offered is no longer configured here: the task database's
# "Robot task trigger info" drives it (2_decision_making/robot_trigger_policy.py).
TASK_DATABASE_PATH = "2_decision_making/task_database/task_database.json"
# P(next task | current task) from 2_decision_making/src/task_sequence_analysis.py,
# the variant that keeps Lift (the model head and the database both have it).
TASK_TRANSITION_TABLE_PATH = "2_decision_making/results/task_sequence_lift/transition_probabilities.csv"

# Tracked (database) task -> the robot task that performs it. Tasks the database lets
# the robot do but that have no entry here (Pull Cables, Place) are never dispatched.
TRACKED_TO_ROBOT_TASK = {
    "Lift": TASK_LIFT_PANEL,
    "Bring Tool": TASK_BRING_CLAMPING_TOOL,
    "Bring Connector": TASK_BRING_CONNECTOR,
    "Bring back Tool": TASK_RETURN_CLAMPING_TOOL,
}

# A not-done recognized task is pending when P(task | reference task) reaches this.
PENDING_MIN_PROBABILITY = 0.05
# The recognition stabilizer only switches to a task this likely after the current one...
RECOGNITION_FILTER_MIN_PROBABILITY = 0.02
# ...unless the candidate holds for confirmation_count * this many frames anyway.
RECOGNITION_FILTER_OVERRIDE_FACTOR = 4
# The progress head's raw output divided by this gives 0-1 (the database's "Progress"
# thresholds are 0-1). Check against a replay: training labels ran 0-100.
RECOGNITION_PROGRESS_SCALE = 1.0
# Recognition re-publishes a task update once progress has moved this much (0-1 scale).
RECOGNITION_PROGRESS_PUBLISH_DELTA = 0.05

PERMISSION_MESSAGES = {
    TASK_LIFT_PANEL: "Would you like me to lift the panel? Say yes, no, or later after the beep, or type your reply.",
    TASK_LEAVE: "Screwing is finished. Would you like me to release the panel and move away? Say yes, no, or later after the beep, or type your reply.",
    TASK_BRING_CONNECTOR: "I have moved away from the panel. Would you like me to bring the pipe connector? Say yes, no, or later after the beep, or type your reply.",
    TASK_BRING_CLAMPING_TOOL: "Would you like me to bring the clamping tool? Say yes, no, or later after the beep, or type your reply.",
    TASK_RETURN_CLAMPING_TOOL: "Would you like me to take the clamping tool back? Say yes, no, or later after the beep, or type your reply.",
}

# Per-task overrides; unspecified values use the global durations above.
TASK_TIMINGS = {}

ROS_BRIDGE_HOST = "127.0.0.1"
ROS_BRIDGE_PORT = 9090

ROS_TOPICS = {
    "control": "/Robot/control",  # stop, home, and other robot control commands
    "global_speed": "/Robot/globalSpeed",  # std_msgs/Float64
    "local_speed": "/Robot/localSpeed",  # std_msgs/Float64
    "human_done": "/Human/taskSuccess",  # success
    "robot_success": "/Robot/status/physical",  # running, success, homed
    "robot_position": "/UR10/position/live",  # trajectory_msgs/JointTrajectoryPoint
    "gripper": "/Robot/gripper",  # std_msgs/Bool: False=closed, True=open
    "free_drive": "/Robot/teachMode",
    "r_task_done": "/Task/signal",
    "human_position": "/Human/position/live",  # std_msgs/String: JSON-encoded {header, point,
                                                # keypoints} -- see ros_communication.py's
                                                # publish_human_location() docstring for why this
                                                # isn't a stock geometry_msgs/PointStamped (that
                                                # type has no room for the keypoints dict).

    # Two RTDE readers coexist and each names its own topics -- neither set is dead, so
    # both stay. 4_execution/read_ur_live_data.py publishes the first three;
    # 4_execution/ur_state_reader.py publishes the rest. Both are separate from
    # "robot_position" above, which comes from the ur_robot_driver ROS node.
    "ur_joint_position": "/UR10e/position/live",     # sensor_msgs/JointState
    "ur_tcp_position": "/UR10e/TCPPosition/live",    # geometry_msgs/PoseStamped
    "ur_tcp_force": "/UR10e/TCPForce/live",          # geometry_msgs/WrenchStamped

    "joint_state": "/UR10/joint_states",  # sensor_msgs/JointState
    "tcp_pose": "/UR10/tcp_pose",  # geometry_msgs/PoseStamped
    "ft_wrench": "/UR10/ftsensor/wrench",  # geometry_msgs/WrenchStamped
    "ft_zero": "/UR10/ftsensor/zero",  # std_msgs/Empty: publish to tare the FT sensor
}

# How often recognition publishes HUMAN_LOCATION_UPDATE events, in frames -- see
# run_recognition.py's main loop. 5 -> ~5-6Hz at a 30fps camera, well above what the
# discrete task-state events on this bus were designed for at full frame rate.
HUMAN_LOCATION_PUBLISH_EVERY_N_FRAMES = 5

GH_STEP_MESSAGES = {
    TASK_LIFT_PANEL: {
        "suggested_action": "assist_lifting",
    },
    TASK_LEAVE: {
        "suggested_action": "leave",
    },
    TASK_BRING_CONNECTOR: {
        "suggested_action": "bring_pipe_connector",
    },
    TASK_BRING_CLAMPING_TOOL: {
        "suggested_action": "bring_clamping_tool",
    },
    TASK_RETURN_CLAMPING_TOOL: {
        "suggested_action": "return_clamping_tool",
    },
}

UDP_HOST = "127.0.0.1"
UDP_PORT = 5006

EVENT_TRANSPORT_HOST = "127.0.0.1"
EVENT_TRANSPORT_PORT = 5010

LOG_FILE_PATH = "hrc_communication_events.log"

VOICE_ENABLED = True

VOICE_GPT_ENABLED = False

# List audio devices: .venv\Scripts\python.exe -m sounddevice
VOICE_MODEL_PATH = "3_communication/vosk_fallback/models/vosk-model-small-en-us-0.15"
# Left at None, sounddevice falls back to the OS default input, which on this
# laptop is a virtual NDI webcam audio device (no real signal) rather than the
# physical mic -- pin it explicitly. Run `uv run python -m sounddevice` to list
# devices and update this if the laptop's mic name/index differs.
VOICE_INPUT_DEVICE_NAME = "Microphone Array (AMD Audio Dev"
VOICE_OUTPUT_DEVICE_NAME = None
VOICE_LISTEN_TIMEOUT_SECONDS = 8.0
VOICE_TTS_RATE = 220
VOICE_POST_TTS_GUARD_SECONDS = 0.2
VOICE_MAX_ATTEMPTS = 2
VOICE_ERROR_RETRY_SECONDS = 5.0

test_vid_path = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\raw\cam-04\video__cam-04_uid-01_take-01.mp4"


# ROBOT_IP = "169.254.130.206" 
ROBOT_IP = "127.0.0.1"