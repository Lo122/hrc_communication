"""Configuration values and mappings for the HRC communication system."""

# The LSTM step head's output class names, in output-index order (index == the
# model's step_head column). Mirrors LSTM_HRC/data_proc_3d/src/skeleton_pipeline/
# dataset/labels.py's LABEL_MAP_DICT entries 0-6 -- that repo runs in its own
# separate venv (see that file's docstring) so the names are copied here by hand
# rather than imported; keep in sync if the label taxonomy changes. Entry 7
# ("Mistake") and 8 ("No Related Task") are not part of this 7-class step head
# (best_model/*/config.json's "num_steps": 7).
# STEP_NAMES = [
#     "Pull Cables",
#     "Lift",
#     "Align",
#     "Screw",
#     "Connect Cables",
#     "Clamp Coupling",
#     "Place"
# ]

STEP_NAMES = [
    "Pull Cables",
    "Lift",
    "Place",
    "Align",
    "Screw",
    "Connect Cables",
    "Clamp Coupling",
    "Non Related Task"
]

# The LSTM mistake head's output class names, in output-index order (only models
# trained with config.json's "num_mistakes", e.g. best_model/3d_skeleton_01). Class 0
# is "no mistake" -- RecognitionManager's mistake score is 1 - P(class 0).
MISTAKE_NAMES = [
    "OK",
    "Mistake",
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
TASK_PULL_CABLES = 6
# Move away from the hand-over position once a brought item is handed over.
TASK_LEAVE_HANDOVER = 7

# Robot tasks that end by handing an item to the human, and what the robot calls it:
# it asks to hand the item over, opens the gripper, then leaves (TASK_LEAVE_HANDOVER).
HANDOVER_ITEMS = {
    TASK_BRING_CONNECTOR: "pipe coupling",
    TASK_BRING_CLAMPING_TOOL: "tool",
}
# Hand-overs after which the robot leaves without asking, this many seconds after it
# has said so -- a delayed start, so "cancel" still keeps it there. The others ask
# "May I leave the hand-over position?".
HANDOVER_LEAVE_DELAY_S = {
    TASK_BRING_CONNECTOR: 1.0,
}

# Once the lift has brought the panel into position: False turns free drive on straight
# away for the human to adjust it; True asks first ("Would you like free drive?", a no
# holds the panel without adjusting).
LIFT_ASKS_FREE_DRIVE = False

# When robot tasks are offered is no longer configured here: the task database's
# "Robot task trigger info" drives it (2_decision_making/src/robot_trigger_policy.py).
TASK_DATABASE_PATH = "2_decision_making/task_database/task_database.json"
# P(next task | current task) from 2_decision_making/src/task_sequence_analysis.py,
# the variant that keeps Lift (the model head and the database both have it).
TASK_TRANSITION_TABLE_PATH = "2_decision_making/probability_table/transition_probabilities_02.csv"

# Tracked (database) task -> the robot task that performs it. Tasks the database lets
# the robot do but that have no entry here (Place) are never dispatched.
TRACKED_TO_ROBOT_TASK = {
    "Pull Cables": TASK_PULL_CABLES,
    "Lift": TASK_LIFT_PANEL,
    # Offered only for a panel the robot is holding (TaskManager checks that).
    "Leave from the panel": TASK_LEAVE,
    "Bring Tool": TASK_BRING_CLAMPING_TOOL,
    "Bring Connector": TASK_BRING_CONNECTOR,
    "Bring back Tool": TASK_RETURN_CLAMPING_TOOL,
}

# A not-done recognized task is pending when P(task | reference task) reaches this.
# Recognition moves the human only to a pending task (or a repeat this likely); the
# task they worked on before goes back to pending, so the human works on one at a time.
PENDING_MIN_PROBABILITY = 0.05
# ...unless the tracker has stayed on the reference task (working, or done and not
# moved on) longer than this statistic of its annotated duration: then recognition is
# believed anyway, so a wrong reference can't stall tracking. Per block -- one
# uninterrupted stretch of a task, e.g. all six screws -- since a working task spans one;
# the _lift variant, like the transition table, since it has a Lift row.
TASK_DURATION_STATS_PATH = "2_decision_making/results/task_sequence_lift/duration_stats_block.csv"
TASK_OVERRUN_STAT = "p95"
# The recognition stabilizer only switches to a task this likely after the current one...
RECOGNITION_FILTER_MIN_PROBABILITY = 0.02
# ...unless the candidate holds for confirmation_count * this many frames anyway.
RECOGNITION_FILTER_OVERRIDE_FACTOR = 7
# Steps recognition may switch to from any step, whatever the transition table says --
# e.g. Screw -> Clamp Coupling is rare in the annotations (0.019) but must not wait for
# the override. How sure the model must be per step: the task database's "Action
# Confidence Threshold".
RECOGNITION_FILTER_OPEN_STEPS = ("Screw", "Clamp Coupling")
# The progress head's raw output divided by this gives 0-1 (the database's "Progress"
# thresholds are 0-1). Check against a replay: training labels ran 0-100.
RECOGNITION_PROGRESS_SCALE = 1.0
# Recognition re-publishes a task update once progress has moved this much (0-1 scale)
# -- the stable step's, or any step's own lane for a model with one per step...
RECOGNITION_PROGRESS_PUBLISH_DELTA = 0.02
# ...or once any step's (smoothed) probability has moved this much: the decision layer
# picks the step from them (SEQUENCE_* below), not only from the stable step.
RECOGNITION_PROBABILITY_PUBLISH_DELTA = 0.05
# -- Sequence-aware step selection (2_decision_making/src/sequence_step_selector.py) --
# The models learned from human-only videos; beside the robot, the human's motion often
# looks like another step. From recognition's step probabilities the decision layer
# keeps only the tasks the sequence allows now (TaskTracker.potential_tasks), drops those
# below their "Action Confidence Threshold" in the task database (this default where it
# has none), weights the rest by the transition table and takes the best:
#   score = p * ((1 - STRENGTH) + STRENGTH * P(task | reference task))
# with weight 1 for staying on the reference task. STRENGTH 0: filtering only;
# 1: the full transition probability.
SEQUENCE_DEFAULT_MIN_CONFIDENCE = 0.5
SEQUENCE_PRIOR_STRENGTH = 0.5
# A switch away from the reference task counts once it wins this many task updates in a
# row (staying is immediate). Also how long recognition must show another step before
# Clamp Coupling, once under way, counts as done (CLAMP_COUPLING_DONE_* below).
SEQUENCE_CONFIRM_EVENTS = 3
# Recognition's model needs this long to recognize motion once it runs: its task
# updates count, and the demo opening starts, only this many seconds after the
# recognition process first reports in (a human location or a task update). Typed
# (--fake-recognition / --manual-trigger) and --debug-trigger updates always count.
# 0: from the start, without waiting for recognition.
RECOGNITION_ACTIVATION_S = 10.0

PERMISSION_MESSAGES = {
    TASK_LIFT_PANEL: "Would you like me to lift the panel? Say yes, no, or later after the beep, or type your reply.",
    TASK_LEAVE: "Would you like me to release the panel and move away? Say yes, no, or later after the beep, or type your reply.",
    # No "I have moved away from the panel" here: it is said on its own after a leave
    # (MessageManager.get_left_panel_message), and this offer also comes from the
    # database rule when the robot never held the panel.
    TASK_BRING_CONNECTOR: "Would you like me to bring the pipe connector? Say yes, no, or later after the beep, or type your reply.",
    TASK_BRING_CLAMPING_TOOL: "Would you like me to bring the clamping tool? Say yes, no, or later after the beep, or type your reply.",
    TASK_RETURN_CLAMPING_TOOL: "Would you like me to take the clamping tool back? Say yes, no, or later after the beep, or type your reply.",
    TASK_PULL_CABLES: "Would you like me to pull the cables? Say yes, no, or later after the beep, or type your reply.",
    TASK_LEAVE_HANDOVER: "May I leave the hand-over position and move away? Say yes, no, or later after the beep, or type your reply.",
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
    "gripper": "/Robot/gripper",  # std_msgs/Bool: False=closed, True=open; True also opens it (hand-over)
    "free_drive": "/Robot/teachMode",
    "r_task_done": "/Task/signal",
    "human_position": "/Human/position/live",  # std_msgs/String: JSON-encoded {header, point,
                                                # keypoints} -- see ros_communication.py's
                                                # publish_human_location() docstring for why this
                                                # isn't a stock geometry_msgs/PointStamped (that
                                                # type has no room for the keypoints dict).

    # Two RTDE readers coexist and each names its own topics -- neither set is dead, so
    # both stay. 4_execution/eval/read_ur_live_data.py publishes the first three;
    # 4_execution/eval/ur_state_reader.py publishes the rest, and also "robot_position"
    # above (which the ur_robot_driver ROS node feeds otherwise).
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
    TASK_PULL_CABLES: {
        "suggested_action": "pull_cables",
    },
    TASK_LEAVE_HANDOVER: {
        "suggested_action": "leave_handover",
    },
}

# -- Task detectors (2_decision_making/task_transition_detector.py) --
# "off": not run. "log": run, and log what they would report without changing
# anything -- keep this until their thresholds are tuned. "on": their signals count.
TASK_DETECTORS_MODE = "log"
# Screw count, from recognition's Screw progress: per screw it climbs and drops back
# near 0 right as that screw ends. Climbing to HIGH then falling to LOW counts one
# screw; counts closer than MIN_INTERVAL apart are one. First tuned on two annotated
# cam-05 takes with model 3d_skeleton_01 (0.30 / 0.20: real screws peaked at 0.36-0.62,
# dipped to 0.20 at most, and ended at least 5.3 s apart); HIGH raised to 0.40 for the
# multi-head models' per-step progress lanes.
SCREWS_PER_PANEL = 6
SCREW_PROGRESS_HIGH = 0.40
SCREW_PROGRESS_LOW = 0.20
SCREW_MIN_INTERVAL_S = 3.0
# Connect Cables is done once its progress climbs to HIGH and falls back to LOW (or
# recognition moves on to another step while it is up).
CONNECT_CABLES_DONE_HIGH = 0.60
CONNECT_CABLES_DONE_LOW = 0.15
# Clamp Coupling the same way, from its own progress -- or recognition moving on to
# another step for SEQUENCE_CONFIRM_EVENTS updates in a row while it is up, since its
# progress starts up high. Its progress starts at ~0.5 once the clamping is recognized,
# peaks at ~0.55 and falls back as it ends (run 2026-10-01 09:59, model S3_10fps_8s_bg05:
# 0.50 -> 0.57 -> 0.18 over 15 s; tune on more takes). Not by time, nor by the next
# panel's task being recognized: those marked it done while nobody clamped -- 14.2 s
# after Connect Cables, recognition showing "Non Related Task" throughout -- and again
# the moment the operator reset it, whenever the human was on Pull Cables (run
# 2026-10-01 10:30). If recognition never shows it, "clamped" / "clamp done" (or the
# live view) says it is done; until then the human stays on that piece.
CLAMP_COUPLING_DONE_HIGH = 0.50
CLAMP_COUPLING_DONE_LOW = 0.25
# The wrench the force detectors read: a ROS_TOPICS key (geometry_msgs/WrenchStamped),
# published by 4_execution/eval/read_ur_live_data.py --publish-ros -- which
# run_system.py starts in its own window (--no-robot-live to skip it).
FORCE_WRENCH_TOPIC = "ur_tcp_force"
# How often that reader publishes the wrench to rosbridge (Hz).
ROBOT_LIVE_DATA_ROS_HZ = 50.0
# ScrewingMonitor thresholds (4_execution/src/force_monitors.py). Placeholders: tune
# them on recorded trials with 4_execution/eval/force_logger.py and force_log_review.py.
SCREWING_THRESHOLDS = {
    "active_force_n": 8.0,
    "quiet_force_n": 3.0,
    "min_active_s": 2.0,
    "quiet_s": 2.0,
    "max_push_s": 8.0,
}
# What lets the robot leave the held panel (the "panel secured" signal: it asks to
# release it and leave) -- not Screw being done, which recognition or the human's
# "screw done" decides. Any one of: every screw counted (SCREWS_PER_PANEL), the
# pushes on the panel over, or the panel's weight steady on the frame (below).
# Before the pushes being over count, at least this share of the screws must be
# counted (0.5 = 3 of 6). None: the force alone decides.
SCREW_DONE_MIN_PROGRESS = 0.5
# The load on the TCP changing by at least this much while holding (N) is the "TCP
# weight change" signal: the panel's weight going over to the frame.
TCP_WEIGHT_CHANGE_N = 5.0
# The panel is also secured -- even with no screws counted by recognition -- once the
# load change since the hold began stays within this range (low, high) in N, with no
# push, for TCP_WEIGHT_STEADY_S in a row: the frame carries the panel. None: only the
# screw count and the pushes decide.
# Still to tune on recorded takes (force_logger.py). The one measurement so far (run
# 2026-09-28 13:53, piece 1): the panel's weight going over to the frame changed the load
# by 45 N -- outside the old (5, 40), so the leave was only asked 5 minutes later.
TCP_WEIGHT_RANGE_N = (5.0, 60.0)
TCP_WEIGHT_STEADY_S = 10.0

# -- Demo opening (run_communication.py --demo; 2_decision_making/demo_opening.py) --
# The first panel starts from the dialogue, not recognition: the robot offers to pull
# the cables, then asks about the lift.
# The robot's cable pull takes a fixed time: ask about the lift this long after it
# starts, a little before it ends, so a yes lifts the panel right after the pull.
# PLACEHOLDER: set from the robot program's pull duration.
DEMO_LIFT_ASK_AFTER_PULL_START_S = 45.0
# When the human pulls the cables instead, ask about the lift after Pull Cables'
# duration limit (TASK_OVERRUN_STAT of its annotated duration, p95 = 6 s) plus this.
DEMO_HUMAN_PULL_BUFFER_S = 10.0
# The task detectors (screw detection) in the demo: their signals count.
DEMO_DETECTORS_MODE = "on"

# -- Reactive mode (run_communication.py --reactive; 2_decision_making/reactive_task_manager.py) --
# The robot offers nothing: it acts only when the human commands it ("pull the cables",
# "lift the panel", "give me the connector", "leave"), and the command itself is the
# go-ahead. No camera and no TCP force: the task tracker follows the robot's tasks and
# what the human says, so a command is carried out on the right piece.
# After handing an item over, the robot leaves the hand-over position this many seconds
# later without asking -- for items HANDOVER_LEAVE_DELAY_S has no delay for ("cancel"
# keeps it there).
REACTIVE_HANDOVER_LEAVE_DELAY_S = 1.0

UDP_HOST = "127.0.0.1"
UDP_PORT = 5006

EVENT_TRANSPORT_HOST = "127.0.0.1"
EVENT_TRANSPORT_PORT = 5010

# Communication's event log when a run is not logged to its own directory.
LOG_FILE_PATH = "hrc_communication_events.log"
# Each run of run_system.py / run_communication.py / run_recognition.py logs to
# <RUN_LOG_DIR>/<run name>/ (relative to the repository root; --no-run-log turns it
# off). run_system.py gives both processes the same run name, so one directory holds
# recognition's frames.csv / events.csv / run.json / run.log and communication's
# communication_events.jsonl / timeline.csv, all on the same clock.
RUN_LOG_DIR = "logs/runs"

# Live view of the decision layer -- task pool, robot queue, why each trigger rule
# offers or not, a timeline -- at http://<host>:<port>/ while the communication
# runtime runs (2_decision_making/decision_view.py). None turns it off.
DECISION_VIEW_HOST = "127.0.0.1"
DECISION_VIEW_PORT = 8770

VOICE_ENABLED = True

VOICE_GPT_ENABLED = False

# List audio devices: uv run python -m sounddevice
VOICE_MODEL_PATH = "3_communication/vosk_fallback/models/vosk-model-small-en-us-0.15"
# Left at None, sounddevice falls back to the OS default input, which on this
# laptop is a virtual NDI webcam audio device (no real signal) rather than the
# physical mic -- pin it explicitly. Run `uv run python -m sounddevice` to list
# devices and update this if the laptop's mic name/index differs.
VOICE_INPUT_DEVICE_NAME = None
VOICE_OUTPUT_DEVICE_NAME = None
VOICE_LISTEN_TIMEOUT_SECONDS = 8.0
VOICE_TTS_RATE = 220
VOICE_POST_TTS_GUARD_SECONDS = 0.2
VOICE_MAX_ATTEMPTS = 2
VOICE_ERROR_RETRY_SECONDS = 5.0

test_vid_path = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\raw\cam-04\video__cam-04_uid-01_take-01.mp4"


# The UR10e on the lab network. URSim in Docker: "127.0.0.1" (ports 30001-30004 are
# published to the host).
# ROBOT_IP = "169.254.130.206"
ROBOT_IP = "192.168.1.10"