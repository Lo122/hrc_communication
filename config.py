"""Configuration values and mappings for the HRC communication system."""

RESPONSE_TIMEOUT_SECONDS = 20.0
DEFER_SECONDS = 5.0
RECOVERY_STOP_DELAY_SECONDS = 0.5

DEFAULT_SPEED = 0.2
SPEED_STEP = 0.1

MIN_SPEED = 0.0000001
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
    (-3, 3),  # joint 5
    (-3, 3),  # joint 6
]

# ---------------------------------------------------------------------------
# Workflow
#
# The assembly sequence, the proposals, the GH actions and the trigger
# thresholds all live in 0_core/task_catalog.py now -- one table instead of
# three parallel dicts that had to be kept in sync by hand. What is left here
# is the derivation, so existing imports keep working.
# ---------------------------------------------------------------------------

import sys as _sys
from pathlib import Path as _Path

_CORE = _Path(__file__).resolve().parent / "0_core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

import task_catalog  # noqa: E402

TASKS = task_catalog.TASKS

# step id -> {"progress_threshold", "min_confidence", "task_key"}
TRIGGER_RULES = task_catalog.trigger_rules()

# Kept so nothing that still imports these breaks. Prefer task_catalog.
STEP_PANEL_CUT = task_catalog.STEP_PANEL_CUT
STEP_PULL_CABLE = task_catalog.STEP_PULL_CABLE
STEP_ALIGN = task_catalog.STEP_ALIGN
STEP_SCREW = task_catalog.STEP_SCREW
STEP_CUT_PIPE = task_catalog.STEP_CUT_PIPE
STEP_CONNECT_PIPE = task_catalog.STEP_CONNECT_PIPE
STEP_CLAMP_JOINT = task_catalog.STEP_CLAMP_JOINT

# How long the robot waits in R_HOLDING before reminding the human it is
# still under load. It never releases on its own -- a timeout that dropped a
# panel would be the worst possible failure mode.
HOLD_REMINDER_SECONDS = 25.0

# A cancel on a load-bearing task (see task_catalog.carries_load) has to be
# said twice, within this many seconds. The microphone is open while the
# robot holds a panel, so one overheard word must not make it let go.
CANCEL_CONFIRM_WINDOW_SECONDS = 8.0

# How long an announced task waits before it goes ahead, giving the human
# time to say stop. Long enough to react to speech, short enough that the
# robot does not feel hesitant.
ANNOUNCE_VETO_SECONDS = 4.0

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
    "human_position": "/Human/position/live",  # std_msgs/String: JSON-encoded {header, point,
                                                # keypoints} -- see ros_communication.py's
                                                # publish_human_location() docstring for why this
                                                # isn't a stock geometry_msgs/PointStamped (that
                                                # type has no room for the keypoints dict).
}

# How often recognition publishes HUMAN_LOCATION_UPDATE events, in frames -- see
# run_recognition.py's main loop. 5 -> ~5-6Hz at a 30fps camera, well above what the
# discrete task-state events on this bus were designed for at full frame rate.
HUMAN_LOCATION_PUBLISH_EVERY_N_FRAMES = 5

# Grasshopper payload per task, derived from the catalog. The dispatcher
# looks tasks up by task_key now, not by step id -- chained tasks share a
# step id with whatever triggered the sequence.
GH_TASK_MESSAGES = {
    key: {"suggested_action": spec.gh_action}
    for key, spec in task_catalog.TASKS.items()
}

# Legacy shape, still keyed by the step that triggers each task, for anything
# that has not moved to task_key yet.
GH_STEP_MESSAGES = {
    step_id: {"suggested_action": spec.gh_action}
    for step_id, spec in task_catalog.triggered_tasks().items()
}

PERMISSION_MESSAGES = {
    step_id: spec.proposal
    for step_id, spec in task_catalog.triggered_tasks().items()
}

UDP_HOST = "127.0.0.1"
UDP_PORT = 5006

EVENT_TRANSPORT_HOST = "127.0.0.1"
EVENT_TRANSPORT_PORT = 5010

LOG_FILE_PATH = "hrc_communication_events.log"

VOICE_ENABLED = True

VOICE_GPT_ENABLED = True

VOICE_MODEL_PATH = "3_communication/vosk_fallback/models/vosk-model-small-en-us-0.15"
VOICE_INPUT_DEVICE_NAME = None
# Derived from RESPONSE_TIMEOUT_SECONDS so the microphone stays open for
# exactly as long as the task is actually waiting for an answer.
VOICE_LISTEN_TIMEOUT_SECONDS = RESPONSE_TIMEOUT_SECONDS
VOICE_TTS_RATE = 175
VOICE_POST_TTS_GUARD_SECONDS = 1.0
VOICE_MAX_ATTEMPTS = 2

test_vid_path = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\raw\cam-04\video__cam-04_uid-01_take-01.mp4"
