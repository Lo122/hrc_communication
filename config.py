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

STEP_LIFT_PANEL = 0
STEP_BRING_JOINT = 2
STEP_BRING_NEXT_PANEL = 6



TRIGGER_RULES = {
    STEP_LIFT_PANEL: {
        "progress_threshold": 0.5,
        "min_confidence": 0.8,
    },
    STEP_BRING_JOINT: {
        "progress_threshold": 0.8,
        "min_confidence": 0.8,
    },
}

PERMISSION_MESSAGES = {
    STEP_LIFT_PANEL: "Would you like me to lift the panel?",
    STEP_BRING_JOINT: "Would you like me to bring the joint piece?",
    STEP_BRING_NEXT_PANEL: "Would you like me to bring the next panel?",
}

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

GH_STEP_MESSAGES = {
    STEP_LIFT_PANEL: {
        "suggested_action": "assist_lifting",
    },
    STEP_BRING_JOINT: {
        "suggested_action": "bring_joint",
    },
    STEP_BRING_NEXT_PANEL: {
        "suggested_action": "bring_next_panel",
    },
}

UDP_HOST = "127.0.0.1"
UDP_PORT = 5006

EVENT_TRANSPORT_HOST = "127.0.0.1"
EVENT_TRANSPORT_PORT = 5010

LOG_FILE_PATH = "hrc_communication_events.log"

VOICE_ENABLED = True

VOICE_GPT_ENABLED = True

# List audio devices: .venv\Scripts\python.exe -m sounddevice
VOICE_MODEL_PATH = "3_communication/vosk_fallback/models/vosk-model-small-en-us-0.15"
VOICE_INPUT_DEVICE_NAME = None
VOICE_OUTPUT_DEVICE_NAME = None
VOICE_LISTEN_TIMEOUT_SECONDS = 8.0
VOICE_TTS_RATE = 175
VOICE_POST_TTS_GUARD_SECONDS = 1.0
VOICE_MAX_ATTEMPTS = 2

test_vid_path = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\raw\cam-04\video__cam-04_uid-01_take-01.mp4"
