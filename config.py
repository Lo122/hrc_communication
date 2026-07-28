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
    (-1, 1),  # joint 1
    (-1, 1), # joint 2
    (-1, 1),  # joint 3
    (-1, 1),  # joint 4
    (-1, 1),  # joint 5
    (-1, 1),  # joint 6
]

STEP_LIFT_PANEL = 0
STEP_BRING_JOINT = 2
STEP_BRING_NEXT_PANEL = 6



TRIGGER_RULES = {
    STEP_LIFT_PANEL: {
        "progress_threshold": 0.1,
        "min_confidence": 0.1,
    },
    STEP_BRING_JOINT: {
        "progress_threshold": 0.1,
        "min_confidence": 0.1,
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
    "r_task_done": "/Task/signal",
}

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

test_vid_path = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\raw\cam-04\video__cam-04_uid-01_take-01.mp4"
