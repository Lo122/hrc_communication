"""Configuration values and mappings for the HRC communication system."""

RESPONSE_TIMEOUT_SECONDS = 20.0
DEFER_SECONDS = 5.0

DEFAULT_SPEED = 0.2
SPEED_STEP = 0.1

MIN_SPEED = 0.0000001
MAX_SPEED = 0.75

LAST_SPEED = DEFAULT_SPEED

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
    "pause": "/Robot/control", #pause
    "resume": "/Robot/control", #resume
    "restart": "/Robot/control", #not yet
    "cancel": "/Robot/control",#stop
    "global_speed": "/Robot/globalSpeed", #float64
    "local_speed":"/Robot/localSpeed",
    "human_done": "/Human/taskSuccess", #success
    "robot_success": "/Robot/status/physical", #success
    "free_drive": "/Robot/teachMode",
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
