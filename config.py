"""Configuration values and mappings for the HRC communication system."""

RESPONSE_TIMEOUT_SECONDS = 5.0
DEFER_SECONDS = 5.0

DEFAULT_SPEED = 0.5
SPEED_STEP = 0.1
MIN_SPEED = 0.1
MAX_SPEED = 1.0

TASK_LIFT_PANEL = 0
TASK_BRING_JOINT = 1
TASK_BRING_NEXT_PANEL = 2

TRIGGER_RULES = {
    TASK_LIFT_PANEL: {
        "step_id": 0,
        "progress_threshold": 0.5,
        "min_confidence": 0.8,
    },
    TASK_BRING_JOINT: {
        "step_id": 2,
        "progress_threshold": 0.8,
        "min_confidence": 0.8,
    },
}

PERMISSION_MESSAGES = {
    TASK_LIFT_PANEL: "Would you like me to lift the panel?",
    TASK_BRING_JOINT: "Would you like me to bring the joint piece?",
}

ROS_TOPICS = {
    "pause": "/robot/pause",
    "resume": "/robot/resume",
    "restart": "/robot/restart",
    "cancel": "/robot/cancel",
    "speed": "/robot/speed",
    "human_done": "/human/done",
    "robot_success": "/robot/success",
}

UDP_HOST = "127.0.0.1"
UDP_PORT = 9000

LOG_FILE_PATH = "hrc_communication_events.log"

