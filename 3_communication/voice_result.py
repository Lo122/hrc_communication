"""Non-command outcomes shared by speech backends and communication policy."""

from enum import Enum, auto


class VoiceOutcome(Enum):
    NO_SPEECH = auto()
    UNRECOGNIZED = auto()
    ERROR = auto()
