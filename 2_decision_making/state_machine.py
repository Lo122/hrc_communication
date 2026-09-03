"""Robot task transition validity table."""

from events import EventType, RobotTaskState


class StateMachine:
    """Answers whether an event is valid and what state follows."""

    _TRANSITIONS = {
        (RobotTaskState.R_WAITING_RESPONSE, EventType.H_ACCEPT): RobotTaskState.R_ACCEPTED,
        (RobotTaskState.R_WAITING_RESPONSE, EventType.H_REFUSE): RobotTaskState.R_REFUSED,
        (RobotTaskState.R_WAITING_RESPONSE, EventType.H_DEFER): RobotTaskState.R_DEFER,
        (RobotTaskState.R_WAITING_RESPONSE, EventType.RESPONSE_TIMEOUT): RobotTaskState.R_PENDING,
        (RobotTaskState.R_DEFER, EventType.DEFER_TIMEOUT): RobotTaskState.R_ACCEPTED,
        (RobotTaskState.R_DEFER, EventType.H_CANCEL): RobotTaskState.R_CANCELED,
        (RobotTaskState.R_ACCEPTED, EventType.H_CANCEL): RobotTaskState.R_RECOVERY_EVALUATING,
        (RobotTaskState.R_REDO, EventType.H_CANCEL): RobotTaskState.R_RECOVERY_EVALUATING,
        (RobotTaskState.R_ACCEPTED, EventType.ROBOT_RUNNING): RobotTaskState.R_EXECUTING,
        (RobotTaskState.R_REDO, EventType.ROBOT_RUNNING): RobotTaskState.R_EXECUTING,
        (RobotTaskState.R_EXECUTING, EventType.ROBOT_RUNNING): RobotTaskState.R_EXECUTING,
        (RobotTaskState.R_EXECUTING, EventType.H_PAUSE): RobotTaskState.R_PAUSED,
        (RobotTaskState.R_PAUSED, EventType.H_RESUME): RobotTaskState.R_EXECUTING,
        (RobotTaskState.R_EXECUTING, EventType.H_RESTART): RobotTaskState.R_REDO,
        (RobotTaskState.R_PAUSED, EventType.H_RESTART): RobotTaskState.R_REDO,
        (RobotTaskState.R_EXECUTING, EventType.H_CANCEL): RobotTaskState.R_RECOVERY_EVALUATING,
        (RobotTaskState.R_PAUSED, EventType.H_CANCEL): RobotTaskState.R_RECOVERY_EVALUATING,
        (RobotTaskState.R_EXECUTING, EventType.H_SPEEDUP): RobotTaskState.R_EXECUTING,
        (RobotTaskState.R_EXECUTING, EventType.H_SLOWDOWN): RobotTaskState.R_EXECUTING,
        (RobotTaskState.R_EXECUTING, EventType.H_DONE): RobotTaskState.R_EXECUTING,
        (RobotTaskState.R_EXECUTING, EventType.ROBOT_SUCCESS): RobotTaskState.R_DONE,
        (RobotTaskState.R_PAUSED, EventType.ROBOT_SUCCESS): RobotTaskState.R_DONE,
        (RobotTaskState.R_WAITING_FREE_DRIVE, EventType.H_FREE_GO): RobotTaskState.R_FREE_DRIVE,
        (RobotTaskState.R_WAITING_FREE_DRIVE, EventType.H_REFUSE): RobotTaskState.R_DONE,
        (RobotTaskState.R_WAITING_FREE_DRIVE, EventType.H_CANCEL): RobotTaskState.R_CANCELED,
        (RobotTaskState.R_FREE_DRIVE, EventType.H_DONE): RobotTaskState.R_DONE,
        (RobotTaskState.R_FREE_DRIVE, EventType.H_CANCEL): RobotTaskState.R_CANCELED,
        (RobotTaskState.R_RECOVERY_EVALUATING, EventType.RECOVERY_HOME_AVAILABLE): RobotTaskState.R_WAITING_HOME_PERMISSION,
        (RobotTaskState.R_RECOVERY_EVALUATING, EventType.RECOVERY_MANUAL_REQUIRED): RobotTaskState.R_MANUAL_RECOVERY,
        (RobotTaskState.R_WAITING_HOME_PERMISSION, EventType.H_RETURN_HOME): RobotTaskState.R_RETURNING_HOME,
        (RobotTaskState.R_WAITING_HOME_PERMISSION, EventType.H_MANUAL_RECOVERY): RobotTaskState.R_MANUAL_RECOVERY,
        (RobotTaskState.R_WAITING_HOME_PERMISSION, EventType.H_REFUSE): RobotTaskState.R_MANUAL_RECOVERY,
        (RobotTaskState.R_MANUAL_RECOVERY, EventType.H_DONE): RobotTaskState.R_CANCELED,
        (RobotTaskState.R_MANUAL_RECOVERY, EventType.H_CANCEL): RobotTaskState.R_CANCELED,
        (RobotTaskState.R_RETURNING_HOME, EventType.ROBOT_HOMED): RobotTaskState.R_CANCELED,
        # HOLD WHEN DISASSEMBLE is a special case that can happen at any time, so we allow it from any state.
        # (RobotTaskState.R_EXECUTING, EventType.H_DISASSEMBLE): RobotTaskState.R_HOLDING_WHEN_DISASSEMBLE,
    }

    def is_valid_transition(
        self,
        current_state: RobotTaskState,
        event_type: EventType,
    ) -> bool:
        return (current_state, event_type) in self._TRANSITIONS

    def get_next_state(
        self,
        current_state: RobotTaskState,
        event_type: EventType,
    ) -> RobotTaskState | None:
        return self._TRANSITIONS.get((current_state, event_type))
