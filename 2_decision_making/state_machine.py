"""Robot task transition validity table."""

import config
from events import EventType, RobotTaskState


class StateMachine:
    """Answers whether an event is valid and what state follows."""

    _TRANSITIONS = {
        (RobotTaskState.R_WAITING_RESPONSE, EventType.H_ACCEPT): RobotTaskState.R_ACCEPTED,
        # No: the human does the task (or, one only the robot can do, it waits pending).
        (RobotTaskState.R_WAITING_RESPONSE, EventType.H_REFUSE): RobotTaskState.R_REFUSED,
        # Later: pending until the human asks for it -- no timer.
        (RobotTaskState.R_WAITING_RESPONSE, EventType.H_DEFER): RobotTaskState.R_PENDING,
        (RobotTaskState.R_WAITING_RESPONSE, EventType.RESPONSE_TIMEOUT): RobotTaskState.R_PENDING,
        # The workflow's own delayed start (leaving the hand-over position).
        (RobotTaskState.R_WAITING_RESPONSE, EventType.DELAYED_START): RobotTaskState.R_DEFER,
        # The human did the offered task themselves; the offer is withdrawn.
        (RobotTaskState.R_WAITING_RESPONSE, EventType.H_TASK_DONE): RobotTaskState.R_CANCELED,
        # A pending task is offered again, never dispatched without a fresh H_ACCEPT.
        (RobotTaskState.R_REFUSED, EventType.H_EXECUTE_PENDING_TASK): RobotTaskState.R_WAITING_RESPONSE,
        (RobotTaskState.R_PENDING, EventType.H_EXECUTE_PENDING_TASK): RobotTaskState.R_WAITING_RESPONSE,
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
        (RobotTaskState.R_WAITING_FREE_DRIVE, EventType.H_ACCEPT): RobotTaskState.R_FREE_DRIVE,
        (RobotTaskState.R_WAITING_FREE_DRIVE, EventType.H_REFUSE): RobotTaskState.R_HOLDING,
        (RobotTaskState.R_WAITING_FREE_DRIVE, EventType.H_CANCEL): RobotTaskState.R_CANCELED,
        (RobotTaskState.R_FREE_DRIVE, EventType.H_DONE): RobotTaskState.R_HOLDING,
        (RobotTaskState.R_FREE_DRIVE, EventType.H_CANCEL): RobotTaskState.R_CANCELED,
        (RobotTaskState.R_HOLDING, EventType.H_SCREW_DONE): RobotTaskState.R_DONE,
        # A detector says the held panel is secured, or the human says "leave": the
        # robot may release it.
        (RobotTaskState.R_HOLDING, EventType.TASK_SIGNAL): RobotTaskState.R_DONE,
        (RobotTaskState.R_HOLDING, EventType.H_REQUEST_ROBOT_TASK): RobotTaskState.R_DONE,
        (RobotTaskState.R_HOLDING, EventType.H_CANCEL): RobotTaskState.R_RECOVERY_EVALUATING,
        # Handing over a brought item: the gripper opens on a yes or when asked for it.
        (RobotTaskState.R_WAITING_HANDOVER, EventType.H_ACCEPT): RobotTaskState.R_DONE,
        (RobotTaskState.R_WAITING_HANDOVER, EventType.H_HANDOVER): RobotTaskState.R_DONE,
        (RobotTaskState.R_WAITING_HANDOVER, EventType.H_REFUSE): RobotTaskState.R_HOLDING_HANDOVER,
        (RobotTaskState.R_WAITING_HANDOVER, EventType.H_CANCEL): RobotTaskState.R_RECOVERY_EVALUATING,
        (RobotTaskState.R_HOLDING_HANDOVER, EventType.H_HANDOVER): RobotTaskState.R_DONE,
        (RobotTaskState.R_HOLDING_HANDOVER, EventType.H_CANCEL): RobotTaskState.R_RECOVERY_EVALUATING,
        (RobotTaskState.R_RECOVERY_EVALUATING, EventType.RECOVERY_HOME_AVAILABLE): RobotTaskState.R_WAITING_HOME_PERMISSION,
        (RobotTaskState.R_RECOVERY_EVALUATING, EventType.RECOVERY_MANUAL_REQUIRED): RobotTaskState.R_MANUAL_RECOVERY,
        (RobotTaskState.R_WAITING_HOME_PERMISSION, EventType.H_RETURN_HOME): RobotTaskState.R_RETURNING_HOME,
        (RobotTaskState.R_WAITING_HOME_PERMISSION, EventType.H_ACCEPT): RobotTaskState.R_RETURNING_HOME,
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
        task_id: int | None = None,
    ) -> bool:
        return self.get_next_state(current_state, event_type, task_id) is not None

    def events_from(self, current_state: RobotTaskState,
                    task_id: int | None = None) -> dict[str, str]:
        """{event name: next state name} for every event the state accepts -- what a
        task in this state waits for (the live view shows it)."""
        return {event_type.name: next_state.name for event_type in EventType
                if (next_state := self.get_next_state(current_state, event_type, task_id)) is not None}

    def get_next_state(
        self,
        current_state: RobotTaskState,
        event_type: EventType,
        task_id: int | None = None,
    ) -> RobotTaskState | None:
        if event_type == EventType.ROBOT_SUCCESS and current_state in {RobotTaskState.R_EXECUTING,
                                                                       RobotTaskState.R_PAUSED}:
            if task_id == config.TASK_LIFT_PANEL:
                return (RobotTaskState.R_WAITING_FREE_DRIVE if config.LIFT_ASKS_FREE_DRIVE
                        else RobotTaskState.R_FREE_DRIVE)
            if task_id in config.HANDOVER_ITEMS:
                return RobotTaskState.R_WAITING_HANDOVER
        if current_state == RobotTaskState.R_DEFER and event_type == EventType.H_CANCEL:
            if task_id == config.TASK_LEAVE:
                return RobotTaskState.R_HOLDING
            if task_id == config.TASK_LEAVE_HANDOVER:
                # Not leaving after all: the robot stays, as after a no.
                return RobotTaskState.R_REFUSED
        return self._TRANSITIONS.get((current_state, event_type))
