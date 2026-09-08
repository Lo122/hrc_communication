"""Human-facing message text."""

import config
from events import EventType, RobotTaskState


class MessageManager:
    """Centralizes CLI text without printing or changing state."""

    def get_permission_message(self, task_id: int) -> str:
        return config.PERMISSION_MESSAGES[task_id]

    def get_execution_message(self, task) -> str:
        messages = {
            config.TASK_LIFT_PANEL: "I am lifting the panel. I will ask about manual adjustment when the lift is complete.",
            config.TASK_LEAVE: "I am releasing the panel and moving away.",
            config.TASK_BRING_CONNECTOR: "I am bringing the pipe connector.",
            config.TASK_BRING_CLAMPING_TOOL: "I am bringing the clamping tool.",
            config.TASK_RETURN_CLAMPING_TOOL: "I am taking the clamping tool back.",
        }
        return messages[task.task_id] + " Say or type pause, cancel, restart, faster, or slower."

    def ask_permission_for_free_drive(self) -> str:
        return 'The panel is lifted. Would you like free drive for manual adjustment? Say yes or no after the beep, or type your reply. You can also say free drive.'

    def get_holding_message(self) -> str:
        return 'I will keep holding the panel. When screwing is finished, say or type "screw done". I will then ask before releasing the panel.'

    def get_pending_message(self, task) -> str:
        reason = "No reply received." if task.pending_reason == "timeout" else "Okay."
        holding = " I will keep holding the panel." if task.task_id == config.TASK_LEAVE else ""
        return f'{reason} This task is pending.{holding} To start it when ready, type "execute {task.task_instance_id}".'

    def get_defer_message(self, task, duration: float) -> str:
        holding = " I will keep holding the panel until then." if task.task_id == config.TASK_LEAVE else ""
        return f"Okay, I will start in {duration:g} seconds.{holding} Say or type cancel to cancel the delayed task."

    def get_return_home_permission_message(self) -> str:
        return 'Robot stopped in a validated recovery zone. Type "home" to return home or "no" for manual recovery.'

    def get_manual_recovery_message(self) -> str:
        return 'Free-drive mode is enabled. Complete the manual adjustment and type "done" when finished.'

    def get_acknowledgement(self, event_type: EventType) -> str:
        messages = {
            EventType.H_ACCEPT: "Robot task accepted.",
            EventType.H_REFUSE: "Robot task moved to pending.",
            EventType.H_DEFER: "Robot task deferred.",
            EventType.H_PAUSE: "The robot task has been paused.",
            EventType.H_RESUME: "The robot task has resumed.",
            EventType.H_RESTART: "The robot task is restarting.",
            EventType.H_CANCEL: "The robot task has been canceled.",
            EventType.H_SPEEDUP: "The robot speed has been increased.",
            EventType.H_SLOWDOWN: "The robot speed has been decreased.",
            EventType.H_DONE: "Human-done signal sent.",
            EventType.ROBOT_SUCCESS: "Robot task completed.",
            EventType.ROBOT_HOMED: "Robot returned to its home position.",
            EventType.H_FREE_GO: 'Free drive is on. Adjust the panel, then say or type "done". I will then keep holding the panel while you screw it in place.',
            EventType.H_RETURN_HOME: "The robot is returning to its home position.",
            EventType.H_MANUAL_RECOVERY: "Manual recovery started.",
        }
        return messages.get(event_type, "Command processed.")

    def get_invalid_event_message(
        self,
        current_state: RobotTaskState | None,
        event_type: EventType,
    ) -> str:
        state_name = current_state.name if current_state is not None else "no active task"
        return f"Cannot process {event_type.name} while in {state_name}."
