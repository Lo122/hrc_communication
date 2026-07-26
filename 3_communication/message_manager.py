"""Human-facing message text."""

from config import PERMISSION_MESSAGES
from events import EventType, RobotTaskState


class MessageManager:
    """Centralizes CLI text without printing or changing state."""

    def get_permission_message(self, step_id: int) -> str:
        return PERMISSION_MESSAGES.get(step_id, f"Would you like me to execute step {step_id}?")

    def get_execution_message(self, task) -> str:
        actions = ["pause", "cancel", "restart", "faster", "slower"]
        message = "Robot task is executing. You can type: " + ", ".join(actions) + "."
        return message

    def ask_permission_for_free_drive(self) -> str:
        return 'Robot task finished. Enter free drive mode for manual adjustment? Type "free drive" to confirm or "no" to finish.'

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
            EventType.H_DONE: "Human-done signal sent.",
            EventType.ROBOT_SUCCESS: "Robot task completed.",
            EventType.ROBOT_HOMED: "Robot returned to its home position.",
            EventType.H_FREE_GO: "Now free drive mode is on, and you can move the robot to align the piece; let me know when you are done.",
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
