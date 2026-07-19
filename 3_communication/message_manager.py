"""Human-facing message text."""

from config import PERMISSION_MESSAGES
from events import EventType, RobotTaskState


class MessageManager:
    """Centralizes CLI text without printing or changing state."""

    def get_permission_message(self, step_id: int) -> str:
        return PERMISSION_MESSAGES.get(step_id, f"Would you like me to execute step {step_id}?")

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
        }
        return messages.get(event_type, "Command processed.")

    def get_invalid_event_message(
        self,
        current_state: RobotTaskState | None,
        event_type: EventType,
    ) -> str:
        state_name = current_state.name if current_state is not None else "no active task"
        return f"Cannot process {event_type.name} while in {state_name}."
