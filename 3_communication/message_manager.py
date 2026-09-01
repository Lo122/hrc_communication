"""Everything the robot says to the human.

The old version mapped an EventType to one fixed sentence, so it could
never name the piece, the task, or what happens next -- "Robot task
accepted." tells a worker on a ladder nothing useful.

Every method here now takes the task when there is one, so sentences can
carry three things a spoken interface needs:

  * what just happened, in the worker's words rather than state names
  * what the robot is doing about it
  * what the human is expected to do next, if anything

Sentences are written to be HEARD, not read: short, front-loaded, no
state names, no ids read aloud.
"""

import task_catalog
from events import EventType, RobotTaskState


def _spec(task):
    """The catalog entry for a task, or None if it predates the catalog."""
    if task is None:
        return None
    return task_catalog.task_by_key(getattr(task, "task_key", "") or "")


class MessageManager:
    """Builds human-facing text. Never prints, never changes state."""

    # -- proposals ---------------------------------------------------------

    def get_permission_message(self, step_id: int) -> str:
        """Kept for callers that still only know the step id."""
        spec = task_catalog.triggered_tasks().get(step_id)
        if spec is None:
            return f"Would you like me to help with {task_catalog.step_name(step_id)}?"
        return spec.proposal

    def get_task_proposal(self, task) -> str:
        spec = _spec(task)
        if spec is None:
            return self.get_permission_message(task.step_id)
        return spec.proposal

    def get_proposal_reason(self, task) -> str:
        """Why the robot is offering now.

        Not spoken by default -- it is what the human gets when they ask
        "why", and what the operator dashboard shows. A proposal a worker
        cannot interrogate is a proposal they stop trusting.
        """
        spec = _spec(task)
        if spec is None or not spec.reason:
            return "I do not have a reason recorded for this one."
        return f"I offered because {spec.reason}."

    # -- acknowledgements --------------------------------------------------

    def get_acknowledgement(self, event_type: EventType, task=None) -> str:
        spec = _spec(task)

        if event_type == EventType.H_ACCEPT and spec is not None:
            return spec.on_accept
        if event_type == EventType.ROBOT_SUCCESS and spec is not None:
            return spec.on_success
        if event_type == EventType.H_REFUSE and spec is not None:
            return f"Fine. {spec.label.capitalize()} is in the queue if you change your mind."
        if event_type == EventType.H_DEFER and spec is not None:
            return f"Waiting a moment, then {spec.label}."

        messages = {
            EventType.H_ACCEPT: "Okay, starting.",
            EventType.H_REFUSE: "Fine, I will keep it in the queue.",
            EventType.H_DEFER: "Waiting a moment.",
            EventType.H_PAUSE: "Stopped. Say resume when you are ready.",
            EventType.H_RESUME: "Carrying on.",
            EventType.H_RESTART: "Starting that again from the beginning.",
            EventType.H_CANCEL: "Cancelled.",
            EventType.H_SPEEDUP: "Speeding up.",
            EventType.H_SLOWDOWN: "Slowing down.",
            EventType.H_DONE: "Got it.",
            EventType.H_SECURED: "Good. Letting go now.",
            EventType.H_NOT_YET: "No problem, still holding.",
            EventType.ROBOT_SUCCESS: "Done.",
            EventType.ROBOT_HOMED: "I am back at home position.",
            EventType.H_FREE_GO: "Arm is soft. Move it where you want it, then say done.",
            EventType.H_RETURN_HOME: "Going back to home position.",
            EventType.H_MANUAL_RECOVERY: "Arm is loose. Move it clear by hand and say done.",
        }
        return messages.get(event_type, "Okay.")

    # -- the hold-until-secured gate --------------------------------------

    def get_hold_message(self, task) -> str:
        """Said the moment the robot takes the load."""
        spec = _spec(task)
        if spec is not None and spec.hold_prompt:
            return spec.hold_prompt
        return "I am holding it. Say secured when it is fixed and I can let go."

    def get_hold_reminder(self, task) -> str:
        """Repeated while the robot is still under load.

        The robot never releases on a timer. The only thing a timeout does
        is ask again.
        """
        return "Still holding. Say secured when I can let go, or cancel to stop."

    def get_release_message(self, task) -> str:
        return "Letting go now. Stand clear of the panel."

    # -- chaining ----------------------------------------------------------

    def get_followup_proposal(self, spec) -> str:
        return spec.proposal

    def get_announcement(self, spec) -> str:
        """Said when the robot proceeds without asking.

        It must still be obvious that stopping is possible -- an
        announcement the human does not know they can refuse is not an
        announcement, it is a robot doing what it likes.
        """
        return f"{spec.on_accept} Say no or stop to hold me off."

    # -- state-specific prompts -------------------------------------------

    def get_execution_message(self, task) -> str:
        spec = _spec(task)
        what = spec.label if spec is not None else "the task"
        return f"Running: {what}. You can say pause, faster, slower or cancel."

    def ask_permission_for_free_drive(self) -> str:
        # The task itself was already accepted; this is the brake-release
        # confirmation, not a second proposal, so it gets different words.
        return "I am in position. Say free drive and I will go soft, or no to leave it rigid."

    def get_return_home_permission_message(self) -> str:
        return "I stopped in a safe spot. Shall I go back to home position? Say home, or no to move me by hand."

    def get_manual_recovery_message(self) -> str:
        return "Arm is loose. Move it clear by hand and say done when it is out of the way."

    def get_pending_summary(self, tasks) -> str:
        """What the human hears when they ask what is queued."""
        if not tasks:
            return "Nothing is queued."
        labels = []
        for task in tasks:
            spec = _spec(task)
            labels.append(spec.label if spec is not None else "a task")
        if len(labels) == 1:
            return f"One thing queued: {labels[0]}. Say go to run it."
        listed = ", ".join(labels[:-1]) + f" and {labels[-1]}"
        return f"{len(labels)} things queued: {listed}. Say go to run the first one."

    def get_status_message(self, task) -> str:
        """Answer to "what are you doing" -- the single most likely question
        from someone who was looking at the ceiling and missed a prompt."""
        if task is None:
            return "I am idle and waiting."
        spec = _spec(task)
        what = spec.label if spec is not None else "a task"
        readable = {
            RobotTaskState.R_WAITING_RESPONSE: f"I asked whether I should be {what}. Say yes or no.",
            RobotTaskState.R_ACCEPTED: f"Starting {what}.",
            RobotTaskState.R_EXECUTING: f"I am {what}.",
            RobotTaskState.R_PAUSED: f"I stopped partway through {what}. Say resume.",
            RobotTaskState.R_HOLDING: "I am holding the panel. Say secured when I can let go.",
            RobotTaskState.R_FREE_DRIVE: "The arm is soft. Move it by hand, then say done.",
            RobotTaskState.R_DEFER: f"About to start {what}.",
            RobotTaskState.R_WAITING_FREE_DRIVE: "Waiting to know whether you want to align by hand.",
            RobotTaskState.R_WAITING_HOME_PERMISSION: "Waiting to know whether I should return home.",
            RobotTaskState.R_MANUAL_RECOVERY: "Waiting for you to move me clear by hand.",
            RobotTaskState.R_RETURNING_HOME: "Going back to home position.",
        }
        return readable.get(task.state, f"I am busy with {what}.")

    def get_repeat_message(self, last_message: str | None) -> str:
        if not last_message:
            return "I have not said anything yet."
        return last_message

    # -- confirmation for destructive commands ----------------------------

    def get_cancel_confirmation(self, task) -> str:
        spec = _spec(task)
        what = spec.label if spec is not None else "the task"
        if task is not None and task.state == RobotTaskState.R_HOLDING:
            return (
                f"I am holding the panel. Cancelling means letting go. "
                "Say cancel again to confirm."
            )
        return f"Cancel {what}? Say cancel again to confirm."

    # -- errors ------------------------------------------------------------

    def get_invalid_event_message(self, current_state, event_type: EventType) -> str:
        """Say what CAN be done, not which state name was violated."""
        if current_state is None:
            return "Nothing is running right now."
        options = {
            RobotTaskState.R_WAITING_RESPONSE: "say yes, no or later",
            RobotTaskState.R_EXECUTING: "say pause, faster, slower or cancel",
            RobotTaskState.R_PAUSED: "say resume, restart or cancel",
            RobotTaskState.R_HOLDING: "say secured, or cancel",
            RobotTaskState.R_FREE_DRIVE: "move the arm by hand, then say done",
            RobotTaskState.R_MANUAL_RECOVERY: "move the arm clear, then say done",
            RobotTaskState.R_WAITING_FREE_DRIVE: "say yes or no",
            RobotTaskState.R_WAITING_HOME_PERMISSION: "say home or no",
        }
        can_do = options.get(current_state)
        if can_do is None:
            return "I cannot do that right now."
        return f"I cannot do that right now. You can {can_do}."
