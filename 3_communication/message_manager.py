"""Human-facing message text."""

import config
from events import EventType, RobotTaskState

# What to say to have a pending task asked about again (cmd_parser's request phrases).
REQUEST_PHRASES = {
    config.TASK_LIFT_PANEL: "lift the panel",
    config.TASK_LEAVE: "leave",
    config.TASK_BRING_CONNECTOR: "bring the connector",
    config.TASK_BRING_CLAMPING_TOOL: "bring the tool",
    config.TASK_RETURN_CLAMPING_TOOL: "take the tool back",
    config.TASK_PULL_CABLES: "pull the cables",
    config.TASK_LEAVE_HANDOVER: "leave",
}

# A commanded robot action, for reactive mode's replies; {panel} names the piece.
COMMAND_ACTIONS = {
    config.TASK_PULL_CABLES: "pull the cables of {panel}",
    config.TASK_LIFT_PANEL: "lift {panel}",
    config.TASK_BRING_CONNECTOR: "bring the pipe connector for {panel}",
    config.TASK_BRING_CLAMPING_TOOL: "bring the clamping tool for {panel}",
    config.TASK_RETURN_CLAMPING_TOOL: "take the clamping tool back from {panel}",
    config.TASK_LEAVE: "release {panel} and move away",
    config.TASK_LEAVE_HANDOVER: "move away from the hand-over position",
}


def panel_name(piece_id: int, location: str | None = None, *, spoken=False) -> str:
    """'the middle panel (piece 2)'; spoken 'the middle panel'. 'panel 2' without a location."""
    if not location:
        return f"panel {piece_id}"
    return f"the {location} panel" if spoken else f"the {location} panel (piece {piece_id})"


class MessageManager:
    """Centralizes detailed CLI text and concise speech without changing state.

    reactive: the robot acts only on the human's commands (reactive_task_manager.py),
    so the texts that promise a question ask for a command instead."""

    def __init__(self, reactive: bool = False):
        self.reactive = reactive

    def get_permission_message(self, task_id: int, *, spoken=False) -> str:
        if spoken:
            return {
                config.TASK_LIFT_PANEL: "May I lift the panel?",
                config.TASK_LEAVE: "May I release the panel and move away?",
                config.TASK_BRING_CONNECTOR: "Shall I bring the pipe connector?",
                config.TASK_BRING_CLAMPING_TOOL: "Shall I bring the clamping tool?",
                config.TASK_RETURN_CLAMPING_TOOL: "Shall I take the clamping tool back?",
                config.TASK_PULL_CABLES: "Shall I pull the cables?",
                config.TASK_LEAVE_HANDOVER: "May I move away now?",
            }[task_id]
        return config.PERMISSION_MESSAGES[task_id]

    def get_execution_message(self, task, *, spoken=False) -> str:
        if spoken:
            return {
                config.TASK_LIFT_PANEL: "Lifting the panel.",
                config.TASK_LEAVE: "Releasing the panel and moving away.",
                config.TASK_BRING_CONNECTOR: "Bringing the pipe connector.",
                config.TASK_BRING_CLAMPING_TOOL: "Bringing the clamping tool.",
                config.TASK_RETURN_CLAMPING_TOOL: "Taking the clamping tool back.",
                config.TASK_PULL_CABLES: "Pulling the cables.",
                config.TASK_LEAVE_HANDOVER: "Moving away.",
            }[task.task_id]
        lift_arrival = ("I will ask about manual adjustment when the lift is complete."
                        if config.LIFT_ASKS_FREE_DRIVE else
                        "Free drive turns on when the panel is in position, so you can adjust it.")
        messages = {
            config.TASK_LIFT_PANEL: "I am lifting the panel. " + lift_arrival,
            config.TASK_LEAVE: "I am releasing the panel and moving away.",
            config.TASK_BRING_CONNECTOR: "I am bringing the pipe connector. I will ask before handing it over.",
            config.TASK_BRING_CLAMPING_TOOL: "I am bringing the clamping tool. I will ask before handing it over.",
            config.TASK_RETURN_CLAMPING_TOOL: "I am taking the clamping tool back.",
            config.TASK_PULL_CABLES: "I am pulling the cables.",
            config.TASK_LEAVE_HANDOVER: "I am moving away from the hand-over position.",
        }
        return messages[task.task_id] + " Say or type pause, cancel, restart, faster, or slower."

    def get_handover_question(self, task, *, spoken=False) -> str:
        item = config.HANDOVER_ITEMS[task.task_id]
        if spoken:
            return f"Can I hand over the {item}?"
        return (f"Can I hand over the {item}? Yes opens the gripper, so hold it first. "
                "Say yes or no, or type your reply.")

    def get_handover_wait_message(self, task, *, spoken=False) -> str:
        item = config.HANDOVER_ITEMS[task.task_id]
        if spoken:
            return f'Okay. Say "give me the {item}" when you are ready.'
        return (f"Okay, I will keep holding the {item}. Let me know when you are ready to receive it: "
                f'say or type "give me the {item}".')

    def get_handed_over_message(self, task, leave_in: float | None = None, *, spoken=False) -> str:
        """leave_in: the robot leaves on its own this many seconds from now. The delay is
        kept short and not said -- a countdown sounds too explicit."""
        item = config.HANDOVER_ITEMS[task.task_id]
        if spoken:
            leaving = "" if leave_in is None else " I'm moving away."
            return f"Here is the {item}.{leaving}"
        leaving = ("" if leave_in is None else
                   " I'm moving away. Say or type cancel to keep me here.")
        return f"Opening the gripper. Here is the {item}.{leaving}"

    def get_left_handover_message(self, *, spoken=False) -> str:
        if spoken:
            return "I have moved away."
        return "I have moved away from the hand-over position."

    def get_free_drive_on_arrival_message(self, *, spoken=False) -> str:
        if spoken:
            return 'Panel in position. Free drive on. Adjust it, then say "done".'
        return ('The panel is in position and free drive is on. Adjust the panel, then say or type "done". '
                "I will then keep holding the panel while you screw it in place.")

    def ask_permission_for_free_drive(self, *, spoken=False) -> str:
        if spoken:
            return "Panel lifted. Enable free drive for adjustment?"
        return 'The panel is lifted. Would you like free drive for manual adjustment? Say yes or no, or type your reply. You can also say free drive.'

    def get_left_panel_message(self, *, spoken=False) -> str:
        if spoken:
            return "I have moved away from the panel."
        return "I have released the panel and moved away from it."

    def get_holding_message(self, *, spoken=False) -> str:
        if self.reactive:
            if spoken:
                return 'Holding the panel. Say "leave" to release it.'
            return 'I will keep holding the panel. Say or type "leave" when I should release it and move away.'
        if spoken:
            return 'Holding the panel.'
        return 'I will keep holding the panel. I will then ask before releasing the panel.'

    def get_pending_message(self, task, *, spoken=False) -> str:
        if spoken:
            staying = {
                config.TASK_LEAVE: " Still holding the panel.",
                config.TASK_LEAVE_HANDOVER: " Staying here.",
            }.get(task.task_id, "")
            return "Task pending." + staying
        reason = {"timeout": "No reply received.", "later": "Okay, later."}.get(task.pending_reason, "Okay.")
        staying = {
            config.TASK_LEAVE: " I will keep holding the panel.",
            config.TASK_LEAVE_HANDOVER: " I will stay at the hand-over position.",
        }.get(task.task_id, "")
        phrase = REQUEST_PHRASES.get(task.task_id, f"execute {task.task_instance_id}")
        if self.reactive:
            return f'{reason} This task is pending.{staying} Say or type "{phrase}" when I should do it.'
        return f'{reason} This task is pending.{staying} To be asked again when ready, say or type "{phrase}".'

    def get_human_does_message(self, task, *, spoken=False) -> str:
        """A no to a task the human can do: they do it, and the robot does not ask again."""
        if spoken:
            return "Okay, you do it."
        phrase = REQUEST_PHRASES.get(task.task_id, f"execute {task.task_instance_id}")
        return f'Okay, you do this one. If you want me to do it after all, say or type "{phrase}".'

    def get_advance_acknowledgement(self, event_type: EventType, *, spoken=False) -> str:
        """A yes to a task asked while the current one still runs (a no or later is
        answered at once: get_human_does_message / get_pending_message)."""
        if spoken:
            return "Okay, right after this task."
        return "Okay, I will start as soon as the current task is finished."

    def get_return_home_permission_message(self, *, spoken=False) -> str:
        if spoken:
            return 'Return home? Say "home", or "no" for manual recovery.'
        return 'Robot stopped in a validated recovery zone. Type "home" to return home or "no" for manual recovery.'

    def get_manual_recovery_message(self, *, spoken=False) -> str:
        if spoken:
            return 'Adjust manually. Say "done" when finished.'
        return 'Free-drive mode is enabled. Complete the manual adjustment and type "done" when finished.'

    def get_acknowledgement(self, event_type: EventType, *, spoken=False) -> str:
        if spoken:
            return {
                EventType.H_ACCEPT: "Task accepted.",
                EventType.H_REFUSE: "Okay, you do it.",
                EventType.H_DEFER: "Task pending.",
                EventType.H_PAUSE: "Paused.",
                EventType.H_RESUME: "Resumed.",
                EventType.H_RESTART: "Restarting.",
                EventType.H_CANCEL: "Canceled.",
                EventType.H_SPEEDUP: "Speed increased.",
                EventType.H_SLOWDOWN: "Speed decreased.",
                EventType.H_DONE: "Done signal sent.",
                EventType.ROBOT_SUCCESS: "Task complete.",
                EventType.ROBOT_HOMED: "Back home.",
                EventType.H_FREE_GO: 'Free drive on. Adjust the panel, then say "done".',
                EventType.H_RETURN_HOME: "Returning home.",
                EventType.H_MANUAL_RECOVERY: "Manual recovery started.",
            }.get(event_type, "Command processed.")
        messages = {
            EventType.H_ACCEPT: "Robot task accepted.",
            EventType.H_REFUSE: "Robot task refused; the human does it.",
            EventType.H_DEFER: "Robot task moved to pending.",
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

    # -- reactive mode: replies to the human's commands ---------------------------------

    def get_command_message(self, task_id: int, piece_id: int, location: str | None = None,
                            *, queued=False, spoken=False) -> str:
        """The robot carries out a command (queued: after the current task), naming the
        piece it resolved, so a wrong one can be canceled straight away."""
        action = COMMAND_ACTIONS[task_id].format(panel=panel_name(piece_id, location, spoken=spoken))
        if queued:
            return f"Okay, I will {action} after this task." if spoken else \
                f"Okay, I will {action} after the current task."
        return f"Okay, I will {action}." if spoken else f"Okay, I will {action}. Say or type cancel to stop me."

    def get_already_queued_message(self, task_id: int, piece_id: int, location: str | None = None,
                                   *, spoken=False) -> str:
        if spoken:
            return "That is next already."
        action = COMMAND_ACTIONS[task_id].format(panel=panel_name(piece_id, location))
        return f"I will {action} after the current task already."

    def get_queued_skipped_message(self, task_id: int, piece_id: int, location: str | None = None,
                                   *, spoken=False) -> str:
        """A queued command whose task was done (or started) meanwhile."""
        if spoken:
            return "Skipping that, it is done already."
        action = COMMAND_ACTIONS[task_id].format(panel=panel_name(piece_id, location))
        return f"I will not {action} after all: it is already done or under way."

    def get_nothing_left_message(self, task_name: str, *, spoken=False) -> str:
        if spoken:
            return "Nothing left to do for that."
        return f"{task_name} is already done or under way on every panel left."

    def get_screw_done_holding_message(self, *, spoken=False) -> str:
        if spoken:
            return 'Screwing done. Say "leave" to release the panel.'
        return 'Okay, screwing is done. I keep holding the panel: say or type "leave" when I should release it.'

    def get_cannot_leave_message(self, task, *, spoken=False) -> str:
        """"leave" with no panel held and no leave pending: what the robot is doing instead."""
        state = task.state if task is not None else None
        if state in (RobotTaskState.R_WAITING_FREE_DRIVE, RobotTaskState.R_FREE_DRIVE):
            if spoken:
                return 'Say "done" after the adjustment first.'
            return 'Free drive is on. Say or type "done" once the panel is adjusted; I then hold it until you say "leave".'
        if state in (RobotTaskState.R_WAITING_HANDOVER, RobotTaskState.R_HOLDING_HANDOVER):
            item = config.HANDOVER_ITEMS[task.task_id]
            if spoken:
                return f'Say "give me the {item}" to take it first.'
            return f'I am holding the {item} out to you. Say or type "give me the {item}" to take it, or cancel.'
        return "I am not holding a panel."

    def get_invalid_event_message(
        self,
        current_state: RobotTaskState | None,
        event_type: EventType,
        *, spoken=False,
    ) -> str:
        if spoken:
            return "That command is not available now."
        state_name = current_state.name if current_state is not None else "no active task"
        return f"Cannot process {event_type.name} while in {state_name}."
