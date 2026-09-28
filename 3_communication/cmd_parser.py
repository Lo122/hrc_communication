"""Human CLI command parser."""

from events import Event, EventType


class CommandParser:
    """Converts raw text into standardized events."""

    _ALIASES = {
        "yes": EventType.H_ACCEPT,
        "accept": EventType.H_ACCEPT,
        "okay": EventType.H_ACCEPT,
        "ok": EventType.H_ACCEPT,
        "no": EventType.H_REFUSE,
        "refuse": EventType.H_REFUSE,
        "later": EventType.H_DEFER,
        "defer": EventType.H_DEFER,
        "pause": EventType.H_PAUSE,
        "stop": EventType.H_PAUSE,
        "continue": EventType.H_RESUME,
        "resume": EventType.H_RESUME,
        "restart": EventType.H_RESTART,
        "redo": EventType.H_RESTART,
        "cancel": EventType.H_CANCEL,
        "cancel task": EventType.H_CANCEL,
        "faster": EventType.H_SPEEDUP,
        "speed up": EventType.H_SPEEDUP,
        "slower": EventType.H_SLOWDOWN,
        "slow down": EventType.H_SLOWDOWN,
        "free drive": EventType.H_FREE_GO,
        "free go": EventType.H_FREE_GO,
        "home": EventType.H_RETURN_HOME,
        "return home": EventType.H_RETURN_HOME,
        "manual recovery": EventType.H_MANUAL_RECOVERY,
        "done": EventType.H_DONE,
        "finished": EventType.H_DONE,
        "adjustment done": EventType.H_DONE,
        "screw done": EventType.H_SCREW_DONE,
        "screwing done": EventType.H_SCREW_DONE,
        "finished screwing": EventType.H_SCREW_DONE,
        "next piece": EventType.H_NEXT_PIECE,
        # Ready to take the item the robot brought: it opens the gripper.
        "hand over": EventType.H_HANDOVER,
        "give me the tool": EventType.H_HANDOVER,
        "give me the coupling": EventType.H_HANDOVER,
        "give me the pipe coupling": EventType.H_HANDOVER,
        "give me the connector": EventType.H_HANDOVER,
        "give me the pipe connector": EventType.H_HANDOVER,
    }

    # Human confirms a task is finished -> H_TASK_DONE {task_name} (task database names).
    # "screw done" stays H_SCREW_DONE above: it confirms Screw, and while the robot
    # holds the panel it also ends the holding, after which the task database's
    # "Leave from the panel" rule offers to release the panel (see TaskManager).
    _TASK_DONE_ALIASES = {
        "cables pulled": "Pull Cables",
        "pull cables done": "Pull Cables",
        "lift done": "Lift",
        "lifted": "Lift",
        "place done": "Place",
        "placed": "Place",
        "align done": "Align",
        "aligned": "Align",
        "cables connected": "Connect Cables",
        "connect done": "Connect Cables",
        "clamp done": "Clamp Coupling",
        "clamped": "Clamp Coupling",
        "tool brought": "Bring Tool",
        "connector brought": "Bring Connector",
        "tool returned": "Bring back Tool",
    }

    # Human asks the robot for a task -> H_REQUEST_ROBOT_TASK {task_name}. It skips the
    # trigger rules, but the robot still asks permission before executing. Naming a
    # pending task asks about it again; the robot's state picks the task and piece
    # (TaskManager._request_by_name) -- "leave" is either leave.
    _ROBOT_REQUEST_ALIASES = {
        "leave": "Leave from the panel",
        "leave the panel": "Leave from the panel",
        "release the panel": "Leave from the panel",
        "move away": "Leave from the panel",
        "pull the cables": "Pull Cables",
        "pull cables": "Pull Cables",
        "lift the panel": "Lift",
        "lift": "Lift",
        "bring the tool": "Bring Tool",
        "bring tool": "Bring Tool",
        "bring the connector": "Bring Connector",
        "bring connector": "Bring Connector",
        "take the tool back": "Bring back Tool",
        "bring back tool": "Bring back Tool",
    }

    @classmethod
    def phrases(cls) -> list[str]:
        """Every command phrase, for the voice recognizer's grammar."""
        return [*cls._ALIASES, *cls._TASK_DONE_ALIASES, *cls._ROBOT_REQUEST_ALIASES]

    def parse(self, raw_text: str, source: str = "human_cli") -> Event | None:
        text = raw_text.strip().lower()
        if not text:
            return None

        if text.startswith("execute "):
            task_instance_id = text.split(maxsplit=1)[1]
            return Event(
                event_type=EventType.H_EXECUTE_PENDING_TASK,
                source=source,
                task_instance_id=task_instance_id,
            )

        if text in self._TASK_DONE_ALIASES:
            return Event(EventType.H_TASK_DONE, source, payload={"task_name": self._TASK_DONE_ALIASES[text]})
        if text in self._ROBOT_REQUEST_ALIASES:
            return Event(EventType.H_REQUEST_ROBOT_TASK, source,
                         payload={"task_name": self._ROBOT_REQUEST_ALIASES[text]})

        event_type = self._ALIASES.get(text)
        if event_type is None:
            return None

        return Event(event_type=event_type, source=source)
