"""Human CLI and voice command parser."""

import re

from events import Event, EventType, RobotTaskState

# Spoken contractions, written out so one key covers both ("i'll do it" = "i will do it").
_CONTRACTIONS = {"i'll": "i will", "i'm": "i am", "let's": "let us", "it's": "it is",
                 "that's": "that is", "don't": "do not", "you're": "you are"}
# Politeness around a command that does not change it ("yes please", "robot, pause").
_FILLERS = ("please", "robot", "now", "then", "okay so", "um", "uh")


def normalize(text: str) -> str:
    """Lower case, no punctuation, contractions written out, single spaces."""
    text = text.lower().replace("[unk]", " ")
    words = [_CONTRACTIONS.get(word, word) for word in re.sub(r"[^a-z0-9' ]+", " ", text).split()]
    return " ".join(words).replace("'", "")


class CommandParser:
    """Converts raw text into standardized events."""

    # Voice: the robot is addressed by name -- "hey UR, lift the panel" -- so talk in the
    # room is not taken for a command. The recognizer hears "UR" as these.
    WAKE_WORDS = ("hey ur", "hey u r", "hey you are", "hey your", "hey you r", "hey robot",
                  # "hey" is short and often lost: "you are, carry on" still names it.
                  "you are")
    # Said alone they work without the name: stopping the robot must never wait for it.
    EMERGENCY_PHRASES = ("stop", "pause", "cancel", "stop stop", "stop the robot", "stop it")

    _ALIASES = {
        "yes": EventType.H_ACCEPT,
        "yes please": EventType.H_ACCEPT,
        "yeah": EventType.H_ACCEPT,
        "yep": EventType.H_ACCEPT,
        "sure": EventType.H_ACCEPT,
        "accept": EventType.H_ACCEPT,
        "okay": EventType.H_ACCEPT,
        "ok": EventType.H_ACCEPT,
        "alright": EventType.H_ACCEPT,
        "all right": EventType.H_ACCEPT,
        "go ahead": EventType.H_ACCEPT,
        "do it": EventType.H_ACCEPT,
        "please do": EventType.H_ACCEPT,
        "of course": EventType.H_ACCEPT,
        "sounds good": EventType.H_ACCEPT,
        "let us go": EventType.H_ACCEPT,
        "no": EventType.H_REFUSE,
        "nope": EventType.H_REFUSE,
        "no thanks": EventType.H_REFUSE,
        "no thank you": EventType.H_REFUSE,
        "refuse": EventType.H_REFUSE,
        "later": EventType.H_DEFER,
        "maybe later": EventType.H_DEFER,
        "not now": EventType.H_DEFER,
        "in a minute": EventType.H_DEFER,
        "defer": EventType.H_DEFER,
        "pause": EventType.H_PAUSE,
        "stop": EventType.H_PAUSE,
        "stop stop": EventType.H_PAUSE,
        "stop the robot": EventType.H_PAUSE,
        "stop it": EventType.H_PAUSE,
        "hold on": EventType.H_PAUSE,
        "continue": EventType.H_RESUME,
        "resume": EventType.H_RESUME,
        "carry on": EventType.H_RESUME,
        "go on": EventType.H_RESUME,
        "keep going": EventType.H_RESUME,
        "restart": EventType.H_RESTART,
        "redo": EventType.H_RESTART,
        "start over": EventType.H_RESTART,
        "start again": EventType.H_RESTART,
        "cancel": EventType.H_CANCEL,
        "cancel task": EventType.H_CANCEL,
        "cancel that": EventType.H_CANCEL,
        "abort": EventType.H_CANCEL,
        "faster": EventType.H_SPEEDUP,
        "go faster": EventType.H_SPEEDUP,
        "speed up": EventType.H_SPEEDUP,
        "slower": EventType.H_SLOWDOWN,
        "go slower": EventType.H_SLOWDOWN,
        "slow down": EventType.H_SLOWDOWN,
        "free drive": EventType.H_FREE_GO,
        "free go": EventType.H_FREE_GO,
        "home": EventType.H_RETURN_HOME,
        "go home": EventType.H_RETURN_HOME,
        "return home": EventType.H_RETURN_HOME,
        "manual recovery": EventType.H_MANUAL_RECOVERY,
        "by hand": EventType.H_MANUAL_RECOVERY,
        "done": EventType.H_DONE,
        "finished": EventType.H_DONE,
        "i am done": EventType.H_DONE,
        "all done": EventType.H_DONE,
        "i am finished": EventType.H_DONE,
        "it is in place": EventType.H_DONE,
        "adjustment done": EventType.H_DONE,
        "screw done": EventType.H_SCREW_DONE,
        "screws done": EventType.H_SCREW_DONE,
        "screwing done": EventType.H_SCREW_DONE,
        "finished screwing": EventType.H_SCREW_DONE,
        "all screwed": EventType.H_SCREW_DONE,
        "screwed in": EventType.H_SCREW_DONE,
        "next piece": EventType.H_NEXT_PIECE,
        "next panel": EventType.H_NEXT_PIECE,
        # Ready to take the item the robot brought: it opens the gripper.
        "hand over": EventType.H_HANDOVER,
        "hand it over": EventType.H_HANDOVER,
        "give it to me": EventType.H_HANDOVER,
        "i will take it": EventType.H_HANDOVER,
        "take it": EventType.H_HANDOVER,
    }

    # Replies whose meaning depends on the question: (phrase, state) -> event. Said while
    # the robot is in none of its states, the phrase is not understood rather than taken
    # for something it does not mean there (state_machine.py decides what each event
    # does in each state; a reply must not land on another one). Without a state --
    # typed at the CLI -- the first meaning listed counts.
    _STATE_REPLIES = {
        # "Shall I ...?" -- the human does the task themselves.
        ("i will do it", RobotTaskState.R_WAITING_RESPONSE): EventType.H_REFUSE,
        ("let me do it", RobotTaskState.R_WAITING_RESPONSE): EventType.H_REFUSE,
        # "Move on?" (the demo opening) and any other "Would you like me to ...?".
        ("next step", RobotTaskState.R_WAITING_RESPONSE): EventType.H_ACCEPT,
        # "Not yet": later to a question, not ready for the item held out.
        ("not yet", RobotTaskState.R_WAITING_RESPONSE): EventType.H_DEFER,
        ("not yet", RobotTaskState.R_WAITING_HANDOVER): EventType.H_REFUSE,
        # "Adjust by hand?" -- no adjustment, just hold the panel.
        ("just hold", RobotTaskState.R_WAITING_FREE_DRIVE): EventType.H_REFUSE,
    }

    # "Say that again": the robot repeats its question (or its last line). Not an event:
    # nothing in the decision layer changes (CommunicationManager repeats it).
    REPEAT_PHRASES = ("repeat", "repeat that", "repeat again", "repeat the question", "say again",
                      "say that again", "can you repeat", "could you repeat", "pardon", "what did you say")

    # Asking for an item -> H_HANDOVER {task_name: the task that brings it}. Ready to take
    # the item the robot holds out, it opens the gripper; in reactive mode, with no item
    # held out, it asks the robot to bring it (ReactiveTaskManager._handle_handover).
    _HANDOVER_ALIASES = {
        "give me the tool": "Bring Tool",
        "give me the coupling": "Bring Connector",
        "give me the pipe coupling": "Bring Connector",
        "give me the connector": "Bring Connector",
        "give me the pipe connector": "Bring Connector",
        "give me the clamp": "Bring Tool",
        "give me the clamping tool": "Bring Tool",
    }

    # Human confirms a task is finished -> H_TASK_DONE {task_name} (task database names).
    # "screw done" stays H_SCREW_DONE above: it confirms Screw, and while the robot
    # holds the panel it also ends the holding, after which the task database's
    # "Leave from the panel" rule offers to release the panel (see TaskManager).
    _TASK_DONE_ALIASES = {
        "cables pulled": "Pull Cables",
        "cables are pulled": "Pull Cables",
        "cables done": "Pull Cables",
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
        "lift a panel": "Lift",
        "lift panel": "Lift",
        "lift": "Lift",
        "bring the tool": "Bring Tool",
        "bring tool": "Bring Tool",
        "bring the clamp": "Bring Tool",
        "bring me the tool": "Bring Tool",
        "bring the connector": "Bring Connector",
        "bring the pipe connector": "Bring Connector",
        "bring me the connector": "Bring Connector",
        "bring connector": "Bring Connector",
        "take the tool back": "Bring back Tool",
        "bring back tool": "Bring back Tool",
        "put the clamp away": "Bring back Tool",
    }

    @classmethod
    def phrases(cls) -> list[str]:
        """Every command phrase."""
        state_replies = list(dict.fromkeys(phrase for phrase, _ in cls._STATE_REPLIES))
        return [*cls._ALIASES, *state_replies, *cls._HANDOVER_ALIASES, *cls._TASK_DONE_ALIASES,
                *cls._ROBOT_REQUEST_ALIASES]

    @classmethod
    def voice_phrases(cls, wake_word: bool = True) -> list[str]:
        """The voice recognizer's grammar. With the wake word: the name alone (the command
        may follow a pause), the name before every command, every command alone (heard,
        then ignored unless it is an emergency word or follows the name) -- and [unk], so
        other speech is not forced onto the nearest command."""
        commands = [*cls.phrases(), *cls.REPEAT_PHRASES]
        if not wake_word:
            return [*commands, "[unk]"]
        wakes = [wake for wake in cls.WAKE_WORDS if wake not in ("hey u r", "hey you r")]
        return [*wakes, *(f"{wake} {command}" for wake in wakes for command in commands),
                *commands, "[unk]"]

    def addressed(self, raw_text: str) -> str | None:
        """The command in an utterance meant for the robot: what follows the wake word
        ("" when the name came alone), or an emergency word said on its own. None: it
        was not meant for the robot."""
        text = normalize(raw_text)
        for wake in sorted(self.WAKE_WORDS, key=len, reverse=True):
            if text == wake or text.startswith(wake + " "):
                return text[len(wake):].strip()
        return text if text in self.EMERGENCY_PHRASES else None

    def parse(self, raw_text: str, source: str = "human_cli", state=None) -> Event | None:
        """state: the robot task's state the reply is for (voice knows it); a reply that
        only means something in some states (_STATE_REPLIES) is understood only there."""
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

        text = self._known(normalize(text))
        if text is None:
            return None
        if text in self._state_phrases():
            event_type = self._state_reply(text, state)
            return None if event_type is None else Event(event_type, source)
        if text in self._TASK_DONE_ALIASES:
            return Event(EventType.H_TASK_DONE, source, payload={"task_name": self._TASK_DONE_ALIASES[text]})
        if text in self._ROBOT_REQUEST_ALIASES:
            return Event(EventType.H_REQUEST_ROBOT_TASK, source,
                         payload={"task_name": self._ROBOT_REQUEST_ALIASES[text]})
        if text in self._HANDOVER_ALIASES:
            return Event(EventType.H_HANDOVER, source, payload={"task_name": self._HANDOVER_ALIASES[text]})

        return Event(event_type=self._ALIASES[text], source=source)

    @classmethod
    def _state_phrases(cls) -> set[str]:
        return {phrase for phrase, _ in cls._STATE_REPLIES}

    def _state_reply(self, phrase: str, state) -> EventType | None:
        if state is None:
            return next(event for (said, _), event in self._STATE_REPLIES.items() if said == phrase)
        return self._STATE_REPLIES.get((phrase, state))

    def is_repeat(self, raw_text: str) -> bool:
        """Is this a request to say the last question again?"""
        text = normalize(raw_text)
        return text in self.REPEAT_PHRASES or self._without_fillers(text) in self.REPEAT_PHRASES

    def _known(self, text: str) -> str | None:
        """The phrase as a key: as said, or without polite words around it."""
        known = (self._ALIASES, self._state_phrases(), self._HANDOVER_ALIASES, self._TASK_DONE_ALIASES,
                 self._ROBOT_REQUEST_ALIASES)
        if any(text in aliases for aliases in known):
            return text
        stripped = self._without_fillers(text)
        return stripped if any(stripped in aliases for aliases in known) else None

    @staticmethod
    def _without_fillers(text: str) -> str:
        words, fillers = text.split(), [filler.split() for filler in _FILLERS]
        trimmed = True
        while trimmed and words:
            trimmed = False
            for filler in fillers:
                if words[:len(filler)] == filler:
                    words, trimmed = words[len(filler):], True
                elif words[-len(filler):] == filler:
                    words, trimmed = words[:-len(filler)], True
        return " ".join(words)
