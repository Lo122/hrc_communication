"""The command vocabulary -- one source of truth for typing and for speech.

This module owns every phrase the system understands. The Vosk grammar
and the GPT intent prompt are both built from PHRASES below, so the two
input paths can no longer drift apart (they previously kept separate
lists in gpt_stt.py and here).

Two kinds of entry:

  * ALIASES -- exact phrases. Vosk decodes against these directly, which
    is what makes it accurate in site noise: the decoder is constrained
    to this vocabulary instead of transcribing freely.
  * The GPT path maps arbitrary speech onto the same intents, so a
    worker can say "yeah go on" and still land on H_ACCEPT.
"""

from events import Event, EventType


class CommandParser:
    """Converts raw text into standardized events."""

    _ALIASES = {
        # --- answering a proposal ---------------------------------------
        "yes": EventType.H_ACCEPT,
        "yeah": EventType.H_ACCEPT,
        "yep": EventType.H_ACCEPT,
        "accept": EventType.H_ACCEPT,
        "okay": EventType.H_ACCEPT,
        "ok": EventType.H_ACCEPT,
        "go ahead": EventType.H_ACCEPT,
        "do it": EventType.H_ACCEPT,
        "please": EventType.H_ACCEPT,

        "no": EventType.H_REFUSE,
        "nope": EventType.H_REFUSE,
        "refuse": EventType.H_REFUSE,
        "no thanks": EventType.H_REFUSE,
        "skip": EventType.H_REFUSE,
        "not now": EventType.H_REFUSE,

        "later": EventType.H_DEFER,
        "defer": EventType.H_DEFER,
        "wait": EventType.H_DEFER,
        "hold on": EventType.H_DEFER,
        "one moment": EventType.H_DEFER,

        # --- while the robot is moving ----------------------------------
        "pause": EventType.H_PAUSE,
        "stop": EventType.H_PAUSE,
        "hold it": EventType.H_PAUSE,
        "freeze": EventType.H_PAUSE,

        "resume": EventType.H_RESUME,
        "continue": EventType.H_RESUME,
        "carry on": EventType.H_RESUME,
        "go on": EventType.H_RESUME,

        "restart": EventType.H_RESTART,
        "redo": EventType.H_RESTART,
        "again": EventType.H_RESTART,
        "start over": EventType.H_RESTART,

        "cancel": EventType.H_CANCEL,
        "cancel task": EventType.H_CANCEL,
        "abort": EventType.H_CANCEL,
        "forget it": EventType.H_CANCEL,

        "faster": EventType.H_SPEEDUP,
        "speed up": EventType.H_SPEEDUP,
        "quicker": EventType.H_SPEEDUP,

        "slower": EventType.H_SLOWDOWN,
        "slow down": EventType.H_SLOWDOWN,
        "easy": EventType.H_SLOWDOWN,

        # --- the hold-until-secured gate --------------------------------
        # The single most safety-relevant command in the vocabulary: it is
        # what makes the robot let go of a panel over someone's head. It
        # gets many spellings because failing to be understood here means
        # the worker keeps shouting at a robot that will not release.
        "secured": EventType.H_SECURED,
        "secure": EventType.H_SECURED,
        "it's fixed": EventType.H_SECURED,
        "its fixed": EventType.H_SECURED,
        "fixed": EventType.H_SECURED,
        "let go": EventType.H_SECURED,
        "release": EventType.H_SECURED,
        "you can let go": EventType.H_SECURED,
        "i'm done screwing": EventType.H_SECURED,

        "not yet": EventType.H_NOT_YET,
        "keep holding": EventType.H_NOT_YET,
        "hold": EventType.H_NOT_YET,
        "still going": EventType.H_NOT_YET,

        # --- free drive and recovery ------------------------------------
        "free drive": EventType.H_FREE_GO,
        "free go": EventType.H_FREE_GO,
        "go soft": EventType.H_FREE_GO,
        "soft mode": EventType.H_FREE_GO,

        "home": EventType.H_RETURN_HOME,
        "return home": EventType.H_RETURN_HOME,
        "go home": EventType.H_RETURN_HOME,

        "manual recovery": EventType.H_MANUAL_RECOVERY,
        "i'll move it": EventType.H_MANUAL_RECOVERY,

        "done": EventType.H_DONE,
        "finished": EventType.H_DONE,
        "that's it": EventType.H_DONE,
    }

    #: Commands that need a second confirmation before they take effect.
    #: An overheard word should not stop a robot that is holding a panel.
    CONFIRM_REQUIRED = {EventType.H_CANCEL}

    #: Asked, answered locally, never reaching the state machine.
    QUERY_ALIASES = {
        "status": "status",
        "what are you doing": "status",
        "what's happening": "status",
        "repeat": "repeat",
        "say again": "repeat",
        "what": "repeat",
        "why": "why",
        "why now": "why",
        "queue": "queue",
        "what's queued": "queue",
        "go": "run_pending",
        "run it": "run_pending",
    }

    @classmethod
    def phrases(cls) -> list[str]:
        """Every phrase the system accepts.

        The Vosk grammar and the GPT prompt are both built from this, so
        adding a phrase here is the only edit needed to teach both paths.
        """
        return sorted(set(cls._ALIASES) | set(cls.QUERY_ALIASES))

    def parse(self, raw_text: str, source: str = "human_cli") -> Event | None:
        text = raw_text.strip().lower().rstrip(".!?")
        if not text:
            return None

        if text.startswith("execute "):
            task_instance_id = text.split(maxsplit=1)[1]
            return Event(
                event_type=EventType.H_EXECUTE_PENDING_TASK,
                source=source,
                task_instance_id=task_instance_id,
            )

        event_type = self._ALIASES.get(text)
        if event_type is None:
            return None

        return Event(event_type=event_type, source=source)

    def parse_query(self, raw_text: str) -> str | None:
        """Recognize a question that is answered without touching task state."""
        text = raw_text.strip().lower().rstrip(".!?")
        return self.QUERY_ALIASES.get(text)
