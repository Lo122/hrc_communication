"""Watch screen model: one glanceable screen per robot task state.

This module is pure Python (no FastAPI, no I/O) so it can be unit-tested and
reused by other front ends (e.g. a native Wear OS app later).

Design rules encoded here (see Interface/UX_GUIDE.md):
  * One screen answers one question: "what is the robot doing / what do you
    want?" -> short title (<= 3 words) + one supporting line.
  * At most one primary action and two secondary actions per screen.
  * Destructive actions (cancel/stop) are always reachable while the robot
    moves, and require a press-and-hold on the watch (``hold=True``).
  * Every action is filtered through the real StateMachine, so the watch can
    never offer a command the decision layer would reject.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import config
from events import EventType, RobotTaskState, TaskStatus


TASK_NAMES = {
    config.TASK_LIFT_PANEL: "Lift the panel",
    config.TASK_LEAVE: "Let go of the panel",
    config.TASK_BRING_CONNECTOR: "Bring the connector",
    config.TASK_BRING_CLAMPING_TOOL: "Bring the clamp",
    config.TASK_RETURN_CLAMPING_TOOL: "Put the clamp away",
    config.TASK_PULL_CABLES: "Pull the cables",
    config.TASK_LEAVE_HANDOVER: "Step back",
}

# The robot asks (R_WAITING_RESPONSE): the question, and the line under it.
QUESTIONS = {
    config.TASK_LIFT_PANEL: "Lift the panel?",
    config.TASK_LEAVE: "Let go of the panel?",
    config.TASK_BRING_CONNECTOR: "Need the connector?",
    config.TASK_BRING_CLAMPING_TOOL: "Need the clamp?",
    config.TASK_RETURN_CLAMPING_TOOL: "Done with the clamp?",
    config.TASK_PULL_CABLES: "Pull the cables?",
    config.TASK_LEAVE_HANDOVER: "Shall I step back?",
}
ASK_DETAILS = {
    config.TASK_LIFT_PANEL: "I'll hold it while you work",
    config.TASK_LEAVE: "Only once it's screwed tight",
    config.TASK_BRING_CONNECTOR: "I can bring it over",
    config.TASK_BRING_CLAMPING_TOOL: "I can bring it over",
    config.TASK_RETURN_CLAMPING_TOOL: "I can put it away",
    config.TASK_PULL_CABLES: "I can do it for you",
    config.TASK_LEAVE_HANDOVER: "To give you room to work",
}

# Asked before the robot moves away: "no" means stay, not "I'll do it".
STAY_TASKS = {config.TASK_LEAVE, config.TASK_LEAVE_HANDOVER}

RUNNING_TITLES = {
    config.TASK_LIFT_PANEL: "Lifting the panel",
    config.TASK_LEAVE: "Stepping back",
    config.TASK_BRING_CONNECTOR: "Bringing the connector",
    config.TASK_BRING_CLAMPING_TOOL: "Bringing the clamp",
    config.TASK_RETURN_CLAMPING_TOOL: "Putting the clamp away",
    config.TASK_PULL_CABLES: "Pulling the cables",
    config.TASK_LEAVE_HANDOVER: "Stepping back",
}

# Robot tasks the human can ask for from the idle screen (H_REQUEST_ROBOT_TASK),
# keyed by task database name, in the order they are offered. Labels fit a pill.
# "Leave from the panel" is left out: it is only asked while the robot holds one.
# Only the ones next in the flow are shown (_requestable).
REQUEST_LABELS = {
    "Pull Cables": "Pull cables",
    "Lift": "Lift panel",
    "Bring Tool": "Bring clamp",
    "Bring Connector": "Connector",
    "Bring back Tool": "Clamp away",
}
MAX_REQUESTS = 2


@dataclass(frozen=True)
class WatchAction:
    command: str          # EventType name sent back to the backend
    label: str            # <= 12 characters, fits one watch button
    icon: str             # icon key understood by the front end
    role: str = "secondary"  # primary | secondary | danger | ghost
    hold: bool = False    # require press-and-hold (safety-critical)
    task_name: str | None = None  # H_REQUEST_ROBOT_TASK: the task database name


@dataclass
class WatchScreen:
    screen: str                     # stable id for the front end / logging
    tone: str                       # ask | run | wait | hand | alert | idle
    title: str
    detail: str = ""
    eyebrow: str = ""
    haptic: str = "none"            # none | tap | ask | alert | success
    countdown_total: float | None = None
    countdown_deadline: float | None = None
    show_speed: bool = False
    actions: list[WatchAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["actions"] = [asdict(action) for action in self.actions]
        return data


A = WatchAction
CANCEL_HOLD = A("H_CANCEL", "Stop", "stop", "danger", hold=True)

# Curated UX per state. Order = visual order on the watch.
_STATE_SCREENS: dict[RobotTaskState, dict[str, Any]] = {
    RobotTaskState.R_WAITING_RESPONSE: dict(
        screen="ask", tone="ask", haptic="ask",
        actions=[A("H_ACCEPT", "Yes, please", "check", "primary"),
                 A("H_REFUSE", "I'll do it", "x"),
                 A("H_DEFER", "Later", "clock")],
    ),
    RobotTaskState.R_DEFER: dict(
        screen="defer", tone="wait", haptic="tap", title="Starting shortly",
        detail="Cancel if you change your mind",
        actions=[A("H_CANCEL", "Cancel", "x", "secondary")],
    ),
    RobotTaskState.R_ACCEPTED: dict(
        screen="starting", tone="run", haptic="tap", title="Getting ready",
        detail="Please keep clear of the arm", actions=[CANCEL_HOLD],
    ),
    RobotTaskState.R_REDO: dict(
        screen="restarting", tone="run", haptic="tap", title="Starting over",
        detail="Back to the beginning", actions=[CANCEL_HOLD],
    ),
    RobotTaskState.R_EXECUTING: dict(
        screen="running", tone="run", haptic="none", show_speed=True,
        actions=[A("H_PAUSE", "Pause", "pause", "primary"),
                 A("H_RESTART", "Start over", "redo"),
                 CANCEL_HOLD],
    ),
    RobotTaskState.R_PAUSED: dict(
        screen="paused", tone="wait", haptic="tap", title="Paused",
        detail="I'm holding still",
        actions=[A("H_RESUME", "Carry on", "play", "primary"),
                 A("H_RESTART", "Start over", "redo"),
                 CANCEL_HOLD],
    ),
    RobotTaskState.R_WAITING_FREE_DRIVE: dict(
        screen="ask-free-drive", tone="ask", haptic="ask",
        eyebrow="PANEL IS UP", title="Adjust by hand?",
        detail="I'll go soft so you can move it",
        actions=[A("H_FREE_GO", "Yes", "hand", "primary"),
                 A("H_REFUSE", "Just hold", "x")],
    ),
    RobotTaskState.R_FREE_DRIVE: dict(
        screen="free-drive", tone="hand", haptic="tap",
        eyebrow="FREE DRIVE", title="Guide it into place",
        detail="Tap Done when it sits right",
        actions=[A("H_DONE", "Done", "check", "primary"), CANCEL_HOLD],
    ),
    RobotTaskState.R_HOLDING: dict(
        screen="holding", tone="hand", haptic="tap",
        eyebrow="HOLDING THE PANEL", title="Screw it in",
        detail="I'll hold it until you're done",
        actions=[A("H_SCREW_DONE", "All screwed", "check", "primary"), CANCEL_HOLD],
    ),
    RobotTaskState.R_WAITING_HANDOVER: dict(
        screen="ask-handover", tone="ask", haptic="ask",
        title="Ready to take it?", detail="Hold it, then I'll let go",
        actions=[A("H_ACCEPT", "Take it", "hand", "primary"),
                 A("H_REFUSE", "Not yet", "x")],
    ),
    RobotTaskState.R_HOLDING_HANDOVER: dict(
        screen="holding-handover", tone="hand", haptic="tap",
        title="Whenever you're ready", detail="I'll keep holding it for you",
        actions=[A("H_HANDOVER", "Hand it over", "hand", "primary"), CANCEL_HOLD],
    ),
    RobotTaskState.R_RECOVERY_EVALUATING: dict(
        screen="stopping", tone="alert", haptic="alert",
        title="Stopping", detail="Finding a safe way back", actions=[],
    ),
    RobotTaskState.R_WAITING_HOME_PERMISSION: dict(
        screen="ask-home", tone="alert", haptic="ask",
        eyebrow="STOPPED", title="Head back home?",
        detail="The way back is clear",
        actions=[A("H_RETURN_HOME", "Go home", "home", "primary"),
                 A("H_MANUAL_RECOVERY", "By hand", "hand")],
    ),
    RobotTaskState.R_RETURNING_HOME: dict(
        screen="homing", tone="run", haptic="tap",
        title="Heading home", detail="Please keep clear of the arm", actions=[],
    ),
    RobotTaskState.R_MANUAL_RECOVERY: dict(
        screen="manual", tone="hand", haptic="alert",
        eyebrow="MANUAL RECOVERY", title="Guide me by hand",
        detail="I'm in free drive. Tap Done after",
        actions=[A("H_DONE", "Done", "check", "primary"),
                 A("H_CANCEL", "Abort", "x", "danger", hold=True)],
    ),
}

# Questions that belong to no robot task (TaskManager.ask_question): the demo
# opening's. A yes / no answers them like any robot question.
_QUESTION_SCREENS: dict[str, dict[str, Any]] = {
    "start": dict(
        eyebrow="NEW ASSEMBLY", title="Shall we start?",
        detail="First up: the cables",
        actions=[A("H_ACCEPT", "Let's go", "check", "primary"),
                 A("H_REFUSE", "Not yet", "clock")],
    ),
    "continue": dict(
        eyebrow="CABLES ARE YOURS", title="Move on?",
        detail="Next, I can lift the panel",
        actions=[A("H_ACCEPT", "Next step", "check", "primary"),
                 A("H_REFUSE", "Not yet", "clock")],
    ),
}

# Commands that are valid by handler logic although the transition table
# expresses them differently (e.g. H_REFUSE in R_WAITING_FREE_DRIVE -> holding).
_HANDLER_ONLY = {
    (RobotTaskState.R_ACCEPTED, EventType.H_CANCEL),
    (RobotTaskState.R_FREE_DRIVE, EventType.H_CANCEL),
    (RobotTaskState.R_HOLDING, EventType.H_CANCEL),
    (RobotTaskState.R_MANUAL_RECOVERY, EventType.H_CANCEL),
}

WATCH_COMMANDS = frozenset(
    {a.command for spec in _STATE_SCREENS.values() for a in spec["actions"]}
    | {"H_SPEEDUP", "H_SLOWDOWN", "H_EXECUTE_PENDING_TASK",
       "H_TASK_DONE", "H_REQUEST_ROBOT_TASK"}
)


def _allowed(state_machine, state, command: str, task_id) -> bool:
    event_type = EventType[command]
    if (state, event_type) in _HANDLER_ONLY:
        return True
    return state_machine.is_valid_transition(state, event_type, task_id)


def build_screen(task, pending: list, state_machine, now: float,
                 last_message: str = "", *, advance=None, advance_answer: str | None = None,
                 tracker=None, question: str | None = None, opening: bool = False) -> WatchScreen:
    """Return the screen for the active task (or the question/pending/idle screen).

    advance: the robot task asked about while the active one still runs
    (TaskManager.advance_task), and advance_answer the human's yes / later to it
    so far. tracker: the assembly-task tracker, for the idle screen's inputs.
    question: the open question outside any robot task (TaskManager.question).
    opening: the demo opening leads (TaskManager.in_opening): no robot requests."""

    if task is None:
        if question in _QUESTION_SCREENS:
            spec = _QUESTION_SCREENS[question]
            return WatchScreen(screen="ask-" + question, tone="ask", haptic="ask",
                               eyebrow=spec["eyebrow"], title=spec["title"],
                               detail=spec["detail"], actions=list(spec["actions"]))
        if pending:
            first = pending[0]
            more = f" +{len(pending) - 1}" if len(pending) > 1 else ""
            return WatchScreen(
                screen="pending", tone="idle", haptic="none",
                eyebrow="ON HOLD" + more,
                title=TASK_NAMES.get(first.task_id, "A task on hold"),
                detail="Tap whenever you want my help",
                actions=[A("H_EXECUTE_PENDING_TASK", "Start now", "play", "primary")],
            )
        return _idle_screen(tracker, opening)

    if advance is not None and advance_answer is None and task.state is RobotTaskState.R_EXECUTING:
        return _advance_screen(task, advance, state_machine)

    spec = dict(_STATE_SCREENS.get(task.state, {}))
    if not spec:
        return WatchScreen(screen="unknown", tone="idle", title=task.state.name)

    name = TASK_NAMES.get(task.task_id, f"Task {task.task_id}")
    spec.setdefault("eyebrow", name.upper())
    if task.state is RobotTaskState.R_WAITING_RESPONSE:
        spec["title"] = QUESTIONS.get(task.task_id, name + "?")
        spec["detail"] = ASK_DETAILS.get(task.task_id, "Happy to help")
        if task.task_id in STAY_TASKS:
            # A no keeps the robot where it is; the human does not take it over.
            spec["actions"] = [replace(a, label="Not yet") if a.command == "H_REFUSE" else a
                               for a in spec["actions"]]
    elif task.state is RobotTaskState.R_EXECUTING:
        spec["title"] = RUNNING_TITLES.get(task.task_id, name)
        spec["detail"] = "Please keep clear of the arm"
        if advance is not None:
            # Answered already: it starts once this task has succeeded.
            spec["detail"] = "Next: " + TASK_NAMES.get(advance.task_id, "the next task").lower()

    screen = WatchScreen(**{k: v for k, v in spec.items() if k != "actions"})
    screen.actions = [
        a for a in spec["actions"]
        if _allowed(state_machine, task.state, a.command, task.task_id)
    ]

    # Countdown rings mirror the backend timers (TimerManager).
    started = task.updated_at or now
    timings = config.TASK_TIMINGS.get(task.task_id, {})
    if task.state is RobotTaskState.R_WAITING_RESPONSE:
        total = timings.get("response_timeout_seconds", config.RESPONSE_TIMEOUT_SECONDS)
        screen.countdown_total, screen.countdown_deadline = total, started + total
    elif task.state is RobotTaskState.R_DEFER:
        total = task.defer_seconds or timings.get("defer_seconds", config.DEFER_SECONDS)
        screen.countdown_total, screen.countdown_deadline = total, started + total
        if task.task_id == config.TASK_LEAVE:
            screen.detail = "Letting go of the panel"
        elif task.task_id == config.TASK_LEAVE_HANDOVER:
            screen.detail = "Stepping back"
    return screen


def _idle_screen(tracker, opening: bool = False) -> WatchScreen:
    """Nothing to answer: confirm the task the human is on, or ask the robot for the
    task that comes next."""
    screen = WatchScreen(screen="idle", tone="idle", title="I'm ready",
                         detail="I'll ask when I can help")
    if tracker is None:
        return screen
    working = tracker.working_task()
    if working is not None:
        screen.eyebrow = working.task_name.upper()
        screen.title = "Over to you"
        screen.detail = "Tap Done when you've finished"
        screen.actions.append(A("H_TASK_DONE", "Done", "check", "primary"))
    elif opening:
        screen.detail = "We'll start together in a moment"
    if opening:
        return screen  # the opening dialogue leads; nothing to ask for yet
    requests = [A("H_REQUEST_ROBOT_TASK", label, "play", task_name=name)
                for name, label in REQUEST_LABELS.items() if _requestable(tracker, name)]
    screen.actions += requests[:MAX_REQUESTS]
    return screen


def _requestable(tracker, task_name: str) -> bool:
    """The robot may do it, and it is next in the flow on the piece it is for:
      - a task chained after another (Lift after Pull Cables, the "robot task" chain
        of a trigger rule) waits until that one is done;
      - the human's next assembly step on their piece can be asked for;
      - a task a trigger rule starts from other tasks (Bring Tool after Screw, the
        next panel's cables after Connect Cables) once one of those has begun.
    Nobody may be working on it already (as TaskManager checks)."""
    database = tracker.database
    if not database.can_execute(task_name, "Robot"):
        return False
    piece_id = tracker.first_open_piece_for(task_name)
    if piece_id is None:
        return False
    task = tracker.get(task_name, piece_id)
    if task is not None and task.status == TaskStatus.WORKING:
        return False
    if _waits_for_chain(tracker, task_name, piece_id):
        return False
    return _next_step(tracker, piece_id) == task_name or _rule_started(tracker, task_name, piece_id)


def _waits_for_chain(tracker, task_name: str, piece_id: int) -> bool:
    for rule in tracker.database.trigger_rules.values():
        chain = list(rule.robot_tasks)
        if task_name in chain[1:]:
            before = tracker.get(chain[chain.index(task_name) - 1], piece_id)
            if before is not None and before.status != TaskStatus.DONE:
                return True
    return False


def _next_step(tracker, piece_id: int) -> str | None:
    """The first assembly step (a recognized human step) not done on the human's piece."""
    if piece_id != tracker.human_piece_id:
        return None
    for piece in tracker.database.pieces:
        if piece.piece_id == piece_id:
            for name in piece.task_list:
                task = tracker.get(name, piece_id)
                if tracker.is_recognized(name) and task is not None and task.status != TaskStatus.DONE:
                    return name
    return None


def _rule_started(tracker, task_name: str, piece_id: int) -> bool:
    rule = tracker.database.trigger_rules.get(task_name)
    if rule is None:
        return False
    for previous in rule.previous_tasks:
        source = piece_id - rule.piece_offset(previous)
        task = tracker.get(previous, source) if source in tracker.piece_ids else None
        if task is not None and task.status in (TaskStatus.WORKING, TaskStatus.DONE):
            return True
    return False


def _advance_screen(task, advance, state_machine) -> WatchScreen:
    """The robot asks about its next task while the active one still moves: a yes
    starts it right after, a no makes it pending. Stop keeps acting on the move."""
    doing = RUNNING_TITLES.get(task.task_id, "working").lower()
    stop = [CANCEL_HOLD] if _allowed(state_machine, task.state, "H_CANCEL", task.task_id) else []
    return WatchScreen(
        screen="ask-next", tone="ask", haptic="ask",
        eyebrow="NEXT UP",
        title=QUESTIONS.get(advance.task_id, TASK_NAMES.get(advance.task_id, "Next task") + "?"),
        detail=f"Right after {doing}",
        actions=[A("H_ACCEPT", "Yes, please", "check", "primary"), A("H_REFUSE", "No", "x"), *stop],
    )


def allowed_commands(task, pending: list, state_machine, **context) -> set[tuple[str, str | None]]:
    """(command, task_name) pairs the watch may send right now (server-side
    allowlist). context: build_screen's advance / advance_answer / tracker /
    question / opening."""
    screen = build_screen(task, pending, state_machine, now=0.0, **context)
    commands = {(a.command, a.task_name) for a in screen.actions}
    if screen.show_speed:
        commands |= {("H_SPEEDUP", None), ("H_SLOWDOWN", None)}
    return commands
