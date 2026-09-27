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

from dataclasses import asdict, dataclass, field
from typing import Any

import config
from events import EventType, RobotTaskState


TASK_NAMES = {
    config.TASK_LIFT_PANEL: "Lift panel",
    config.TASK_LEAVE: "Release panel",
    config.TASK_BRING_CONNECTOR: "Bring connector",
    config.TASK_BRING_CLAMPING_TOOL: "Bring clamp tool",
    config.TASK_RETURN_CLAMPING_TOOL: "Return clamp tool",
    config.TASK_PULL_CABLES: "Pull cables",
    config.TASK_LEAVE_HANDOVER: "Leave hand-over",
}

QUESTIONS = {
    config.TASK_LIFT_PANEL: "Lift the panel?",
    config.TASK_LEAVE: "Release & move away?",
    config.TASK_BRING_CONNECTOR: "Bring connector?",
    config.TASK_BRING_CLAMPING_TOOL: "Bring clamp tool?",
    config.TASK_RETURN_CLAMPING_TOOL: "Take clamp tool back?",
    config.TASK_PULL_CABLES: "Pull the cables?",
    config.TASK_LEAVE_HANDOVER: "Move away?",
}

RUNNING_TITLES = {
    config.TASK_LIFT_PANEL: "Lifting panel",
    config.TASK_LEAVE: "Moving away",
    config.TASK_BRING_CONNECTOR: "Bringing connector",
    config.TASK_BRING_CLAMPING_TOOL: "Bringing clamp tool",
    config.TASK_RETURN_CLAMPING_TOOL: "Returning clamp tool",
    config.TASK_PULL_CABLES: "Pulling cables",
    config.TASK_LEAVE_HANDOVER: "Moving away",
}


@dataclass(frozen=True)
class WatchAction:
    command: str          # EventType name sent back to the backend
    label: str            # <= 12 characters, fits one watch button
    icon: str             # icon key understood by the front end
    role: str = "secondary"  # primary | secondary | danger | ghost
    hold: bool = False    # require press-and-hold (safety-critical)


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
        actions=[A("H_ACCEPT", "Yes", "check", "primary"),
                 A("H_REFUSE", "No", "x"),
                 A("H_DEFER", "In 5 s", "clock")],
    ),
    RobotTaskState.R_DEFER: dict(
        screen="defer", tone="wait", haptic="tap", title="Starting soon",
        actions=[A("H_CANCEL", "Cancel", "x", "secondary")],
    ),
    RobotTaskState.R_ACCEPTED: dict(
        screen="starting", tone="run", haptic="tap", title="Starting…",
        detail="Robot is getting ready", actions=[CANCEL_HOLD],
    ),
    RobotTaskState.R_REDO: dict(
        screen="restarting", tone="run", haptic="tap", title="Restarting…",
        detail="Robot is getting ready", actions=[CANCEL_HOLD],
    ),
    RobotTaskState.R_EXECUTING: dict(
        screen="running", tone="run", haptic="none", show_speed=True,
        actions=[A("H_PAUSE", "Pause", "pause", "primary"),
                 A("H_RESTART", "Redo", "redo"),
                 CANCEL_HOLD],
    ),
    RobotTaskState.R_PAUSED: dict(
        screen="paused", tone="wait", haptic="tap", title="Paused",
        detail="Robot is holding still",
        actions=[A("H_RESUME", "Resume", "play", "primary"),
                 A("H_RESTART", "Redo", "redo"),
                 CANCEL_HOLD],
    ),
    RobotTaskState.R_WAITING_FREE_DRIVE: dict(
        screen="ask-free-drive", tone="ask", haptic="ask",
        eyebrow="PANEL LIFTED", title="Guide by hand?",
        detail="Free drive lets you adjust the panel",
        actions=[A("H_FREE_GO", "Yes", "hand", "primary"),
                 A("H_REFUSE", "Hold", "x")],
    ),
    RobotTaskState.R_FREE_DRIVE: dict(
        screen="free-drive", tone="hand", haptic="tap",
        eyebrow="FREE DRIVE", title="Move the panel",
        detail="Tap Done when it is in place",
        actions=[A("H_DONE", "Done", "check", "primary"), CANCEL_HOLD],
    ),
    RobotTaskState.R_HOLDING: dict(
        screen="holding", tone="hand", haptic="tap",
        eyebrow="HOLDING PANEL", title="Screw it in",
        detail="Robot holds until you are done",
        actions=[A("H_SCREW_DONE", "Screwed", "check", "primary"), CANCEL_HOLD],
    ),
    RobotTaskState.R_WAITING_HANDOVER: dict(
        screen="ask-handover", tone="ask", haptic="ask",
        title="Take it now?", detail="Yes opens the gripper",
        actions=[A("H_ACCEPT", "Yes", "hand", "primary"),
                 A("H_REFUSE", "Not yet", "x")],
    ),
    RobotTaskState.R_HOLDING_HANDOVER: dict(
        screen="holding-handover", tone="hand", haptic="tap",
        title="Ready for it?", detail="Robot holds it until you ask",
        actions=[A("H_HANDOVER", "Give me", "hand", "primary"), CANCEL_HOLD],
    ),
    RobotTaskState.R_RECOVERY_EVALUATING: dict(
        screen="stopping", tone="alert", haptic="alert",
        title="Stopping…", detail="Checking a safe way back", actions=[],
    ),
    RobotTaskState.R_WAITING_HOME_PERMISSION: dict(
        screen="ask-home", tone="alert", haptic="ask",
        eyebrow="STOPPED", title="Return home?",
        detail="Path is clear to go home",
        actions=[A("H_RETURN_HOME", "Home", "home", "primary"),
                 A("H_MANUAL_RECOVERY", "By hand", "hand")],
    ),
    RobotTaskState.R_RETURNING_HOME: dict(
        screen="homing", tone="run", haptic="tap",
        title="Going home…", detail="Keep clear of the arm", actions=[],
    ),
    RobotTaskState.R_MANUAL_RECOVERY: dict(
        screen="manual", tone="hand", haptic="alert",
        eyebrow="MANUAL RECOVERY", title="Move arm by hand",
        detail="Free drive is on. Tap Done after",
        actions=[A("H_DONE", "Done", "check", "primary"),
                 A("H_CANCEL", "Abort", "x", "danger", hold=True)],
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
    | {"H_SPEEDUP", "H_SLOWDOWN", "H_EXECUTE_PENDING_TASK"}
)


def _allowed(state_machine, state, command: str, task_id) -> bool:
    event_type = EventType[command]
    if (state, event_type) in _HANDLER_ONLY:
        return True
    return state_machine.is_valid_transition(state, event_type, task_id)


def build_screen(task, pending: list, state_machine, now: float,
                 last_message: str = "") -> WatchScreen:
    """Return the screen for the active task (or the idle/pending screen)."""

    if task is None:
        if pending:
            first = pending[0]
            return WatchScreen(
                screen="pending", tone="idle", haptic="none",
                eyebrow=f"{len(pending)} WAITING",
                title=TASK_NAMES.get(first.task_id, "Pending task"),
                detail="Start it when you are ready",
                actions=[A("H_EXECUTE_PENDING_TASK", "Start now", "play", "primary")],
            )
        return WatchScreen(
            screen="idle", tone="idle", title="Robot ready",
            detail="I will ask when I can help",
        )

    spec = dict(_STATE_SCREENS.get(task.state, {}))
    if not spec:
        return WatchScreen(screen="unknown", tone="idle", title=task.state.name)

    name = TASK_NAMES.get(task.task_id, f"Task {task.task_id}")
    spec.setdefault("eyebrow", name.upper())
    if task.state is RobotTaskState.R_WAITING_RESPONSE:
        spec["title"] = QUESTIONS.get(task.task_id, name + "?")
        spec["detail"] = "Robot is ready to help"
    elif task.state is RobotTaskState.R_EXECUTING:
        spec["title"] = RUNNING_TITLES.get(task.task_id, name)
        spec["detail"] = "Robot is moving"

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
            screen.detail = "Releasing the panel"
        elif task.task_id == config.TASK_LEAVE_HANDOVER:
            screen.detail = "Moving away"
    return screen


def allowed_commands(task, pending: list, state_machine) -> set[str]:
    """Commands the watch may send right now (server-side allowlist)."""
    screen = build_screen(task, pending, state_machine, now=0.0)
    commands = {a.command for a in screen.actions}
    if screen.show_speed:
        commands |= {"H_SPEEDUP", "H_SLOWDOWN"}
    return commands
