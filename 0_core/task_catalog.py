"""The assembly workflow, expressed as data.

Why this file exists
--------------------
Before this, the workflow lived in three separate dicts in config.py
(TRIGGER_RULES, PERMISSION_MESSAGES, GH_STEP_MESSAGES), all keyed by
human step id. Nothing said what the robot does AFTER a task, so every
robot action had to be re-triggered by recognition.

The real ceiling-panel workflow is a chain:

    pull cable (R)
    -> lift panel (R)
    -> align       (R holds in free drive, H aligns)
    -> screw       (R keeps holding, H screws)
    -> R confirms the panel is secured, releases
    -> bring joint piece (R)
    -> cut pipes   (H)
    -> connect pipes (H)
    -> clamp joint (H)

Most of that is known in advance. Recognition only has to say WHEN the
human reaches the entry of a sequence; the workflow model carries the
rest forward. So a task here declares its own follow-up (`next_task`),
and only the sequence entries carry a recognition trigger.

Everything downstream -- trigger rules, spoken proposals, GH messages,
whether the task ends in free drive or in a hold-and-release gate --
is derived from this one table.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Human task steps, as the recognition model labels them.
#
# CONFIRM THESE AGAINST THE TRAINED MODEL before running on hardware:
# best_model/3d_skeleton/config.json currently declares num_steps = 7,
# so the ids below must match the label order used when that model was
# trained. Renaming here is free; renumbering is not.
# ---------------------------------------------------------------------------

STEP_PANEL_CUT = 0
STEP_PULL_CABLE = 1
STEP_ALIGN = 2
STEP_SCREW = 3
STEP_CUT_PIPE = 4
STEP_CONNECT_PIPE = 5
STEP_CLAMP_JOINT = 6

STEP_NAMES = {
    STEP_PANEL_CUT: "cutting the panel",
    STEP_PULL_CABLE: "pulling the cable",
    STEP_ALIGN: "aligning the panel",
    STEP_SCREW: "screwing",
    STEP_CUT_PIPE: "cutting the pipes",
    STEP_CONNECT_PIPE: "connecting the pipes",
    STEP_CLAMP_JOINT: "clamping the joint",
}


# ---------------------------------------------------------------------------
# How a robot task ends.
# ---------------------------------------------------------------------------

#: Robot reaches the goal, reports success, task is done.
MODE_SIMPLE = "simple"

#: Robot reaches the goal and then hands the arm to the human in free
#: drive so they can position it by hand (the align step).
MODE_FREE_DRIVE = "free_drive"

# ---------------------------------------------------------------------------
# How much consent a task needs before it runs.
# ---------------------------------------------------------------------------

#: The robot asks and waits for an answer. Use where the task starts
#: something new, or takes on load the robot was not already carrying.
PERMISSION_ASK = "ask"

#: The robot says what it is about to do, waits a short veto window, then
#: goes. Use where the human has already consented to the situation and a
#: second question is friction rather than consent -- continuing to hold a
#: panel it is already holding, fetching a part away from the person.
PERMISSION_ANNOUNCE = "announce"


#: Robot reaches the goal and HOLDS there. It must not let go until the
#: human confirms the panel is secured -- this is the "make sure it will
#: not fall" gate. See RobotTaskState.R_HOLDING.
MODE_HOLD_UNTIL_SECURED = "hold_until_secured"


@dataclass(frozen=True)
class RobotTaskSpec:
    """One robot assistance task and everything the system says about it."""

    key: str

    #: Value sent to Grasshopper as `suggested_action`; picks the trajectory template.
    gh_action: str

    #: Short human-readable name, used inside generated sentences.
    label: str

    #: What the robot asks when it proposes the task.
    proposal: str

    #: Said once the human accepts and the task is dispatched.
    on_accept: str

    #: Said when the robot reports success.
    on_success: str

    #: How the task ends.
    mode: str = MODE_SIMPLE

    #: Human step that triggers this task from recognition. None = the task
    #: is only ever reached as another task's follow-up.
    trigger_step: int | None = None

    #: Normalized progress within `trigger_step` at which to propose.
    progress_threshold: float = 0.5

    #: Minimum recognition confidence before a trigger is allowed.
    min_confidence: float = 0.8

    #: Task proposed automatically once this one completes.
    next_task: str | None = None

    #: Plain-language reason the robot is offering now. Logged with the
    #: trigger and available to the UI -- this is what makes a proposal
    #: explainable rather than magic.
    reason: str = ""

    #: Only used by MODE_HOLD_UNTIL_SECURED: what the robot asks while holding.
    hold_prompt: str = ""

    #: Whether this task asks and waits, or announces and proceeds.
    #: This is the proactive/reactive dial from the thesis, made per-task:
    #: asking every time is safe but not fluent, announcing every time is
    #: fluent but takes agency away. Worth varying in the user study.
    permission: str = PERMISSION_ASK

    #: True when the robot is carrying or supporting something during this
    #: task, so cancelling means dropping it. These are the only tasks where
    #: a cancel is made to ask twice -- cancelling a trip to fetch a tool is
    #: harmless, cancelling a panel hold is not.
    carries_load: bool = False


# ---------------------------------------------------------------------------
# The catalog.
#
# Only PULL_CABLE and BRING_CLAMP carry a trigger_step: they are the two
# points where the human's own activity decides the timing. Everything
# between them is chained -- which is precisely the "robot anticipates"
# claim, made explicit instead of implied.
# ---------------------------------------------------------------------------

TASKS: dict[str, RobotTaskSpec] = {
    "pull_cable": RobotTaskSpec(
        key="pull_cable",
        gh_action="assist_pull_cable",
        label="pulling the cable",
        proposal="You are almost done cutting. Shall I pull the cable through?",
        on_accept="Pulling the cable through now.",
        on_success="Cable is through.",
        mode=MODE_SIMPLE,
        trigger_step=STEP_PANEL_CUT,
        progress_threshold=0.7,
        min_confidence=0.8,
        next_task="lift_panel",
        reason="panel cut is nearly finished, so the cable is the next thing needed",
    ),
    "lift_panel": RobotTaskSpec(
        key="lift_panel",
        gh_action="assist_lifting",
        label="lifting the panel",
        proposal="Shall I lift the panel up to the frame?",
        on_accept="Lifting the panel. Stay clear until it is up.",
        on_success="Panel is up at the frame.",
        mode=MODE_SIMPLE,
        trigger_step=None,
        next_task="hold_for_align",
        reason="the cable is through, so the panel can go up",
        carries_load=True,
    ),
    "hold_for_align": RobotTaskSpec(
        key="hold_for_align",
        gh_action="hold_for_align",
        label="holding for alignment",
        proposal="Shall I go soft so you can align it by hand?",
        on_accept="Going soft. Move the panel where you want it.",
        on_success="Alignment locked in.",
        mode=MODE_FREE_DRIVE,
        trigger_step=None,
        next_task="hold_for_screw",
        reason="the panel is up and needs positioning before it can be fixed",
        carries_load=True,
        permission=PERMISSION_ANNOUNCE,
    ),
    "hold_for_screw": RobotTaskSpec(
        key="hold_for_screw",
        gh_action="hold_for_screw",
        label="holding the panel",
        proposal="Shall I hold it while you screw?",
        on_accept="Holding it. Take your time.",
        on_success="I have the panel. Screw it in.",
        mode=MODE_HOLD_UNTIL_SECURED,
        trigger_step=None,
        next_task="bring_joint",
        reason="the panel is aligned and must be held steady while it is fixed",
        hold_prompt="Still holding. Say secured when it is fixed and I can let go.",
        carries_load=True,
        permission=PERMISSION_ANNOUNCE,
    ),
    "bring_joint": RobotTaskSpec(
        key="bring_joint",
        gh_action="bring_joint",
        label="bringing the joint piece",
        proposal="Panel is secured. Shall I bring the joint piece?",
        on_accept="Bringing the joint piece.",
        on_success="Joint piece is here.",
        mode=MODE_SIMPLE,
        trigger_step=None,
        next_task=None,
        reason="the panel is fixed, so the pipe work is next",
        permission=PERMISSION_ANNOUNCE,
    ),
    "bring_clamp": RobotTaskSpec(
        key="bring_clamp",
        gh_action="bring_clamp_tool",
        label="bringing the clamp tool",
        proposal="Shall I bring the clamp tool?",
        on_accept="Bringing the clamp tool.",
        on_success="Clamp tool is here.",
        mode=MODE_SIMPLE,
        trigger_step=STEP_CONNECT_PIPE,
        progress_threshold=0.6,
        min_confidence=0.8,
        next_task=None,
        reason="the pipes are nearly connected and the joint has to be clamped next",
    ),
    "take_clamp_back": RobotTaskSpec(
        key="take_clamp_back",
        gh_action="receive_clamp_tool",
        label="taking the clamp tool back",
        proposal="Shall I take the clamp tool back?",
        on_accept="Reaching for the clamp tool.",
        on_success="Clamp tool is stored.",
        mode=MODE_SIMPLE,
        trigger_step=STEP_CLAMP_JOINT,
        progress_threshold=0.8,
        min_confidence=0.8,
        next_task="bring_next_panel",
        reason="the joint is clamped, so the tool is no longer needed",
    ),
    "bring_next_panel": RobotTaskSpec(
        key="bring_next_panel",
        gh_action="bring_next_panel",
        label="bringing the next panel",
        proposal="This one is finished. Shall I bring the next panel?",
        on_accept="Bringing the next panel.",
        on_success="Next panel is on the table.",
        mode=MODE_SIMPLE,
        trigger_step=None,
        next_task=None,
        reason="this panel cycle is complete",
        permission=PERMISSION_ANNOUNCE,
    ),
}


# ---------------------------------------------------------------------------
# Derived lookups. Everything else in the system should read these rather
# than hard-coding a step id.
# ---------------------------------------------------------------------------

def task_by_key(key: str) -> RobotTaskSpec | None:
    return TASKS.get(key)


def triggered_tasks() -> dict[int, RobotTaskSpec]:
    """Human step id -> the task recognition should propose at that step.

    The trigger manager assumes at most one rule per step, so a second
    task claiming the same trigger_step is a configuration error and is
    reported loudly rather than silently overwriting the first.
    """
    result: dict[int, RobotTaskSpec] = {}
    for spec in TASKS.values():
        if spec.trigger_step is None:
            continue
        if spec.trigger_step in result:
            raise ValueError(
                f"Two tasks trigger on step {spec.trigger_step}: "
                f"{result[spec.trigger_step].key} and {spec.key}. "
                "The trigger manager supports one rule per step."
            )
        result[spec.trigger_step] = spec
    return result


def trigger_rules() -> dict[int, dict]:
    """The shape config.TRIGGER_RULES used to have."""
    return {
        step_id: {
            "progress_threshold": spec.progress_threshold,
            "min_confidence": spec.min_confidence,
            "task_key": spec.key,
        }
        for step_id, spec in triggered_tasks().items()
    }


def step_name(step_id: int) -> str:
    return STEP_NAMES.get(step_id, f"step {step_id}")
