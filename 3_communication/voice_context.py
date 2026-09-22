"""Reviewable GPT instructions for each human-robot interaction stage."""

from dataclasses import dataclass

import config
from events import RobotTaskState as S


@dataclass(frozen=True)
class VoiceContext:
    """Snapshot taken on the runtime thread, never a mutable RobotTask."""

    state: S | None
    task_id: int | None = None
    task_instance_id: str | None = None


BASE_INSTRUCTIONS = """You interpret spoken human intent for a robot collaboration task.
Output exactly one lowercase command from the permitted commands below, or unknown.
Output only the English command, with no explanation.
Use the current interaction context to interpret short or indirect replies.
The robot lifts a panel, the human may adjust it in free drive, and the robot
then holds it while the human screws it in place. Adjustment completion and
screwing completion are separate milestones.
Explicit statements and negation take precedence over contextual assumptions.
For example, 'I finished adjusting but have not finished screwing' does not
mean that screwing is complete. Do not infer agreement or completion from
silence, background noise, unclear audio, or the system's own messages.
If the request is ambiguous or has no permitted command, output unknown.
The current context replaces any earlier task or stage in the conversation.
"""

TASK_DESCRIPTIONS = {
    config.TASK_LIFT_PANEL: "lift the panel, offer manual adjustment, then hold it",
    config.TASK_LEAVE: "release the panel and move away",
    config.TASK_BRING_CONNECTOR: "bring the pipe connector",
    config.TASK_BRING_CLAMPING_TOOL: "bring the clamping tool",
    config.TASK_RETURN_CLAMPING_TOOL: "take the clamping tool back",
}

# Each entry defines the meaning of the stage and its canonical output commands.
STATE_CONTEXTS = {
    S.R_WAITING_RESPONSE: (
        "The robot is asking permission to start the current task. Agreement "
        "such as yes, okay or go ahead means yes; refusal means no; a request "
        "to wait until later means later. A completion statement alone is not permission.",
        ("yes", "no", "later"),
    ),
    S.R_WAITING_FREE_DRIVE: (
        "The panel has been lifted. The robot is asking whether the human wants "
        "free drive for manual panel adjustment. Yes, okay or go ahead means "
        "free drive; declining adjustment means no. Screwing completion is not being requested.",
        ("free drive", "no", "cancel"),
    ),
    S.R_FREE_DRIVE: (
        "Free drive is enabled. The human is adjusting the panel. A general "
        "completion reply such as done, finished or I'm done means panel adjustment "
        "is finished: output done. An explicit statement about finishing screwing "
        "alone is not adjustment completion and must not be converted to done.",
        ("done", "cancel"),
    ),
    S.R_HOLDING: (
        "The robot is holding the panel while the human screws it in place. "
        "It is waiting for screwing to finish. A general completion reply such "
        "as done, finished or I'm done means screwing is finished: output screw done. "
        "Finishing adjustment alone does not mean screwing is finished. "
        "This command requests a separate permission question before the robot leaves.",
        ("screw done", "cancel"),
    ),
    S.R_WAITING_HOME_PERMISSION: (
        "The robot has stopped and asks permission to return home. Agreement "
        "means return home. Refusal means no; an explicit request for manual "
        "recovery means manual recovery. This is not permission to restart the task.",
        ("return home", "no", "manual recovery"),
    ),
    S.R_MANUAL_RECOVERY: (
        "Free drive is enabled for manual recovery after cancellation. A general "
        "completion reply means manual recovery is finished: output done. "
        "Do not interpret it as screwing completion.",
        ("done", "cancel"),
    ),
    S.R_EXECUTING: (
        "The current robot task is executing. Interpret requests to pause/stop, "
        "cancel the task, restart it, or change speed using the matching command. "
        "Human completion means done; it does not mean the robot task has completed.",
        ("pause", "cancel", "restart", "faster", "slower", "done"),
    ),
    S.R_PAUSED: (
        "The current robot task is paused. Continue or resume means resume. "
        "Restart means restart from the beginning. Cancel means abandon the task.",
        ("resume", "restart", "cancel"),
    ),
    S.R_DEFER: (
        "The task is delayed and will start when its defer timer expires. "
        "An explicit request to cancel the delayed task means cancel.",
        ("cancel",),
    ),
}


def build_instructions(context: VoiceContext, question: str | None = None) -> str:
    description, commands = STATE_CONTEXTS.get(
        context.state, ("No voice interaction is currently active.", ()),
    )
    task = TASK_DESCRIPTIONS.get(context.task_id, "unspecified robot task")
    state = context.state.name if context.state is not None else "none"
    parts = [
        BASE_INSTRUCTIONS,
        f"Current task: {task} (task_id={context.task_id}).",
        f"Task instance: {context.task_instance_id or 'none'}.",
        f"Current state: {state}.",
        description,
        "Permitted commands: " + ", ".join((*commands, "unknown")) + ".",
    ]
    if question:
        parts.append("Current system question (context, not a human reply):\n" + question)
    return "\n\n".join(parts)
