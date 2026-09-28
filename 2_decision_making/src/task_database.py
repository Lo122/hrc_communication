"""Task database: pieces, their tasks, who can do each task, and robot trigger rules.

Reads 2_decision_making/task_database/task_database.json. Key spellings ("task pool",
"Task excutor", "Robot task trigger info", "previous task", "Progress", "Done signal",
"Condition", "Piece id", "robot task") are the file's own and are read as-is.

A trigger rule's task need not be on the pieces' task lists: a robot action like
"Leave from the panel" only exists when the robot held the panel, so it is not part
of any piece's assembly and does not count towards completing one.

A trigger rule, with n the piece the human is on:

    "Bring Connector": {
        "previous task": ["Screw"],            what triggers it, on piece n
        "Progress": 0.5,                       ...at this progress (default 0)
        "Done signal": true,                   ...or only once a previous task is done
        "Condition": {"Screw": [{"Done signal": true}, {"screw count": 0.5}]},
        "Piece id": "n",                       or "n + 1", or {"<previous task>": "n + 1"}
        "robot task": ["Bring Connector"]      what the robot does, in order
    }

"Condition" must hold as well as the trigger. Per task, a list gives alternatives
(any one will do) and the entries of one alternative must all hold. "Done signal" is
that task being done; any other name is a signal a detector reported on it
(task_transition_detector.py): a number passes at or above the value given,
true/false must match.

"Action Confidence Threshold" gives, per task, how sure recognition's step model must be
(0-1) before recognition switches to that step. Steps it leaves out use
STEP_MIN_CONFIDENCE (1_recognition/recognition_manager.py).
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

HUMAN = "Human"
ROBOT = "Robot"
DONE_SIGNAL = "Done signal"
# A detector reporting how far a task is (0-1), in place of recognition's progress.
PROGRESS_SIGNAL = "progress"
# A detector reporting that the held panel is fixed to the frame: the robot may release
# it and leave. Not the panel's Screw being done -- recognition or the human says that.
PANEL_SECURED = "panel secured"

_PIECE_EXPRESSION = re.compile(r"^\s*n\s*(?:([+-])\s*(\d+))?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Piece:
    piece_id: int
    location: str
    task_list: tuple[str, ...]


@dataclass(frozen=True)
class TriggerRule:
    """When the robot may offer task_name on its own, and what it does then.

    Triggered by one of previous_tasks on piece n (the piece the human is on): when
    it is the task the human is on at >= progress (0-1), or already done there. With
    done_signal ("Done signal": true), only once it is done there -- recognized
    progress alone never satisfies that, so progress is ignored then.

    conditions ("Condition"): {task: alternatives} that must hold on piece n as well
    (see the module docstring). piece_offsets ("Piece id"): the robot task is for piece
    n + offset, per previous task (0 when not given). robot_tasks ("robot task"): the
    robot's actions in order, starting with task_name; each next one is offered, with
    its own permission question, when the robot finishes the one before, for the same
    piece.
    """

    task_name: str
    previous_tasks: tuple[str, ...]
    progress: float
    done_signal: bool = False
    conditions: dict = field(default_factory=dict)
    piece_offsets: dict = field(default_factory=dict)
    robot_tasks: tuple[str, ...] = ()

    def piece_offset(self, previous_task: str) -> int:
        return self.piece_offsets.get(previous_task, 0)


def _parse_offset(expression, rule_name: str) -> int:
    match = _PIECE_EXPRESSION.match(str(expression))
    if match is None:
        raise ValueError(f"Invalid task database: trigger rule '{rule_name}': 'Piece id' "
                         f"{expression!r} is not n, n + k or n - k")
    sign, amount = match.groups()
    if amount is None:
        return 0
    return int(amount) if sign == "+" else -int(amount)


def _parse_piece_ids(value, previous_tasks: tuple[str, ...], rule_name: str) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {task: _parse_offset(expression, rule_name) for task, expression in value.items()}
    offset = _parse_offset(value, rule_name)
    return {task: offset for task in previous_tasks}


def _parse_conditions(value, rule_name: str) -> dict[str, tuple[dict, ...]]:
    if value is None:
        return {}
    problem = f"Invalid task database: trigger rule '{rule_name}': 'Condition' "
    if not isinstance(value, dict):
        raise ValueError(problem + "must map task names to requirements")
    conditions = {}
    for task, alternatives in value.items():
        if isinstance(alternatives, dict):
            alternatives = [alternatives]
        if (not isinstance(alternatives, list) or not alternatives
                or not all(isinstance(item, dict) and item for item in alternatives)):
            raise ValueError(problem + f"for '{task}' must be {{signal: value}} or a list of them")
        conditions[task] = tuple(dict(item) for item in alternatives)
    return conditions


class TaskDatabase:
    def __init__(self, pieces: list[Piece], executors: dict[str, tuple[str, ...]],
                 trigger_rules: dict[str, TriggerRule], confidence_thresholds: dict[str, float] | None = None):
        self.pieces = pieces
        self.executors = executors
        self.trigger_rules = trigger_rules
        # "Action Confidence Threshold": per task, how sure recognition's step model must
        # be before it switches to that step (1_recognition/src/step_stabilizer.py).
        self.confidence_thresholds = dict(confidence_thresholds or {})
        self._validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "TaskDatabase":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pieces = [
            Piece(int(piece["piece_id"]), piece.get("piece_location", ""), tuple(piece["task_list"]))
            for piece in data["task pool"]
        ]
        executors = {name: tuple(who) for name, who in data.get("Task excutor", {}).items()}
        trigger_rules = {}
        for name, rule in data.get("Robot task trigger info", {}).items():
            previous = tuple(rule.get("previous task", ()))
            trigger_rules[name] = TriggerRule(
                name, previous, float(rule.get("Progress", 0.0)), bool(rule.get(DONE_SIGNAL, False)),
                conditions=_parse_conditions(rule.get("Condition"), name),
                piece_offsets=_parse_piece_ids(rule.get("Piece id"), previous, name),
                robot_tasks=tuple(rule.get("robot task", ())) or (name,),
            )
        thresholds = {name: float(value) for name, value in data.get("Action Confidence Threshold", {}).items()}
        return cls(pieces, executors, trigger_rules, thresholds)

    def can_execute(self, task_name: str, executor: str) -> bool:
        return executor in self.executors.get(task_name, ())

    def in_task_lists(self, task_name: str) -> bool:
        """Whether any piece lists the task (and so tracks it per piece)."""
        return any(task_name in piece.task_list for piece in self.pieces)

    def next_robot_task(self, task_name: str) -> str | None:
        """The robot task that follows task_name in a "robot task" chain -- its own
        rule's first -- or None."""
        rules = sorted(self.trigger_rules.values(), key=lambda rule: rule.task_name != task_name)
        for rule in rules:
            chain = rule.robot_tasks
            if task_name in chain[:-1]:
                return chain[chain.index(task_name) + 1]
        return None

    def _validate(self) -> None:
        known = set(self.executors)
        problems = []
        for piece in self.pieces:
            problems += [f"piece {piece.piece_id}: '{name}' has no executor entry"
                         for name in piece.task_list if name not in known]
        for rule in self.trigger_rules.values():
            name = rule.task_name
            if name not in known:
                problems.append(f"trigger rule for unknown task '{name}'")
            problems += [f"trigger rule '{name}': unknown previous task '{task}'"
                         for task in rule.previous_tasks if task not in known]
            problems += [f"trigger rule '{name}': unknown condition task '{task}'"
                         for task in rule.conditions if task not in known]
            problems += [f"trigger rule '{name}': 'Piece id' names '{task}', which is not a previous task"
                         for task in rule.piece_offsets if task not in rule.previous_tasks]
            problems += [f"trigger rule '{name}': unknown robot task '{task}'"
                         for task in rule.robot_tasks if task not in known]
            if rule.robot_tasks and rule.robot_tasks[0] != name:
                problems.append(f"trigger rule '{name}': 'robot task' must start with '{name}'")
        problems += [f"'Action Confidence Threshold' for unknown task '{name}'"
                     for name in self.confidence_thresholds if name not in known]
        problems += [f"'Action Confidence Threshold' for '{name}' is {value}, not between 0 and 1"
                     for name, value in self.confidence_thresholds.items() if not 0.0 <= value <= 1.0]
        if problems:
            raise ValueError("Invalid task database: " + "; ".join(problems))
