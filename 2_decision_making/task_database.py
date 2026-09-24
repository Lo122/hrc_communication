"""Task database: pieces, their tasks, who can do each task, and robot trigger rules.

Reads 2_decision_making/task_database/task_database.json. Key spellings ("task pool",
"Task excutor", "Robot task trigger info", "previous task", "Progress") are the
file's own and are read as-is.
"""

import json
from dataclasses import dataclass
from pathlib import Path

HUMAN = "Human"
ROBOT = "Robot"


@dataclass(frozen=True)
class Piece:
    piece_id: int
    location: str
    task_list: tuple[str, ...]


@dataclass(frozen=True)
class TriggerRule:
    """A robot task may be offered once the human's task is one of previous_tasks
    and has reached progress (0-1)."""

    task_name: str
    previous_tasks: tuple[str, ...]
    progress: float


class TaskDatabase:
    def __init__(self, pieces: list[Piece], executors: dict[str, tuple[str, ...]],
                 trigger_rules: dict[str, TriggerRule]):
        self.pieces = pieces
        self.executors = executors
        self.trigger_rules = trigger_rules
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
        trigger_rules = {
            name: TriggerRule(name, tuple(rule.get("previous task", ())), float(rule.get("Progress", 0.0)))
            for name, rule in data.get("Robot task trigger info", {}).items()
        }
        return cls(pieces, executors, trigger_rules)

    def can_execute(self, task_name: str, executor: str) -> bool:
        return executor in self.executors.get(task_name, ())

    def _validate(self) -> None:
        known = set(self.executors)
        problems = []
        for piece in self.pieces:
            problems += [f"piece {piece.piece_id}: '{name}' has no executor entry"
                         for name in piece.task_list if name not in known]
        for rule in self.trigger_rules.values():
            if rule.task_name not in known:
                problems.append(f"trigger rule for unknown task '{rule.task_name}'")
            problems += [f"trigger rule '{rule.task_name}': unknown previous task '{name}'"
                         for name in rule.previous_tasks if name not in known]
        if problems:
            raise ValueError("Invalid task database: " + "; ".join(problems))
