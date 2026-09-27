"""Task transition probabilities, P(next task | current task), and task durations.

Loaded from the transition_probabilities.csv that
2_decision_making/src/task_sequence_analysis.py writes: rows are the current task
(plus START), columns the next task (plus END). Shared by the decision layer (which
tasks are pending) and the recognition layer (which step changes to believe).
The same script's duration_stats_*.csv give how long each task usually takes.
"""

import csv
from pathlib import Path

START = "START"
END = "END"


def load_duration_stat(path: str | Path, stat: str) -> dict[str, float]:
    """{task name: seconds}: one column (mean, median, p95, max, ...) of a
    duration_stats_*.csv."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and stat not in rows[0]:
        raise ValueError(f"{path} has no column {stat!r}; it has {', '.join(rows[0])}.")
    return {row["task_name"]: float(row[stat]) for row in rows}


class TransitionModel:
    """Read-only lookup over the observed task-order probabilities."""

    def __init__(self, probabilities: dict[str, dict[str, float]]):
        self._probabilities = probabilities

    @classmethod
    def from_csv(cls, path: str | Path) -> "TransitionModel":
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header = rows[0][1:]
        probabilities = {
            row[0]: {name: float(value) for name, value in zip(header, row[1:])}
            for row in rows[1:]
        }
        return cls(probabilities)

    def probability(self, current: str | None, following: str) -> float:
        """P(following | current); current None means nothing has happened yet."""
        return self._probabilities.get(current or START, {}).get(following, 0.0)

    def is_observed(self, current: str | None) -> bool:
        """Whether the table has any outgoing transition for this task."""
        return any(self._probabilities.get(current or START, {}).values())

    def ranked_next(self, current: str | None, candidates=None) -> list[tuple[str, float]]:
        row = self._probabilities.get(current or START, {})
        names = row if candidates is None else candidates
        ranked = [(name, row.get(name, 0.0)) for name in names if name != END]
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def allowed_transitions(self, step_names: list[str], min_probability: float) -> dict[int, list[int]]:
        """{step index: [step indices it may switch to]} for the recognition stabilizer.

        Staying on the same step is always allowed. A step the table never saw
        leave (no data) is left unrestricted rather than frozen.
        """
        allowed = {}
        for index, name in enumerate(step_names):
            if not self.is_observed(name):
                allowed[index] = list(range(len(step_names)))
                continue
            allowed[index] = [index] + [
                other for other, other_name in enumerate(step_names)
                if other != index and self.probability(name, other_name) >= min_probability
            ]
        return allowed
