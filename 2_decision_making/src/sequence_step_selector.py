"""Which step the human is on, from recognition's scores and the task sequence.

The step models learned from videos of the human working alone. Beside the robot,
the human's motion during a robot-led step often looks like some other action, so the
model's first choice can be wrong while the right step scores close behind. The
decision layer knows the sequence, so it picks the step itself:

  1. keep only the tasks the human could be doing now (TaskTracker.potential_tasks:
     on their piece, not done, a human's task, plausible after the reference task);
  2. drop those scoring below their "Action Confidence Threshold" (task database);
  3. weight the rest by how much the sequence expects them:
         score = p * ((1 - strength) + strength * sequence_weight)
     with sequence_weight 1 for the reference task and P(task | reference) otherwise;
  4. take the best -- a switch away from the reference task only once it has won
     confirm_events updates in a row.

So a first option the sequence rules out gives way to a second option it expects,
if that one scores high enough; and nothing is chosen when no plausible task does
(the human is idle, or on something the sequence has no place for).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepChoice:
    """What the selector made of one task update. task_name None: no plausible task
    scored high enough. scores: {task: (p, weight, score)} of the tasks it weighed."""

    task_name: str | None
    progress: float
    model_task: str | None
    reason: str
    scores: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        """Whether the choice is not the model's own step."""
        return self.task_name != self.model_task


class SequenceStepSelector:
    def __init__(self, step_names, thresholds=None, default_threshold: float = 0.5,
                 strength: float = 0.5, confirm_events: int = 3):
        self.step_names = list(step_names)
        self.thresholds = dict(thresholds or {})
        self.default_threshold = float(default_threshold)
        self.strength = float(strength)
        self.confirm_events = max(int(confirm_events), 1)
        self._candidate: str | None = None
        self._candidate_wins = 0

    def threshold(self, task_name: str) -> float:
        return self.thresholds.get(task_name, self.default_threshold)

    def select(self, probabilities, step_progress, fallback_progress: float, tracker,
               model_task: str | None = None) -> StepChoice:
        """probabilities / step_progress: over step_names (step_progress may be None, for
        a model with one progress value: fallback_progress is then every step's)."""
        scores = {}
        for task in tracker.potential_tasks():
            if task.task_name not in self.step_names:
                continue
            p = float(probabilities[self.step_names.index(task.task_name)])
            if p < self.threshold(task.task_name):
                continue
            weight = (1.0 - self.strength) + self.strength * tracker.sequence_weight(task.task_name)
            scores[task.task_name] = (p, weight, p * weight)

        if not scores:
            self._candidate, self._candidate_wins = None, 0
            return StepChoice(None, 0.0, model_task, "no plausible task scores high enough")
        best = max(scores, key=lambda name: scores[name][2])
        reference = tracker.reference_task

        if best == reference:
            self._candidate, self._candidate_wins = None, 0
            reason = "the model's step" if best == model_task else "the reference task, by the sequence"
            return self._choice(best, step_progress, fallback_progress, model_task, reason, scores)

        self._candidate_wins = self._candidate_wins + 1 if best == self._candidate else 1
        self._candidate = best
        if self._candidate_wins < self.confirm_events:
            # Hold: the reference task while it still scores, else no task for now.
            reason = f"confirming {best} ({self._candidate_wins}/{self.confirm_events})"
            if reference in scores:
                return self._choice(reference, step_progress, fallback_progress, model_task,
                                    reason, scores)
            return StepChoice(None, 0.0, model_task, reason, scores)
        reason = "the model's step" if best == model_task else "by the sequence"
        return self._choice(best, step_progress, fallback_progress, model_task, reason, scores)

    def _choice(self, task_name, step_progress, fallback_progress, model_task, reason,
                scores) -> StepChoice:
        progress = (float(step_progress[self.step_names.index(task_name)])
                    if step_progress is not None else float(fallback_progress))
        return StepChoice(task_name, progress, model_task, reason, scores)
