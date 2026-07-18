"""Step stabilization skeleton.

The filename keeps the current project spelling. A later cleanup can rename it
to step_stabilizer.py and update imports.
"""


class StepStabilizer:
    """Selects a stable step from noisy step probabilities."""

    def __init__(self, min_confidence: float = 0.0, confirmation_count: int = 1):
        self.min_confidence = min_confidence
        self.confirmation_count = confirmation_count
        self._candidate_step = None
        self._candidate_count = 0
        self._stable_step = None

    def update(self, step_probabilities) -> int | None:
        """Return a stable step id when confidence and count are sufficient."""
        if not step_probabilities:
            return self._stable_step

        # TODO: Replace max selection with the project's real smoothing logic.
        step_id, confidence = max(
            enumerate(step_probabilities),
            key=lambda item: item[1],
        )

        if confidence < self.min_confidence:
            return self._stable_step

        if step_id == self._candidate_step:
            self._candidate_count += 1
        else:
            self._candidate_step = step_id
            self._candidate_count = 1

        if self._candidate_count >= self.confirmation_count:
            self._stable_step = step_id

        return self._stable_step
