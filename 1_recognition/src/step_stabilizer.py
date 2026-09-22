from collections import deque

import numpy as np


class StepIdStabilizer:
    def __init__(
        self,
        num_steps=7,
        smoothing_window=5,
        confirmation_count=3,
        min_confidence=0.6,
        min_margin=0.15,
        allowed_transitions=None,
    ):
        self.num_steps = int(num_steps)
        self.prob_history = deque(maxlen=int(smoothing_window))
        self.confirmation_count = int(confirmation_count)
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.allowed_transitions = allowed_transitions or self._default_transitions(self.num_steps)

        self.stable_step_id = None
        self.pending_step_id = None
        self.pending_count = 0

    @staticmethod
    def _default_transitions(num_steps):
        transitions = {}
        for step_id in range(num_steps):
            next_steps = [step_id]
            if step_id + 1 < num_steps:
                next_steps.append(step_id + 1)
            transitions[step_id] = next_steps
        return transitions

    def update(self, probabilities):
        probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if probs.size != self.num_steps:
            raise ValueError(f"Expected {self.num_steps} probabilities, got {probs.size}")

        self.prob_history.append(probs)
        raw_candidate = int(np.argmax(probs))
        averaged_probs = np.mean(np.stack(self.prob_history), axis=0)
        candidate = int(np.argmax(averaged_probs))

        self._track_pending(raw_candidate)

        if not self._is_confident(averaged_probs, candidate):
            return self.stable_step_id

        if self.stable_step_id is None:
            self.stable_step_id = candidate
            self.pending_step_id = None
            self.pending_count = 0
            return self.stable_step_id

        if not self._is_allowed_transition(candidate):
            self._clear_pending()
            return self.stable_step_id

        if candidate == self.stable_step_id:
            self._clear_pending()
            return self.stable_step_id

        if candidate == self.pending_step_id and self.pending_count >= self.confirmation_count:
            self.stable_step_id = candidate
            self._clear_pending()

        return self.stable_step_id

    def _is_confident(self, averaged_probs, candidate):
        sorted_probs = np.sort(averaged_probs)
        top_prob = float(averaged_probs[candidate])
        second_prob = float(sorted_probs[-2]) if sorted_probs.size > 1 else 0.0
        return top_prob >= self.min_confidence and (top_prob - second_prob) >= self.min_margin

    def _is_allowed_transition(self, candidate):
        allowed = self.allowed_transitions.get(self.stable_step_id, [self.stable_step_id])
        return candidate in allowed

    def _clear_pending(self):
        self.pending_step_id = None
        self.pending_count = 0

    def _track_pending(self, candidate):
        if candidate == self.stable_step_id:
            self._clear_pending()
        elif candidate == self.pending_step_id:
            self.pending_count += 1
        else:
            self.pending_step_id = candidate
            self.pending_count = 1


# Backward-compatible name used by older runtime code and tests.
StepStabilizer = StepIdStabilizer
