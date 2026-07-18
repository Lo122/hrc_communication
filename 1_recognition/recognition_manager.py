"""Recognition pipeline integration skeleton."""

import time

from models import RecognitionResult


class RecognitionManager:
    """Converts model or sensor output into a RecognitionResult."""

    def __init__(self, step_stabilizer=None):
        self.step_stabilizer = step_stabilizer
        # TODO: Load or inject the real recognition model.

    def update(self, input_data) -> RecognitionResult | None:
        """Return the latest standardized recognition result."""
        if input_data is None:
            return None

        # TODO: Replace dictionary passthrough with model inference.
        step_id = input_data.get("step_id", 0)
        if self.step_stabilizer is not None and "step_probabilities" in input_data:
            step_id = self.step_stabilizer.update(input_data["step_probabilities"])

        return RecognitionResult(
            round_id=input_data.get("round_id", 0),
            step_id=step_id,
            progress=input_data.get("progress", 0.0),
            piece_id=input_data.get("piece_id", 0),
            confidence=input_data.get("confidence", 0.0),
            timestamp=input_data.get("timestamp", time.time()),
        )
