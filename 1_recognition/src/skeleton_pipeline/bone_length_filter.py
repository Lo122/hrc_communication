"""Bone-length-constrained filter for lifted 3D skeletons -- copy of
world_pose/pose/bone_length_filter.py (pure numpy, no extra dependency).
See that module's docstring for the full reasoning: rescales each bone to a
slowly-adapting per-bone target length while leaving bone directions
(posture) untouched, countering the shrink/stretch monocular lifters
produce from frame-to-frame depth ambiguity.

Also doubles as this project's per-subject SCALE STABILIZER for LSTM
training-data generation (see app/generate_lstm_training_data.py): reset()
between videos/subjects so each person's own calibrated bone lengths are
independent, then read target_lengths() at the end of a clip for a stable,
per-subject scale summary to save alongside the (now length-stabilized)
per-frame skeleton -- see that script's docstring for why this specific
split (stabilize within a subject, but do NOT force a canonical size across
subjects) is the intended usage here.
"""
import numpy as np

H36M_BONE_TREE = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


class BoneLengthConstraintFilter:
    def __init__(self, length_alpha=0.02, min_length_ratio=0.7, max_length_ratio=1.3,
                 warmup_frames=10):
        self.length_alpha = length_alpha
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
        self.warmup_frames = warmup_frames
        self.reset()

    def reset(self):
        self._target_length = {}
        self._frames_seen = {}

    def target_lengths(self):
        """Returns {(parent, child): target_length} for every bone this
        filter has calibrated so far -- the per-subject scale summary to
        save as LSTM training metadata (see module docstring)."""
        return dict(self._target_length)

    def filter(self, skeleton_3d):
        if skeleton_3d is None or np.any(np.isnan(skeleton_3d)):
            return skeleton_3d

        skeleton_3d = np.asarray(skeleton_3d, dtype=np.float64)
        out = skeleton_3d.copy()
        for parent, child in H36M_BONE_TREE:
            vec = skeleton_3d[child] - skeleton_3d[parent]
            length = float(np.linalg.norm(vec))
            if length < 1e-8:
                out[child] = out[parent]
                continue
            direction = vec / length

            edge = (parent, child)
            target = self._target_length.get(edge)
            frames = self._frames_seen.get(edge, 0)
            if target is None:
                target = length
            elif frames < self.warmup_frames:
                alpha = 1.0 / (frames + 1)
                target = (1 - alpha) * target + alpha * length
            else:
                ratio = length / target
                if self.min_length_ratio <= ratio <= self.max_length_ratio:
                    target = (1 - self.length_alpha) * target + self.length_alpha * length
            self._target_length[edge] = target
            self._frames_seen[edge] = frames + 1

            out[child] = out[parent] + direction * target
        return out
