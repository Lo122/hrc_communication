"""2D-keypoint hold/outlier-reject filter, applied between the 2D detector
(YOLO) and the 3D lifter. Copy of world_pose/pose/keypoint_filter.py
(pure numpy, no extra dependency) -- see that module's docstring for the
full accept/hold/lost state-machine reasoning.
"""
import numpy as np


class KeypointOutlierHoldFilter:
    def __init__(self, max_jump_ratio=1.5, confirm_frames=3, max_hold_frames=30,
                 conf_threshold=0.3, min_valid_joints=4):
        self.max_jump_ratio = max_jump_ratio
        self.confirm_frames = confirm_frames
        self.max_hold_frames = max_hold_frames
        self.conf_threshold = conf_threshold
        self.min_valid_joints = min_valid_joints
        self.reset()

    def reset(self):
        self._prev_keypoints = None
        self._prev_conf = None
        self._frames_since_accepted = 0
        self._consecutive_rejects = 0

    def filter(self, keypoints_2d, keypoints_conf):
        num_confident = 0 if keypoints_conf is None else int(np.sum(keypoints_conf >= self.conf_threshold))
        has_input = (
            keypoints_2d is not None
            and np.any(keypoints_2d)
            and num_confident >= self.min_valid_joints
        )

        if not has_input:
            return self._hold("held_no_detection")

        if self._prev_keypoints is None:
            self._accept(keypoints_2d, keypoints_conf)
            return keypoints_2d, keypoints_conf, "accepted"

        displacement, body_scale = self._compute_displacement(keypoints_2d, keypoints_conf)
        if displacement is None or body_scale <= 1e-6:
            return self._hold("held_no_detection")

        if displacement <= self.max_jump_ratio * body_scale:
            self._accept(keypoints_2d, keypoints_conf)
            return keypoints_2d, keypoints_conf, "accepted"

        self._consecutive_rejects += 1
        if self._consecutive_rejects >= self.confirm_frames:
            self._accept(keypoints_2d, keypoints_conf)
            return keypoints_2d, keypoints_conf, "accepted"

        return self._hold("held_outlier")

    def _hold(self, status):
        self._frames_since_accepted += 1
        if self._prev_keypoints is not None and self._frames_since_accepted <= self.max_hold_frames:
            return self._prev_keypoints, self._prev_conf, status
        self.reset()
        return None, None, "lost"

    def _accept(self, keypoints_2d, keypoints_conf):
        self._prev_keypoints = keypoints_2d
        self._prev_conf = keypoints_conf
        self._frames_since_accepted = 0
        self._consecutive_rejects = 0

    def _compute_displacement(self, keypoints_2d, keypoints_conf):
        both_confident = (keypoints_conf >= self.conf_threshold) & (self._prev_conf >= self.conf_threshold)
        if not np.any(both_confident):
            return None, 0.0
        displacement = float(
            np.linalg.norm(keypoints_2d[both_confident] - self._prev_keypoints[both_confident], axis=-1).mean())

        confident_now = keypoints_2d[keypoints_conf >= self.conf_threshold]
        if confident_now.shape[0] < 2:
            return displacement, 0.0
        body_scale = float(np.linalg.norm(confident_now.max(axis=0) - confident_now.min(axis=0)))
        return displacement, body_scale
