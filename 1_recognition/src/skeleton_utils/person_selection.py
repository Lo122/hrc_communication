"""Pick the person detection that best matches the previously tracked pose."""
import numpy as np


def _ensure_keypoint_conf(keypoints_xy, keypoints_conf):
    if keypoints_conf is None:
        return np.ones(np.asarray(keypoints_xy).shape[0], dtype=np.float32)
    return np.asarray(keypoints_conf, dtype=np.float32)


def _tracking_points(keypoints_xy, keypoints_conf, conf_threshold):
    keypoints_xy = np.asarray(keypoints_xy, dtype=np.float32)
    if keypoints_xy.ndim != 2 or keypoints_xy.shape[-1] != 2:
        return np.empty((0, 2), dtype=np.float32)

    finite = np.isfinite(keypoints_xy).all(axis=1)
    keypoints_conf = _ensure_keypoint_conf(keypoints_xy, keypoints_conf)
    confident = finite & np.isfinite(keypoints_conf) & (keypoints_conf >= conf_threshold)

    points = keypoints_xy[confident]
    if points.shape[0] > 0:
        return points
    return keypoints_xy[finite]


def _tracking_center(keypoints_xy, keypoints_conf, conf_threshold):
    points = _tracking_points(keypoints_xy, keypoints_conf, conf_threshold)
    if points.shape[0] == 0:
        return None
    return points.mean(axis=0)


def _tracking_scale(keypoints_xy, keypoints_conf, conf_threshold):
    points = _tracking_points(keypoints_xy, keypoints_conf, conf_threshold)
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _valid_joint_mask(keypoints_xy, keypoints_conf, conf_threshold):
    keypoints_xy = np.asarray(keypoints_xy, dtype=np.float32)
    keypoints_conf = _ensure_keypoint_conf(keypoints_xy, keypoints_conf)
    return (
        np.isfinite(keypoints_xy).all(axis=1)
        & np.isfinite(keypoints_conf)
        & (keypoints_conf >= conf_threshold)
    )


def _pose_jump_metrics(keypoints_xy, keypoints_conf, previous_keypoints, previous_conf, conf_threshold):
    if previous_keypoints is None:
        return None

    keypoints_xy = np.asarray(keypoints_xy, dtype=np.float32)
    previous_keypoints = np.asarray(previous_keypoints, dtype=np.float32)
    if keypoints_xy.shape != previous_keypoints.shape:
        return None

    current_mask = _valid_joint_mask(keypoints_xy, keypoints_conf, conf_threshold)
    previous_mask = _valid_joint_mask(previous_keypoints, previous_conf, conf_threshold)
    shared_mask = current_mask & previous_mask
    if not np.any(shared_mask):
        return None

    displacements = np.linalg.norm(keypoints_xy[shared_mask] - previous_keypoints[shared_mask], axis=1)
    return float(displacements.mean()), float(displacements.max())


def _normalized_detection_scores(all_keypoints_conf, detection_scores):
    if detection_scores is not None:
        scores = np.asarray(detection_scores, dtype=np.float32)
        if scores.ndim == 1 and scores.shape[0] > 0:
            return np.where(np.isfinite(scores), scores, 0.0)

    if all_keypoints_conf is None:
        return np.ones(0, dtype=np.float32)

    scores = np.nanmean(np.where(np.isfinite(all_keypoints_conf), all_keypoints_conf, np.nan), axis=1)
    return np.where(np.isfinite(scores), scores, 0.0).astype(np.float32)


def _candidate_indices(all_keypoints_xy, all_keypoints_conf, scores, conf_threshold,
                       min_valid_joints, min_detection_score):
    finite = np.isfinite(all_keypoints_xy).all(axis=2)
    confident = finite & np.isfinite(all_keypoints_conf) & (all_keypoints_conf >= conf_threshold)
    confident_counts = confident.sum(axis=1)

    candidates = np.flatnonzero(
        (confident_counts >= min_valid_joints) & (scores >= min_detection_score)
    )
    if candidates.size:
        return candidates

    candidates = np.flatnonzero(confident_counts >= min_valid_joints)
    if candidates.size:
        return candidates

    candidates = np.flatnonzero(finite.any(axis=1))
    if candidates.size:
        return candidates

    return np.arange(all_keypoints_xy.shape[0])


def select_tracked_person_keypoints(all_keypoints_xy, all_keypoints_conf=None, detection_scores=None,
                                    previous_keypoints=None, previous_conf=None, conf_threshold=0.2,
                                    min_valid_joints=4, min_detection_score=0.25,
                                    max_jump_ratio=None):
    """Choose the detection closest to the previous pose, normalized by body size.

    The selection order is:
      1) filter out detections with too few confident joints or weak person score
      2) among the remaining people, minimize per-joint jump from the last pose
      3) break ties by center distance, then detection score

    If no previous pose is available, falls back to the highest-score detection.

    If ``max_jump_ratio`` is set and a previous pose is available, the chosen
    detection is compared against that previous pose: when its mean or max
    per-joint displacement (normalized by body scale) exceeds
    ``max_jump_ratio``, this is treated as a likely wrong-person pick (e.g. a
    bystander overlapping the target) rather than genuine fast motion, and
    the previous pose is returned unchanged instead of the jumped detection.
    """
    all_keypoints_xy = np.asarray(all_keypoints_xy, dtype=np.float32)
    if all_keypoints_xy.ndim != 3 or all_keypoints_xy.shape[0] == 0:
        return None, None

    if all_keypoints_conf is None:
        all_keypoints_conf = np.ones(all_keypoints_xy.shape[:2], dtype=np.float32)
    else:
        all_keypoints_conf = np.asarray(all_keypoints_conf, dtype=np.float32)

    scores = _normalized_detection_scores(all_keypoints_conf, detection_scores)
    if scores.shape[0] != all_keypoints_xy.shape[0]:
        scores = np.nanmean(np.where(np.isfinite(all_keypoints_conf), all_keypoints_conf, np.nan), axis=1)
        scores = np.where(np.isfinite(scores), scores, 0.0).astype(np.float32)

    candidate_indices = _candidate_indices(
        all_keypoints_xy,
        all_keypoints_conf,
        scores,
        conf_threshold,
        min_valid_joints,
        min_detection_score,
    )

    previous_center = _tracking_center(previous_keypoints, previous_conf, conf_threshold)
    previous_scale = _tracking_scale(previous_keypoints, previous_conf, conf_threshold)
    if previous_center is None or previous_scale <= 1e-6:
        best_idx = int(candidate_indices[np.argmax(scores[candidate_indices])])
        return all_keypoints_xy[best_idx], all_keypoints_conf[best_idx]

    best_idx = None
    best_metric = None
    for idx in candidate_indices:
        center = _tracking_center(all_keypoints_xy[idx], all_keypoints_conf[idx], conf_threshold)
        if center is None:
            continue

        scale = _tracking_scale(all_keypoints_xy[idx], all_keypoints_conf[idx], conf_threshold)
        scale_norm = max(previous_scale, scale, 1.0)
        normalized_center_distance = float(np.linalg.norm(center - previous_center)) / scale_norm
        pose_jump = _pose_jump_metrics(
            all_keypoints_xy[idx],
            all_keypoints_conf[idx],
            previous_keypoints,
            previous_conf,
            conf_threshold,
        )
        if pose_jump is None:
            metric = (
                1,
                normalized_center_distance,
                float("inf"),
                normalized_center_distance,
                -float(scores[idx]),
            )
        else:
            mean_jump, max_jump = pose_jump
            metric = (
                0,
                mean_jump / scale_norm,
                max_jump / scale_norm,
                normalized_center_distance,
                -float(scores[idx]),
            )
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_idx = idx

    if best_idx is None:
        best_idx = int(candidate_indices[np.argmax(scores[candidate_indices])])

    if max_jump_ratio is not None:
        chosen_scale = _tracking_scale(all_keypoints_xy[best_idx], all_keypoints_conf[best_idx], conf_threshold)
        scale_norm = max(previous_scale, chosen_scale, 1.0)
        pose_jump = _pose_jump_metrics(
            all_keypoints_xy[best_idx],
            all_keypoints_conf[best_idx],
            previous_keypoints,
            previous_conf,
            conf_threshold,
        )
        if pose_jump is not None:
            mean_jump, max_jump = pose_jump
            if (mean_jump / scale_norm) > max_jump_ratio or (max_jump / scale_norm) > max_jump_ratio:
                # Best-matching detection still looks like a different
                # person (or a garbage detection) -- hold the previous pose
                # instead of jumping onto it.
                return previous_keypoints, previous_conf

    return all_keypoints_xy[best_idx], all_keypoints_conf[best_idx]
