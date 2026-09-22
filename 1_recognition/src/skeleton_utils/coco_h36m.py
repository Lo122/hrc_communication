"""Local COCO-17 -> H36M-17 keypoint remap -- no mmpose/mmcv dependency.

world_pose/pose/motionbert_lifter.py (the big multi-purpose venv) reaches
into mmpose's ``convert_keypoint_definition`` for this exact remap, since
mmpose/mmdet/mmcv are already hard dependencies of that project's "live"
extra anyway. This app/venv deliberately does NOT want that whole stack
(it only needs YOLO 2D detection + MotionBERT), so the same remap is
reimplemented here directly instead -- same index mapping/derivation logic
as world_pose's own confidence-only version (offline_lift_test.py's
convert_confidence_coco_to_h36m / motionbert_lifter.py's
_coco_conf_to_h36m), just also applied to xy positions (mean() for the
derived joints, instead of min() which only makes sense for a
confidence value).

H36M-17 order: 0=pelvis, 1-3=right hip/knee/ankle, 4-6=left hip/knee/ankle,
7=spine, 8=thorax, 9=neck, 10=head, 11-13=left shoulder/elbow/wrist,
14-16=right shoulder/elbow/wrist. Matches this whole project's H36M
convention (world_pose/demo/live_demo.py's H36M_SKELETON_EDGES).
"""
import numpy as np

_H36M_FROM_COCO_SRC = [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]
_H36M_FROM_COCO_DST = [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]


def coco_to_h36m_xy(keypoints_coco_xy):
    """keypoints_coco_xy: (17, 2). Returns (17, 2) H36M-order pixel coords."""
    keypoints_coco_xy = np.asarray(keypoints_coco_xy, dtype=np.float64)
    h36m = np.zeros((17, 2), dtype=np.float64)
    h36m[0] = (keypoints_coco_xy[11] + keypoints_coco_xy[12]) / 2.0   # pelvis <- l_hip, r_hip
    h36m[8] = (keypoints_coco_xy[5] + keypoints_coco_xy[6]) / 2.0     # thorax <- l_shoulder, r_shoulder
    h36m[7] = (h36m[0] + h36m[8]) / 2.0                               # spine <- pelvis, thorax
    h36m[10] = (keypoints_coco_xy[1] + keypoints_coco_xy[2]) / 2.0    # head <- l_eye, r_eye
    h36m[_H36M_FROM_COCO_DST] = keypoints_coco_xy[_H36M_FROM_COCO_SRC]
    return h36m


def coco_to_h36m_conf(keypoints_coco_conf):
    """keypoints_coco_conf: (17,). Returns (17,) H36M-order confidence --
    min() for derived joints (a joint averaged from two sources is only as
    trustworthy as its least-confident contributor)."""
    keypoints_coco_conf = np.asarray(keypoints_coco_conf, dtype=np.float64)
    h36m = np.zeros(17, dtype=np.float64)
    h36m[0] = min(keypoints_coco_conf[11], keypoints_coco_conf[12])
    h36m[8] = min(keypoints_coco_conf[5], keypoints_coco_conf[6])
    h36m[7] = min(h36m[0], h36m[8])
    h36m[10] = min(keypoints_coco_conf[1], keypoints_coco_conf[2])
    h36m[_H36M_FROM_COCO_DST] = keypoints_coco_conf[_H36M_FROM_COCO_SRC]
    return h36m
