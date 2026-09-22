"""Load/save camera calibration data (intrinsics, extrinsics) as JSON, and
the transforms that keep a stored calibration valid when the image it was
solved for changes shape.

Self-contained copy of world_pose/calibration/calibration_io.py's format
(json/pathlib/numpy only) so camera_utils has no dependency on the
world_pose package -- the two calibration files this reads/writes are
interchangeable with the ones world_pose's calibration scripts produce.
"""
import json
from pathlib import Path

import numpy as np

from logging_setup import get_logger

logger = get_logger(__name__)

# Below this relative difference between the horizontal and vertical scale
# factors, a resize counts as aspect-preserving. 0.5% covers rounding in
# common mode pairs (e.g. 1280x720 -> 854x480) without admitting a real
# aspect change like 16:9 -> 4:3.
_ASPECT_TOLERANCE = 0.005


def save_intrinsics(path, K, dist, image_size, reprojection_error=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "K": np.asarray(K, dtype=np.float64).tolist(),
        "dist": np.asarray(dist, dtype=np.float64).reshape(-1).tolist(),
        "image_size": list(image_size),  # (width, height)
        "reprojection_error": reprojection_error,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_intrinsics(path):
    with open(path) as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64)
    image_size = tuple(data["image_size"])
    return K, dist, image_size


def scale_intrinsics(K, image_size, new_image_size):
    """Re-express K for the same camera delivering a RESIZED image.

    image_size and new_image_size are (width, height). Returns
    (K_new, new_image_size), mirroring _rotate_intrinsics_90 in
    iphone_connection.py -- the same idea for scale instead of rotation.

    With sx = new_w / old_w and sy = new_h / old_h:

        fx' = sx * fx      cx' = (cx + 0.5) * sx - 0.5
        fy' = sy * fy      cy' = (cy + 0.5) * sy - 0.5

    The half-pixel offsets are cv2.resize's own pixel-area convention, where
    a pixel spans [-0.5, w-0.5] rather than [0, w-1]. It differs from a naive
    cx * sx by well under a pixel at usual sizes, but costs nothing to get
    right and matches what the resize actually does.

    Distortion coefficients are NOT returned because they do not change: they
    act on normalized coordinates ((u - cx)/fx, (v - cy)/fy), which are
    invariant once K is scaled along with the image.

    ONLY VALID FOR A RESIZE. A device that changes resolution by cropping its
    sensor instead keeps fx and fy and only shifts cx/cy, so this would be
    confidently wrong. A change of aspect ratio is the readable symptom of
    that, and is warned about -- 1920x1080 (16:9) to 640x480 (4:3) cannot be
    a pure resize, so something is being cropped or letterboxed.
    """
    K = np.asarray(K, dtype=np.float64)
    old_w, old_h = (float(v) for v in image_size)
    new_w, new_h = (int(v) for v in new_image_size)
    if old_w <= 0 or old_h <= 0:
        raise ValueError(f"image_size must be positive, got {tuple(image_size)}.")
    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"new_image_size must be positive, got {tuple(new_image_size)}.")

    sx = new_w / old_w
    sy = new_h / old_h
    if abs(sx - sy) > _ASPECT_TOLERANCE * max(sx, sy):
        logger.warning(
            "Rescaling intrinsics from %dx%d to %dx%d changes the aspect ratio "
            "(%.4f horizontally vs %.4f vertically). A camera changing resolution "
            "across aspect ratios usually CROPS or letterboxes rather than resizing, "
            "and under a crop fx/fy do not change at all -- only cx/cy shift. The "
            "result is correct only if the source really does rescale the full image.",
            int(old_w), int(old_h), new_w, new_h, sx, sy)

    K_new = K.copy()
    K_new[0, 0] = K[0, 0] * sx
    K_new[1, 1] = K[1, 1] * sy
    K_new[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K_new[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    # Skew, if a calibration ever carries one, scales with the x axis it shears.
    K_new[0, 1] = K[0, 1] * sx
    return K_new, (new_w, new_h)


def save_extrinsics(path, T_world_from_camera, ground_z=0.0,
                     T_world_from_robot_base=None, marker_id=None, notes=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "T_world_from_camera": np.asarray(T_world_from_camera, dtype=np.float64).tolist(),
        "ground_z": float(ground_z),
        "T_world_from_robot_base": (
            np.asarray(T_world_from_robot_base, dtype=np.float64).tolist()
            if T_world_from_robot_base is not None else None
        ),
        "marker_id": marker_id,
        "notes": notes,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_extrinsics(path):
    with open(path) as f:
        data = json.load(f)
    T_world_from_camera = np.array(data["T_world_from_camera"], dtype=np.float64)
    ground_z = float(data.get("ground_z", 0.0))
    T_world_from_robot_base = data.get("T_world_from_robot_base")
    if T_world_from_robot_base is not None:
        T_world_from_robot_base = np.array(T_world_from_robot_base, dtype=np.float64)
    return T_world_from_camera, ground_z, T_world_from_robot_base
