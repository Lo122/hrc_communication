"""Load/save camera calibration data (intrinsics, extrinsics) as JSON.

Self-contained copy of world_pose/calibration/calibration_io.py's format
(json/pathlib/numpy only) so camera_utils has no dependency on the
world_pose package -- the two calibration files this reads/writes are
interchangeable with the ones world_pose's calibration scripts produce.
"""
import json
from pathlib import Path

import numpy as np


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
