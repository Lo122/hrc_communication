"""Load calibration data (intrinsics, extrinsics) as JSON -- copy of
world_pose/calibration/calibration_io.py's load side (json/os/numpy only,
no extra dependency). Save-side helpers aren't needed here since this
app/venv only ever reads calibration already produced by the main
world_pose project's calibration scripts.
"""
import json

import numpy as np


def load_intrinsics(path):
    with open(path) as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64)
    image_size = tuple(data["image_size"])
    return K, dist, image_size


def load_extrinsics(path):
    with open(path) as f:
        data = json.load(f)
    T_world_from_camera = np.array(data["T_world_from_camera"], dtype=np.float64)
    ground_z = float(data.get("ground_z", 0.0))
    T_world_from_robot_base = data.get("T_world_from_robot_base")
    if T_world_from_robot_base is not None:
        T_world_from_robot_base = np.array(T_world_from_robot_base, dtype=np.float64)
    return T_world_from_camera, ground_z, T_world_from_robot_base
