"""Rigid-transform math for extrinsic calibration and for projecting
detections into the calibrated world frame (pose_detection_live.py).

Self-contained subset of world_pose/geometry/transforms.py (numpy + cv2
only) -- see that module's docstring for the full convention description:
a transform ``T_a_from_b`` is a 4x4 homogeneous matrix mapping points in
frame ``b`` into frame ``a``.
"""
import cv2
import numpy as np


def make_transform(R, t):
    """Build a 4x4 transform from a 3x3 rotation matrix and a length-3 translation."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rvec_tvec_to_transform(rvec, tvec):
    """solvePnP returns the pose of the *object* frame expressed in the
    *camera* frame, i.e. this returns T_camera_from_object."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return make_transform(R, tvec)


def invert_transform(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return make_transform(R_inv, t_inv)


def compose_transforms(*transforms):
    """Compose transforms left-to-right: compose(T_a_from_b, T_b_from_c) -> T_a_from_c."""
    out = np.eye(4, dtype=np.float64)
    for T in transforms:
        out = out @ np.asarray(T, dtype=np.float64)
    return out


def rpy_deg_to_matrix(roll_deg, pitch_deg, yaw_deg):
    """Intrinsic rotation R = Rz(yaw) @ Ry(pitch) @ Rx(roll), degrees in."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def transform_points(T, points):
    """Apply a 4x4 transform to an array of 3D points, shape (..., 3)."""
    T = np.asarray(T, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def transform_directions(T, directions):
    """Rotate (but do not translate) direction vectors by the rotation part of T."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    directions = np.asarray(directions, dtype=np.float64)
    return directions @ R.T


def pixel_depth_to_camera_point(K, pixel_xy, depth):
    """Back-project a pixel with a known metric depth (along +Z) to a
    camera-frame 3D point (OpenCV convention: +Z out of the lens, +X right,
    +Y down)."""
    K = np.asarray(K, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = pixel_xy
    z = float(depth)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.array([x, y, z], dtype=np.float64)


def camera_point_to_world(T_world_from_camera, point_camera):
    return transform_points(T_world_from_camera, np.asarray(point_camera).reshape(1, 3))[0]


def world_point_to_camera(T_world_from_camera, point_world):
    T_camera_from_world = invert_transform(T_world_from_camera)
    return transform_points(T_camera_from_world, np.asarray(point_world).reshape(1, 3))[0]


def shortest_rotation_aligning(a, b):
    """3x3 rotation matrix R such that R @ (a/|a|) == (b/|b|) -- the
    minimal-angle rotation mapping a onto b (rotation axis a x b, Rodrigues'
    formula). Leaves orientation about the a/b axis itself untouched, so
    applying it to an existing frame's rotation changes that frame as
    little as possible while fixing the one vector -- e.g. correcting a
    frame's "up" to match measured gravity without disturbing its yaw
    (see iphone_extrinsic_calibration.py's --auto-gravity-correct).
    """
    a = np.asarray(a, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3, dtype=np.float64)
        # a and b are opposite (180deg) -- any axis perpendicular to a works.
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        rvec = axis * np.pi
        R, _ = cv2.Rodrigues(rvec)
        return R
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * (1.0 / (1.0 + c))
