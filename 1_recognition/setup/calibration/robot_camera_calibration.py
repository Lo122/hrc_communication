"""Eye-to-hand calibration: where the camera sits in the UR robot's base frame.

The camera is stationary and watches the workspace. A single ArUco marker lies
flat where both the camera and the robot can reach it. Two independent
measurements of that one marker are combined:

  optical   solvePnP on the detected marker      -> T_camera_from_marker
  touch     robot TCP touched to its 4 corners   -> T_base_from_marker

            T_base_from_camera = T_base_from_marker @ inv(T_camera_from_marker)

That is the transform Grasshopper needs to turn a camera-frame human position
into a robot-frame tool-drop location.

The result file is also an extrinsics file for recognition: pass its name as
--extrinsics-file (run_recognition.py / run_system.py). The marker is then the world
-- human positions are reported in its frame -- and T_world_from_robot_base places
the robot in it (camera_utils/calibration_io.py's load_extrinsics). The floor is not
measured here: add "ground_z" (the floor's z in the marker frame, in m) to the file
if the marker is not on the floor.

Robot state is read over RTDE, not ROS: these are a handful of one-shot pose
reads, and rosbridge would add a hop and a running node for nothing.

READ THIS BEFORE RUNNING -- two things this cannot catch for you
---------------------------------------------------------------

1. SET THE TCP TO YOUR TOUCH PIN'S TIP on the pendant. Left at the flange,
   every touch point is displaced by the whole tool vector. If you hold the
   tool orientation roughly constant across the four touches -- which is the
   natural thing to do -- that displacement is a pure translation: it preserves
   every edge length, passes every check below, and silently shifts the entire
   result by the length of your tool. This script prints the orientation spread
   across your touches so the situation is at least visible.

2. A single marker view gives NO independent cross-check. T_base_from_camera is
   *defined* as the composition of the two measurements above, so any residual
   computed between them is zero by construction. The numbers reported here
   (Kabsch residual, reprojection error, edge lengths) check each side's
   internal consistency only. Real validation is running this twice with the
   marker MOVED in between and comparing the two T_base_from_camera results;
   agreement to a few millimetres is the acceptance test. Every run writes its
   own timestamped file so those comparisons stay available.

Procedure
---------

1. Put the marker flat in the shared workspace, fully visible to the camera.
2. Run this script. A preview window opens with the marker's four corners
   labelled C0..C3 and their marker-frame coordinates. Check the labels against
   the physical marker -- this is where a 90 or 180 degree error would come
   from, and seeing which corner is C0 is what prevents it. SPACE accepts.
3. The window closes. Hold the pendant's Freedrive button and touch the pin to
   each labelled corner in turn, pressing ENTER at each. 'r' redoes the last
   one. The script only READS the robot: it never enables Freedrive itself and
   never takes control, so it also works while the robot is in Local mode.
   While it waits, a live line shows the TCP and how far it is from where this
   corner should be relative to the ones already recorded -- aim for zeros. A
   touch that is off (the arm didn't move, a double ENTER, the wrong corner) is
   refused unless you keep it with 'k'.

The optical side degrades fast with distance: the marker's tilt comes from a few
pixels of foreshortening. The script prints how far corner noise moves the solved
camera; at 640x360 a 200 mm marker 5 m away spans ~17 px and moves it by decimetres.
Capture at the camera's full resolution with the matching intrinsics. The touches
stay valid while the marker stays put, so --touches <earlier result> redoes only
the camera side.

Usage:
    uv run python 1_recognition/setup/calibration/robot_camera_calibration.py `
        --intrinsics 1_recognition/calib_data/intrinsics_640x360_obs.json `
        --camera-index 6 --marker-id 0 --marker-length-mm 200 `
        --robot-ip 192.168.1.10

    # camera side only, reusing an earlier run's touches:
    uv run python 1_recognition/setup/calibration/robot_camera_calibration.py `
        --intrinsics 1_recognition/calib_data/intrinsics_3840x2160_obs.json `
        --camera-index 6 --touches 1_recognition/calib_data/robot_camera_calibration_<...>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import msvcrt  # Windows: poll the console, so the live TCP line can refresh while waiting
except ImportError:
    msvcrt = None

if __package__ in (None, ""):
    # src/ for camera_utils, setup/ for the sibling calibration package, repo
    # root for config.ROBOT_IP.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from calibration.extrinsic_calibration import detect_marker_in_frame, make_aruco_detector
from camera_utils import transforms as tf
from camera_utils.calibration_io import load_intrinsics
from logging_setup import configure_logging, get_logger
from video_source import (
    DEFAULT_PREVIEW_WIDTH, forget_preview_windows, open_camera, preview_scale,
    show_preview)

logger = get_logger(__name__)

# The user's board: a 4x4 marker, id 0, 200 mm across, centre at (0, 0, 0).
# DICT_4X4_50 rather than extrinsic_calibration.py's DICT_5X5_50 default -- a
# 4x4 dictionary carries fewer bits, which at 200 mm is still far more than
# enough to be unambiguous and leaves larger, more detectable cells.
DEFAULT_ARUCO_DICT = "DICT_4X4_50"
DEFAULT_MARKER_ID = 0
DEFAULT_MARKER_LENGTH_MM = 200.0

CORNER_LABELS = ("C0", "C1", "C2", "C3")

# A touched corner this far off its expected distance from another corner is
# called out before the fit runs. 5 mm is well outside a careful touch with a
# pin and well inside the ~100 mm error a swapped corner would produce.
TOUCH_DISTANCE_TOLERANCE_M = 0.005

# Orientation spread across the four touches, above which the caveat about a
# wrong TCP offset becomes detectable rather than invisible (see module docstring).
ORIENTATION_SPREAD_WARN_DEG = 10.0

# Typical sub-pixel ArUco corner noise, for the optical uncertainty estimate. It
# matches the ~0.25 px reprojection error these solves report.
CORNER_NOISE_PX = 0.25

# Camera-position uncertainty (either side) above which the result is flagged.
CAMERA_UNCERTAINTY_WARN_M = 0.010

# The live TCP line is cut to this width so it never wraps: a wrapped line
# can't be redrawn in place with '\r'.
STATUS_WIDTH = 79


def marker_corner_points(marker_length_m):
    """The marker's 4 corners in its own frame, (4, 3), metres.

    Order is TL, TR, BR, BL -- the order cv2.aruco reports detected corners in,
    and the order solve_marker_pose builds its solvePnP object points in
    (extrinsic_calibration.py). The optical and touch sides of this calibration
    MUST agree on it, so both derive from here.

    Marker frame: +X right, +Y up across the marker face, +Z out of the face.
    Laid flat and facing up, +Z points at the ceiling.
    """
    half = float(marker_length_m) / 2.0
    return np.array([
        [-half, half, 0.0],   # C0 top-left
        [half, half, 0.0],    # C1 top-right
        [half, -half, 0.0],   # C2 bottom-right
        [-half, -half, 0.0],  # C3 bottom-left
    ], dtype=np.float64)


def solve_marker_pose_with_error(corners_2d, marker_length_m, K, dist):
    """T_camera_from_marker plus the mean reprojection error in pixels.

    Same solve as extrinsic_calibration.solve_marker_pose -- IPPE_SQUARE, which
    is built for exactly this four-coplanar-corner case -- with the residual
    kept, since it is the only quality number the optical side produces.
    """
    object_points = marker_corner_points(marker_length_m)
    image_points = np.asarray(corners_2d, dtype=np.float64).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, float("nan")

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    error_px = float(np.mean(np.linalg.norm(projected.reshape(4, 2) - image_points, axis=1)))
    return tf.rvec_tvec_to_transform(rvec, tvec), error_px


def estimate_optical_uncertainty(T_camera_from_marker, marker_length_m, K, dist,
                                 noise_px=CORNER_NOISE_PX, trials=300, seed=0):
    """How far corner noise moves the SOLVED CAMERA POSITION -- the optical
    counterpart of estimate_camera_uncertainty. Returns (marker_side_px,
    median_m, p90_m).

    Monte Carlo: project the marker through the accepted pose, jitter the corners
    by noise_px, re-solve, and measure where the camera lands. A small marker is
    the failure case: its tilt comes from a pixel or two of foreshortening, and
    IPPE's mirrored solution fits almost as well, so the solve can flip. That
    shows up as a p90 far above the median.
    """
    object_points = marker_corner_points(marker_length_m)
    rvec, _ = cv2.Rodrigues(np.asarray(T_camera_from_marker, dtype=np.float64)[:3, :3])
    tvec = np.asarray(T_camera_from_marker, dtype=np.float64)[:3, 3]
    pixels, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    pixels = pixels.reshape(4, 2)
    side_px = float(np.mean(np.linalg.norm(pixels - np.roll(pixels, -1, axis=0), axis=1)))

    camera_in_marker = tf.invert_transform(T_camera_from_marker)[:3, 3]
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(trials):
        noisy = pixels + rng.normal(0.0, noise_px, pixels.shape)
        ok, rvec_i, tvec_i = cv2.solvePnP(
            object_points, noisy, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            solved = tf.invert_transform(tf.rvec_tvec_to_transform(rvec_i, tvec_i))[:3, 3]
            errors.append(np.linalg.norm(solved - camera_in_marker))
    if not errors:
        return side_px, float("inf"), float("inf")
    return side_px, float(np.median(errors)), float(np.percentile(errors, 90))


CORNER_COLOUR = (0, 128, 255)


def draw_corner_label(display, pixel, center, s, lines):
    """Draw a corner's text lines, (text, font scale) each, just outside the marker in
    that corner's direction -- so the four labels of a small marker don't pile up on
    top of each other and the marker itself stays visible. s: preview_scale."""
    direction = np.asarray(pixel, dtype=np.float64) - center
    direction /= max(np.linalg.norm(direction), 1e-6)
    anchor = np.asarray(pixel, dtype=np.float64) + direction * 8 * s
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(int(1 * s), 1)
    sizes = [cv2.getTextSize(text, font, scale * s, thickness)[0] for text, scale in lines]
    gap = int(4 * s)
    block_h = sum(h for _w, h in sizes) + gap * (len(lines) - 1)
    # Above the marker, the block ends at the anchor; below it, it starts there.
    y = anchor[1] - block_h if direction[1] < 0 else anchor[1]
    for (text, scale), (w, h) in zip(lines, sizes):
        y += h
        x = anchor[0] if direction[0] >= 0 else anchor[0] - w  # left corners: right-aligned
        cv2.putText(display, text, (int(x), int(y)), font, scale * s, CORNER_COLOUR, thickness,
                    cv2.LINE_AA)
        y += gap


def capture_marker_corners_live(camera_index, detector, marker_id, marker_length_m,
                                 K, dist, capture_size=None,
                                 preview_width=DEFAULT_PREVIEW_WIDTH):
    """Preview the marker with its corners labelled, and return the accepted
    (corners_2d, T_camera_from_marker, reprojection_error_px).

    The labelling is the point: the operator has to touch the same physical
    corners, in the same order, that solvePnP used. Showing which corner is C0
    on the live image is what stops a 90 or 180 degree mistake.
    """
    width, height = capture_size if capture_size else (None, None)
    # open_camera requests the mode AND verifies it against a decoded frame --
    # solvePnP here uses the same K as every other solve, so capturing at a
    # resolution it was not solved for would put T_camera_from_marker, and
    # therefore T_base_from_camera, silently out by the size ratio.
    capture, actual = open_camera(camera_index, width, height)
    # Overlays are drawn full-size but shown downscaled; scale them to match.
    s = preview_scale(actual[0], preview_width)

    marker_points = marker_corner_points(marker_length_m)
    accepted = (None, None, float("nan"))
    print("Check the corner labels against the physical marker, then press SPACE. "
          "ESC or Q aborts.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            corners_2d = detect_marker_in_frame(detector, frame, marker_id)
            display = frame.copy()

            if corners_2d is not None:
                cv2.polylines(display, [corners_2d.astype(np.int32)], True, (0, 255, 0),
                              max(int(1 * s), 1))
                center = corners_2d.mean(axis=0)
                for label, pixel, marker_xyz in zip(CORNER_LABELS, corners_2d, marker_points):
                    cv2.circle(display, tuple(pixel.astype(int)), max(int(3 * s), 1), CORNER_COLOUR, -1)
                    draw_corner_label(display, pixel, center, s, [
                        (label, 0.45),
                        (f"({marker_xyz[0]:+.3f}, {marker_xyz[1]:+.3f})", 0.32),
                    ])
                cv2.putText(display, "SPACE to accept this corner order",
                            (int(10 * s), int(24 * s)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (0, 255, 0),
                            max(int(1 * s), 1))
            else:
                cv2.putText(display, f"marker id={marker_id} not found",
                            (int(10 * s), int(24 * s)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (0, 0, 255),
                            max(int(1 * s), 1))

            show_preview("robot-camera calibration - mark the corner order", display,
                         preview_width)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and corners_2d is not None:
                T_camera_from_marker, error_px = solve_marker_pose_with_error(
                    corners_2d, marker_length_m, K, dist)
                if T_camera_from_marker is not None:
                    accepted = (corners_2d, T_camera_from_marker, error_px)
                    break
                logger.warning("solvePnP failed on that detection; keep going.")
            elif key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        forget_preview_windows()

    return accepted


def connect_robot(robot_ip):
    """Read-only RTDE connection. Imported here rather than at module level so
    this file stays importable (and its math testable) without ur-rtde or a
    robot on the network."""
    from rtde_receive import RTDEReceiveInterface

    logger.info("Connecting to UR at %s (read-only)...", robot_ip)
    return RTDEReceiveInterface(robot_ip)


def wait_for_key(status=None, period_s=0.1):
    """One key press, lower-cased, with ENTER as "". status() gives the text of a
    live line, redrawn until the key comes.

    Keys pressed earlier are dropped first, so a double ENTER, or ENTERs typed
    while RTDE was connecting, can't record a corner on their own. Where the
    console can't be polled (not Windows) this falls back to input(), with the
    status shown once.
    """
    if msvcrt is None:
        if status is not None:
            print(status())
        return input("    > ").strip().lower()[:1]

    while msvcrt.kbhit():
        msvcrt.getwch()
    while True:
        if status is not None:
            print("\r" + status().ljust(STATUS_WIDTH)[:STATUS_WIDTH], end="", flush=True)
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key in ("\x00", "\xe0"):  # arrows and F-keys: a prefix, then a code
                msvcrt.getwch()
                continue
            if key == "\x03":
                raise KeyboardInterrupt
            if status is not None:
                print()
            return "" if key in ("\r", "\n") else key.lower()
        time.sleep(period_s)


def read_fresh_pose(rtde_r, wait_s=0.05):
    """The current TCP pose, or None if RTDE has stopped updating. A frozen
    connection keeps returning its last pose, which would record every later
    corner at the same point -- the controller timestamp is what tells."""
    stamp = rtde_r.getTimestamp()
    time.sleep(wait_s)
    if rtde_r.getTimestamp() == stamp:
        return None
    return list(rtde_r.getActualTCPPose())


def touch_distance_errors(point, recorded_poses, marker_points):
    """For the corner after recorded_poses: measured minus expected distance
    from each recorded corner, in metres, in recording order."""
    target = marker_points[len(recorded_poses)]
    return [
        float(np.linalg.norm(np.asarray(point[:3]) - np.asarray(pose[:3]))
              - np.linalg.norm(target - marker_points[j]))
        for j, pose in enumerate(recorded_poses)
    ]


def collect_touch_points(rtde_r, marker_points, labels=CORNER_LABELS, read_key=wait_for_key,
                         tolerance_m=TOUCH_DISTANCE_TOLERANCE_M):
    """Prompt for one TCP touch per marker corner. Returns the full (N, 6) TCP
    poses -- position and axis-angle orientation, as getActualTCPPose gives
    them -- or None if aborted. Orientation is kept for the spread check, not
    for the fit.

    Each touch is checked against the corners already recorded as it is taken,
    not only at the end: a touch that repeats the previous one or lands on the
    wrong corner would otherwise shift every later corner by one.
    """
    poses = []
    last_stamp = None

    def status():
        """The live line: the TCP, and how far it is off each recorded corner."""
        nonlocal last_stamp
        stamp = rtde_r.getTimestamp()
        if stamp == last_stamp:
            return "    (robot data not updating)"
        last_stamp = stamp
        pose = rtde_r.getActualTCPPose()
        line = f"    TCP ({pose[0]:+.4f}, {pose[1]:+.4f}, {pose[2]:+.4f}) m"
        errors = touch_distance_errors(pose, poses, marker_points)
        if errors:
            line += " | off " + " ".join(
                f"{labels[j]} {e * 1000:+.1f}" for j, e in enumerate(errors)) + " mm"
        return line

    index = 0
    while index < len(marker_points):
        label = labels[index]
        marker_xyz = marker_points[index]
        print(f"  Touch {label} at marker ({marker_xyz[0]:+.3f}, {marker_xyz[1]:+.3f}, "
              f"{marker_xyz[2]:+.3f}) -- ENTER records, 'r' redoes the previous, 'q' aborts")
        answer = read_key(status)
        if answer == "q":
            return None
        if answer == "r":
            if index == 0:
                print("    Nothing recorded yet.")
            else:
                index -= 1
                poses.pop()
                print(f"    Dropped {labels[index]}; touch it again.")
            continue
        if answer != "":
            continue

        pose = read_fresh_pose(rtde_r)
        if pose is None:
            print("    Robot data is not updating, so nothing was recorded. Reconnecting...")
            rtde_r.reconnect()
            continue
        errors = touch_distance_errors(pose, poses, marker_points)
        off = [(labels[j], e) for j, e in enumerate(errors) if abs(e) > tolerance_m]
        if off:
            print(f"    {label} at ({pose[0]:+.4f}, {pose[1]:+.4f}, {pose[2]:+.4f}) m is off: "
                  + ", ".join(f"{other} {e * 1000:+.1f} mm" for other, e in off)
                  + " -- the arm didn't move, or the wrong corner?")
            print("    'k' keeps it anyway, any other key touches it again.")
            if read_key() != "k":
                continue
        poses.append(pose)
        print(f"    {label} = ({pose[0]:+.4f}, {pose[1]:+.4f}, {pose[2]:+.4f}) m")
        index += 1

    return np.array(poses, dtype=np.float64)


def load_touch_poses(path):
    """The (4, 6) TCP touch poses saved in an earlier result file."""
    with open(path) as f:
        poses = np.asarray(json.load(f)["touch_poses_tcp"], dtype=np.float64)
    if poses.shape != (len(CORNER_LABELS), 6):
        raise ValueError(f"{path}: expected {len(CORNER_LABELS)} touch poses of 6 values, "
                         f"got shape {poses.shape}")
    return poses


def orientation_spread_deg(poses):
    """Largest angle between any two touches' tool orientations, in degrees.

    Near zero means a wrong TCP offset would act as a pure translation on every
    touch -- undetectable here, and a silent shift of the whole calibration
    (see the module docstring).
    """
    rotations = [cv2.Rodrigues(np.asarray(pose[3:], dtype=np.float64))[0] for pose in poses]
    worst = 0.0
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            relative = rotations[i].T @ rotations[j]
            angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
            worst = max(worst, float(angle))
    return worst


def check_touch_geometry(points_base, marker_points, tolerance_m=TOUCH_DISTANCE_TOLERANCE_M):
    """Compare every pairwise distance between touched points against the same
    distance on the marker. Returns a list of (label_i, label_j, error_m) for
    the pairs that are out by more than tolerance_m.

    Runs before the fit so a bad touch is named and can be redone, rather than
    being averaged into the result. A swapped pair of corners shows up here as
    an error of order the marker size.
    """
    problems = []
    for i in range(len(points_base)):
        for j in range(i + 1, len(points_base)):
            measured = float(np.linalg.norm(points_base[i] - points_base[j]))
            expected = float(np.linalg.norm(marker_points[i] - marker_points[j]))
            error = measured - expected
            if abs(error) > tolerance_m:
                problems.append((CORNER_LABELS[i], CORNER_LABELS[j], error))
    return problems


def solve_robot_camera(touch_poses, T_camera_from_marker, marker_length_m):
    """Compose the two measurements. Returns a dict of every transform and
    residual worth keeping."""
    marker_points = marker_corner_points(marker_length_m)
    points_base = np.asarray(touch_poses, dtype=np.float64)[:, :3]

    T_base_from_marker, touch_rms_m = tf.rigid_transform_from_points(
        marker_points, points_base)
    T_base_from_camera = tf.compose_transforms(
        T_base_from_marker, tf.invert_transform(T_camera_from_marker))

    return {
        "T_base_from_camera": T_base_from_camera,
        "T_base_from_marker": T_base_from_marker,
        "T_camera_from_marker": T_camera_from_marker,
        # The repo-native form: extrinsics.json stores the robot base in the
        # world frame, and the marker is the world origin by this repo's
        # convention (see extrinsic_calibration.py). Kept so merging into
        # extrinsics.json later is a copy, not a re-derivation.
        "T_world_from_robot_base": tf.invert_transform(T_base_from_marker),
        "touch_rms_m": touch_rms_m,
    }


def estimate_camera_uncertainty(touch_rms_m, T_base_from_camera, T_base_from_marker,
                                 marker_length_m):
    """Rough uncertainty in the SOLVED CAMERA POSITION implied by the touch
    residual. Returns (lever_ratio, position_uncertainty_m).

    The touch points only span the marker, but the answer is a pose for
    something metres away, so angular error in T_base_from_marker is amplified
    by the lever arm between them:

        theta ~= touch_rms / r         r = corner radius = marker_length / sqrt(2)
        uncertainty ~= theta * d       d = marker-to-camera distance

    i.e. the error is multiplied by roughly sqrt(2) * d / marker_length. A
    200 mm marker read by a camera 1.8 m away amplifies about 13x, so a 0.5 mm
    touch residual is around 6 mm at the camera. This is why marker size
    matters more than touch precision: doubling the marker halves the result,
    while touching twice as carefully is much harder.

    An estimate, not a bound -- it assumes the residual is isotropic and
    ignores the reprojection error, which adds its own contribution.
    """
    distance_m = float(np.linalg.norm(T_base_from_camera[:3, 3] - T_base_from_marker[:3, 3]))
    corner_radius_m = float(marker_length_m) / np.sqrt(2.0)
    lever_ratio = distance_m / corner_radius_m
    return lever_ratio, float(touch_rms_m) * lever_ratio


def format_grasshopper_block(T_base_from_camera):
    """The camera frame as a Rhino Plane -- origin plus X and Y axes, which is
    what Grasshopper's Plane component takes."""
    origin = T_base_from_camera[:3, 3]
    x_axis = T_base_from_camera[:3, 0]
    y_axis = T_base_from_camera[:3, 1]
    z_axis = T_base_from_camera[:3, 2]

    lines = [
        "--- Grasshopper: camera frame in robot base coordinates ---",
        "Plane(origin, x_axis, y_axis); units are metres, same as UR.",
        "",
        f"origin  {origin[0]:+.6f}, {origin[1]:+.6f}, {origin[2]:+.6f}",
        f"x_axis  {x_axis[0]:+.6f}, {x_axis[1]:+.6f}, {x_axis[2]:+.6f}",
        f"y_axis  {y_axis[0]:+.6f}, {y_axis[1]:+.6f}, {y_axis[2]:+.6f}",
        f"z_axis  {z_axis[0]:+.6f}, {z_axis[1]:+.6f}, {z_axis[2]:+.6f}",
        "",
        "T_base_from_camera (row-major 4x4):",
    ]
    for row in T_base_from_camera:
        lines.append("  " + ", ".join(f"{value:+.6f}" for value in row))
    lines.append("-" * 58)
    return "\n".join(lines)


def save_result(path, result, touch_poses, marker_id, marker_length_m, aruco_dict,
                 reprojection_error_px, notes="", extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        key: np.asarray(value, dtype=np.float64).tolist()
        for key, value in result.items() if key.startswith("T_")
    }
    data.update({
        "touch_poses_tcp": np.asarray(touch_poses, dtype=np.float64).tolist(),
        "touch_labels": list(CORNER_LABELS),
        "marker_corners_m": marker_corner_points(marker_length_m).tolist(),
        "marker_id": marker_id,
        "marker_length_m": float(marker_length_m),
        "aruco_dict": aruco_dict,
        "touch_rms_m": float(result["touch_rms_m"]),
        "reprojection_error_px": float(reprojection_error_px),
        "orientation_spread_deg": float(orientation_spread_deg(touch_poses)),
        "camera_position_uncertainty_m": estimate_camera_uncertainty(
            result["touch_rms_m"], result["T_base_from_camera"],
            result["T_base_from_marker"], marker_length_m)[1],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
    })
    data.update(extra or {})
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def run_robot_camera_calibration(intrinsics_path, camera_index, marker_id, marker_length_mm,
                                  aruco_dict, robot_ip, output, capture_size=None,
                                  preview_width=DEFAULT_PREVIEW_WIDTH, touches_path=None):
    """Full procedure. Returns the result dict, or None if it was aborted.
    touches_path: reuse the touch poses of an earlier result file instead of
    touching again -- valid only while the marker hasn't moved since."""
    K, dist, image_size = load_intrinsics(intrinsics_path)
    marker_length_m = marker_length_mm / 1000.0
    detector = make_aruco_detector(aruco_dict)
    reused_touches = load_touch_poses(touches_path) if touches_path else None

    if reused_touches is None:
        logger.warning(
            "Before touching: the TCP on the pendant must be set to your touch pin's TIP. "
            "Left at the flange, every point is offset by the tool vector, and with a "
            "constant tool orientation that offset passes every check here.")

    capture_size = capture_size or image_size
    corners_2d, T_camera_from_marker, reprojection_error_px = capture_marker_corners_live(
        camera_index, detector, marker_id, marker_length_m, K, dist, capture_size,
        preview_width=preview_width)
    if T_camera_from_marker is None:
        logger.error("No marker pose accepted. Aborting.")
        return None
    logger.info("T_camera_from_marker solved, mean reprojection error %.2f px",
                reprojection_error_px)

    side_px, optical_m, optical_p90_m = estimate_optical_uncertainty(
        T_camera_from_marker, marker_length_m, K, dist)
    logger.info("Optical side: the marker spans %.1f px; %.2f px of corner noise moves the "
                "solved camera by ~%.0f mm (median), %.0f mm in the worst 10%%.",
                side_px, CORNER_NOISE_PX, optical_m * 1000.0, optical_p90_m * 1000.0)
    if optical_m > CAMERA_UNCERTAINTY_WARN_M:
        logger.warning(
            "That is over %.0f mm. Capture at a higher resolution (the intrinsics for it set "
            "the capture size), bring the marker closer to the camera, or use a bigger one.%s",
            CAMERA_UNCERTAINTY_WARN_M * 1000.0,
            "" if reused_touches is not None else
            " The touches stay valid while the marker stays put: --touches <the file this "
            "run saves> redoes only the camera side.")

    if reused_touches is not None:
        logger.info("Reusing the touches in %s -- valid only if the marker hasn't moved since.",
                    touches_path)
        touch_poses = reused_touches
    else:
        rtde_r = connect_robot(robot_ip)
        try:
            print("\nHold the pendant's Freedrive button and touch each corner in turn.")
            touch_poses = collect_touch_points(rtde_r, marker_corner_points(marker_length_m))
        finally:
            rtde_r.disconnect()

    if touch_poses is None:
        logger.error("Touch collection aborted.")
        return None

    problems = check_touch_geometry(touch_poses[:, :3], marker_corner_points(marker_length_m))
    for label_i, label_j, error in problems:
        logger.warning("%s-%s distance is out by %+.1f mm -- a mis-touch, a wrong corner "
                       "order, or a marker that is not the size given.",
                       label_i, label_j, error * 1000.0)

    result = solve_robot_camera(touch_poses, T_camera_from_marker, marker_length_m)

    spread = orientation_spread_deg(touch_poses)
    logger.info("Touch fit RMS residual: %.2f mm", result["touch_rms_m"] * 1000.0)

    lever_ratio, uncertainty_m = estimate_camera_uncertainty(
        result["touch_rms_m"], result["T_base_from_camera"], result["T_base_from_marker"],
        marker_length_m)
    logger.info("Lever arm: the camera is %.1fx further away than the marker's corner "
                "radius, so that %.2f mm residual is roughly %.1f mm of uncertainty in "
                "the camera position.",
                lever_ratio, result["touch_rms_m"] * 1000.0, uncertainty_m * 1000.0)
    if uncertainty_m > CAMERA_UNCERTAINTY_WARN_M:
        logger.warning(
            "That is over %.0f mm. A bigger marker is the effective fix -- the error scales "
            "as 1/marker_length, so doubling the marker halves it, while touching twice as "
            "carefully is far harder.", CAMERA_UNCERTAINTY_WARN_M * 1000.0)

    logger.info("Tool orientation spread across touches: %.1f deg", spread)
    if spread < ORIENTATION_SPREAD_WARN_DEG:
        logger.warning(
            "Tool orientation barely changed (%.1f deg < %.1f). A wrong TCP offset would "
            "act as a pure translation here and is therefore invisible to every check "
            "above -- confirm the TCP is your pin tip before trusting this result.",
            spread, ORIENTATION_SPREAD_WARN_DEG)

    saved = save_result(
        output, result, touch_poses, marker_id, marker_length_m, aruco_dict,
        reprojection_error_px,
        notes=f"touches reused from {Path(touches_path).name}" if touches_path else "",
        extra={
            "capture_size": [int(v) for v in capture_size],
            "corners_px": np.asarray(corners_2d, dtype=np.float64).reshape(4, 2).tolist(),
            "marker_side_px": side_px,
            "optical_position_uncertainty_m": optical_m,
            "optical_position_uncertainty_p90_m": optical_p90_m,
        })
    logger.info("Saved to %s", saved)
    print()
    print(format_grasshopper_block(result["T_base_from_camera"]))
    print("\nRun this again with the marker MOVED and compare the two files -- "
          "that is the only real check on this number.")
    return result


def default_output_path():
    """Timestamped, so a second run never silently overwrites the first. The
    two-marker-position comparison in the module docstring depends on both
    results surviving."""
    calib_dir = Path(__file__).resolve().parents[2] / "calib_data"
    return calib_dir / f"robot_camera_calibration_{time.strftime('%Y%m%d_%H%M%S')}.json"


def main(argv=None):
    configure_logging("robot_camera_calibration")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--intrinsics", required=True,
                        help="Intrinsics JSON for the camera, from setup/calibrate_camera.py. "
                             "Must have been solved at the resolution this will capture at.")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Run setup/video_source.py --list to find it.")
    parser.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID)
    parser.add_argument("--marker-length-mm", type=float, default=DEFAULT_MARKER_LENGTH_MM,
                        help="Side length of the printed marker. Measure it rather than "
                             "trusting the print dialogue -- it scales the result directly.")
    parser.add_argument("--aruco-dict", default=DEFAULT_ARUCO_DICT)
    parser.add_argument("--robot-ip", default=None,
                        help="UR controller IP. Defaults to config.ROBOT_IP.")
    parser.add_argument("--preview-width", type=int, default=DEFAULT_PREVIEW_WIDTH,
                        help=f"Width of the corner-labelling preview (default "
                             f"{DEFAULT_PREVIEW_WIDTH}). Display only -- solvePnP always uses "
                             "the full-resolution frame.")
    parser.add_argument("--output", default=None,
                        help="Where to write the result. Defaults to a timestamped file in "
                             "1_recognition/calib_data/.")
    parser.add_argument("--touches", default=None,
                        help="Reuse the touch poses of an earlier result file and redo only "
                             "the camera side (no robot needed). Valid only while the marker "
                             "hasn't moved since those touches.")
    args = parser.parse_args(argv)

    robot_ip = args.robot_ip
    if robot_ip is None:
        import config
        robot_ip = config.ROBOT_IP

    output = args.output or default_output_path()
    try:
        result = run_robot_camera_calibration(
            args.intrinsics, args.camera_index, args.marker_id, args.marker_length_mm,
            args.aruco_dict, robot_ip, output, preview_width=args.preview_width,
            touches_path=args.touches)
    except (IOError, OSError, RuntimeError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 1
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
