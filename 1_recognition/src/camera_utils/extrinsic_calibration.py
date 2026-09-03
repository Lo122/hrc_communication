"""Extrinsic calibration: solvePnP against a target of known world pose ->
T_world_from_camera. Two target methods, selected with --method:

- ``board`` (default): a ChArUco board (generate_charuco_board.py) -- many
  correspondence points spread over a larger physical area, usually a more
  accurate/stable solvePnP, and tolerant of partial occlusion/cropping
  (see charuco_board.py's docstring).
- ``marker``: a single plain ArUco marker (generate_calibration_targets.py)
  -- quicker to print/set up (one small square instead of a full board),
  useful when the board doesn't fit your calibration space or a full board
  is overkill.

Physical setup this expects (either method):

- Print the target (board or marker) at a known size and place it at a
  known, measured pose relative to your world origin -- typically flat on
  the ground/table near the robot base, facing up, so the target's local
  +Z axis points straight up and its plane defines the ``ground_z = 0``
  plane used later for ankle back-projection.
- By default the target origin *is* the world origin (identity
  T_world_from_board) and the robot base is also registered at the world
  origin. Pass --board-xyz/--board-rpy-deg and/or --robot-base-xyz/
  --robot-base-rpy-deg if the target or the robot base sit somewhere else in
  the world frame (units: meters, degrees, roll-pitch-yaw = rotation about
  X, then Y, then Z of the world frame).

Usage:
    python -m camera_utils.extrinsic_calibration `
        --intrinsics ../dataset/calib_data/intrinsics.json `
        --camera-index 0 --squares-x 7 --squares-y 9 `
        --square-length-mm 25 --marker-length-mm 19 `
        --output ../dataset/calib_data/extrinsics.json

    python -m camera_utils.extrinsic_calibration `
        --method marker --intrinsics ../dataset/calib_data/intrinsics.json `
        --camera-index 0 --marker-id 0 --marker-length-mm 100 `
        --output ../dataset/calib_data/extrinsics.json

Press SPACE to use the current frame's target detection, Q/ESC to abort. Or
pass --image path/to/frame.png to calibrate from a single still image
instead of opening a live camera.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_utils.calibration_io import load_intrinsics, save_extrinsics
from camera_utils.charuco_board import detect_charuco, draw_charuco_detection, make_charuco_board
from camera_utils import transforms as tf


def solve_board_pose(charuco_corners, charuco_ids, board, K, dist, min_corners=6):
    """Returns T_camera_from_board, or None if too few corners to solve reliably."""
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if object_points is None or len(object_points) < min_corners:
        return None
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist)
    if not ok:
        return None
    return tf.rvec_tvec_to_transform(rvec, tvec)


def capture_board_pose_live(camera_index, detector, board, K, dist, min_corners):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Could not open camera index {camera_index}")

    print("Live extrinsic calibration: press SPACE to accept the current board "
          "detection, ESC/Q to abort.")
    T_camera_from_board = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids = detect_charuco(detector, gray, min_corners=min_corners)
            display = frame.copy()
            draw_charuco_detection(display, charuco_corners, charuco_ids)
            if charuco_corners is not None:
                n = len(charuco_ids)
                cv2.putText(display, f"board found ({n} corners) - SPACE to accept", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display, "board not found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("extrinsic calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and charuco_corners is not None:
                T_camera_from_board = solve_board_pose(
                    charuco_corners, charuco_ids, board, K, dist, min_corners=min_corners)
                if T_camera_from_board is not None:
                    break
            elif key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return T_camera_from_board


def make_aruco_detector(dictionary_name):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_marker_in_frame(detector, frame, marker_id):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    ids = ids.ravel()
    for c, i in zip(corners, ids):
        if marker_id is None or i == marker_id:
            return c.reshape(4, 2)
    return None


def solve_marker_pose(corners_2d, marker_length_m, K, dist):
    """corners_2d: (4, 2) array, order TL, TR, BR, BL (as returned by aruco). Returns T_camera_from_marker."""
    half = marker_length_m / 2.0
    object_points = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, np.asarray(corners_2d, dtype=np.float64), K, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return tf.rvec_tvec_to_transform(rvec, tvec)


def capture_marker_pose_live(camera_index, detector, marker_id, marker_length_m, K, dist):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Could not open camera index {camera_index}")

    print("Live extrinsic calibration: press SPACE to accept the current marker "
          "detection, ESC/Q to abort.")
    T_camera_from_marker = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            corners_2d = detect_marker_in_frame(detector, frame, marker_id)
            display = frame.copy()
            if corners_2d is not None:
                cv2.polylines(display, [corners_2d.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(display, "marker found - SPACE to accept", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display, "marker not found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("extrinsic calibration - aruco marker", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and corners_2d is not None:
                T_camera_from_marker = solve_marker_pose(corners_2d, marker_length_m, K, dist)
                if T_camera_from_marker is not None:
                    break
            elif key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return T_camera_from_marker


def run_extrinsic_calibration(intrinsics_path, image, camera_index, squares_x, squares_y,
                               square_length_mm, marker_length_mm, aruco_dict, min_corners,
                               board_xyz, board_rpy_deg, robot_base_xyz, robot_base_rpy_deg,
                               ground_z, output):
    """Shared entry point used by both this module's CLI and the
    calibrate_camera app. Returns T_world_from_camera, or None on failure."""
    K, dist, _ = load_intrinsics(intrinsics_path)
    board, detector = make_charuco_board(
        squares_x, squares_y,
        square_length_mm / 1000.0, marker_length_mm / 1000.0,
        aruco_dict)

    if image:
        frame = cv2.imread(image)
        if frame is None:
            print(f"Could not read image {image}")
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids = detect_charuco(detector, gray, min_corners=min_corners)
        if charuco_corners is None:
            print("No ChArUco board found in the given image.")
            return None
        T_camera_from_board = solve_board_pose(
            charuco_corners, charuco_ids, board, K, dist, min_corners=min_corners)
    else:
        T_camera_from_board = capture_board_pose_live(
            camera_index, detector, board, K, dist, min_corners)

    if T_camera_from_board is None:
        print("Failed to solve board pose. Aborting.")
        return None

    T_world_from_board = tf.make_transform(
        tf.rpy_deg_to_matrix(*board_rpy_deg), board_xyz)
    T_world_from_camera = tf.compose_transforms(
        T_world_from_board, tf.invert_transform(T_camera_from_board))

    T_world_from_robot_base = tf.make_transform(
        tf.rpy_deg_to_matrix(*robot_base_rpy_deg), robot_base_xyz)

    print(f"T_world_from_camera =\n{T_world_from_camera}")

    save_extrinsics(
        output,
        T_world_from_camera,
        ground_z=ground_z,
        T_world_from_robot_base=T_world_from_robot_base,
        marker_id=None,
        notes=(f"charuco: squares={squares_x}x{squares_y}, "
               f"square_length_mm={square_length_mm}, "
               f"marker_length_mm={marker_length_mm}, aruco_dict={aruco_dict}"),
    )
    print(f"Saved extrinsics to {output}")
    return T_world_from_camera


def run_extrinsic_calibration_marker(intrinsics_path, image, camera_index, marker_id,
                                      marker_length_mm, aruco_dict, board_xyz, board_rpy_deg,
                                      robot_base_xyz, robot_base_rpy_deg, ground_z, output):
    """Marker variant of run_extrinsic_calibration -- solves against a single
    plain ArUco marker (generate_calibration_targets.py) instead of a
    ChArUco board. Returns T_world_from_camera, or None on failure."""
    K, dist, _ = load_intrinsics(intrinsics_path)
    detector = make_aruco_detector(aruco_dict)
    marker_length_m = marker_length_mm / 1000.0

    if image:
        frame = cv2.imread(image)
        if frame is None:
            print(f"Could not read image {image}")
            return None
        corners_2d = detect_marker_in_frame(detector, frame, marker_id)
        if corners_2d is None:
            print("No marker found in the given image.")
            return None
        T_camera_from_marker = solve_marker_pose(corners_2d, marker_length_m, K, dist)
    else:
        T_camera_from_marker = capture_marker_pose_live(
            camera_index, detector, marker_id, marker_length_m, K, dist)

    if T_camera_from_marker is None:
        print("Failed to solve marker pose. Aborting.")
        return None

    T_world_from_marker = tf.make_transform(
        tf.rpy_deg_to_matrix(*board_rpy_deg), board_xyz)
    T_world_from_camera = tf.compose_transforms(
        T_world_from_marker, tf.invert_transform(T_camera_from_marker))

    T_world_from_robot_base = tf.make_transform(
        tf.rpy_deg_to_matrix(*robot_base_rpy_deg), robot_base_xyz)

    print(f"T_world_from_camera =\n{T_world_from_camera}")

    save_extrinsics(
        output,
        T_world_from_camera,
        ground_z=ground_z,
        T_world_from_robot_base=T_world_from_robot_base,
        marker_id=marker_id,
        notes=f"aruco_dict={aruco_dict}, marker_length_mm={marker_length_mm}",
    )
    print(f"Saved extrinsics to {output}")
    return T_world_from_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", type=str, choices=("board", "marker"), default="board",
                         help="'board': ChArUco board (default). 'marker': single plain ArUco "
                              "marker -- see module docstring.")
    parser.add_argument("--intrinsics", type=str, required=True)
    parser.add_argument("--image", type=str, default=None,
                         help="Single still image to calibrate from instead of live capture.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--squares-x", type=int, default=None,
                         help="Required for --method board.")
    parser.add_argument("--squares-y", type=int, default=None,
                         help="Required for --method board.")
    parser.add_argument("--square-length-mm", type=float, default=None,
                         help="Required for --method board.")
    parser.add_argument("--marker-length-mm", type=float, default=None,
                         help="Required for both methods (side length of the embedded ArUco "
                              "marker for --method board, or of the whole marker for --method marker).")
    parser.add_argument("--marker-id", type=int, default=None,
                         help="--method marker only. Expected marker id. If omitted, uses the "
                              "first marker detected.")
    parser.add_argument("--aruco-dict", type=str, default="DICT_5X5_50")
    parser.add_argument("--min-corners", type=int, default=6,
                         help="--method board only.")
    parser.add_argument("--board-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                         help="Board/marker origin in world coords, meters. Default: world origin.")
    parser.add_argument("--board-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                         help="Board/marker orientation in world coords, roll pitch yaw degrees.")
    parser.add_argument("--robot-base-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                         help="Robot base origin in world coords, meters.")
    parser.add_argument("--robot-base-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--ground-z", type=float, default=0.0,
                         help="World Z of the ground plane used later for ankle back-projection.")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.marker_length_mm is None:
        parser.error("--marker-length-mm is required.")

    if args.method == "board":
        if args.squares_x is None or args.squares_y is None or args.square_length_mm is None:
            parser.error("--squares-x, --squares-y and --square-length-mm are required "
                         "for --method board.")
        run_extrinsic_calibration(
            args.intrinsics, args.image, args.camera_index, args.squares_x, args.squares_y,
            args.square_length_mm, args.marker_length_mm, args.aruco_dict, args.min_corners,
            args.board_xyz, args.board_rpy_deg, args.robot_base_xyz, args.robot_base_rpy_deg,
            args.ground_z, args.output)
    else:
        run_extrinsic_calibration_marker(
            args.intrinsics, args.image, args.camera_index, args.marker_id,
            args.marker_length_mm, args.aruco_dict, args.board_xyz, args.board_rpy_deg,
            args.robot_base_xyz, args.robot_base_rpy_deg, args.ground_z, args.output)


if __name__ == "__main__":
    main()
