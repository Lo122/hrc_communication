"""Extrinsic calibration for an iPhone connected via Record3D.

Reuses the SAME target-detection/solvePnP code as this folder's
extrinsic_calibration.py -- only the frame source changes (Record3D's RGBD
stream over USB instead of cv2.VideoCapture). See iphone_connection.py for
the connection wrapper. Two target methods, selected with --method (see
extrinsic_calibration.py's docstring for the tradeoffs):

- ``board`` (default): make_charuco_board/detect_charuco/solve_board_pose.
- ``marker``: make_aruco_detector/detect_marker_in_frame/solve_marker_pose.

Record3D also reports ARKit's own fused 6DOF camera pose per frame
(``get_camera_pose()``, the 4th element of ``IPhoneCamera.get_latest_frame()``
's tuple). Its position/yaw are relative to ARKit's own session-start
origin, not this project's world frame, so the board-based
T_world_from_camera computed here is still what actually defines
translation/yaw/scale and what the rest of this project consumes -- but its
ROTATION is gravity-aligned (ARKit fuses the IMU), which the board alone
can't guarantee: T_world_from_camera's roll/pitch are only as accurate as
how precisely the board was physically leveled and how well
--board-rpy-deg describes that. By default (--auto-gravity-correct, on
unless disabled) this script fixes that: at the moment you press SPACE, it
also reads the ARKit pose and applies the *minimal* rotation
(camera_utils.transforms.shortest_rotation_aligning) that makes
T_world_from_camera's implied "up" match ARKit's true gravity exactly,
without needing the board to be perfectly level or --board-rpy-deg's
roll/pitch to be exactly right -- useful when the camera itself is mounted
at some arbitrary/custom orientation rather than a fixed, carefully-leveled
one. It only touches roll/pitch (the component of orientation orthogonal to
gravity); yaw (heading) and origin/translation still come entirely from the
board + --board-xyz/--board-rpy-deg, same as before.

Physical setup and coordinate conventions are identical to
extrinsic_calibration.py -- see that file's docstring.

Usage:
    python -m camera_utils.iphone_extrinsic_calibration `
        --intrinsics ../dataset/calib_data/iphone_intrinsics.json `
        --squares-x 7 --squares-y 9 --square-length-mm 25 --marker-length-mm 19 `
        --capture-rotate90 90 `
        --output ../dataset/calib_data/iphone_extrinsics.json

Press SPACE to use the current frame's board detection, Q/ESC to abort.
Pass --no-auto-gravity-correct to use the board's assumed orientation as-is
(the old behavior), e.g. if you don't trust/want ARKit's gravity fusion.

If the preview looks sideways/upside-down (e.g. Record3D streaming the
phone's native sensor orientation rather than however you're holding it),
either rotate it by a fixed amount with --preview-rotate-deg 90/180/270/...,
or use --auto-level-preview to keep it level automatically every frame from
ARKit's gravity-aligned pose (see iphone_connection.roll_from_camera_pose).
Both are display-only -- board detection/solvePnP always use the raw frame,
and neither affects --auto-gravity-correct (a separate, non-cosmetic step).
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
from camera_utils.extrinsic_calibration import (
    detect_marker_in_frame, make_aruco_detector, solve_board_pose, solve_marker_pose,
)
from camera_utils.iphone_connection import (
    IPhoneCamera, camera_gravity_direction_cv, roll_from_camera_pose, rotate_camera_vector_90,
    rotate_frame,
)
from camera_utils import transforms as tf

DEFAULT_CALIB_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "calib_data"
IPHONE_EXTRINSICS_FILENAME = "iphone_extrinsics.json"


def capture_board_pose_from_iphone(dev_idx, board, detector, K, dist, min_corners=6,
                                    preview_rotate_deg=0.0, auto_level_preview=False,
                                    capture_rotate90=0):
    """preview_rotate_deg/auto_level_preview only affect the cv2.imshow preview.
    capture_rotate90 (0/90/180/270) is different: it rotates the actual
    working frame at the source (see iphone_connection.IPhoneCamera's
    docstring) BEFORE board detection/solvePnP ever see it -- K must be
    calibrated for that same capture_rotate90 (i.e. run
    iphone_intrinsic_calibration.py with the same value first). Returns
    (T_camera_from_board, arkit_pose) -- arkit_pose is the
    record3d.CameraPose from the SAME frame the board was accepted on
    (None if unavailable that frame), for --auto-gravity-correct."""
    print("Live extrinsic calibration: press SPACE to accept the current board "
          "detection, ESC/Q to abort.")
    T_camera_from_board = None
    accepted_pose = None
    with IPhoneCamera(dev_idx=dev_idx, capture_rotate90=capture_rotate90) as cam:
        try:
            while True:
                result = cam.get_latest_frame(timeout=2.0)
                if result is None:
                    status = "reconnecting..." if not cam.is_connected else "no frame (timeout)"
                    print(f"Waiting for iPhone ({status}).")
                    continue
                rgb, _depth, _intrinsic_mat, pose = result
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                charuco_corners, charuco_ids = detect_charuco(detector, gray, min_corners=min_corners)
                display = frame.copy()
                draw_charuco_detection(display, charuco_corners, charuco_ids)
                if charuco_corners is not None:
                    n = len(charuco_ids)
                    cv2.putText(display, f"board found ({n} corners) - SPACE to accept", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "board not found", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if auto_level_preview and pose is not None:
                    roll_deg = roll_from_camera_pose(pose)
                    display = rotate_frame(display, roll_deg)
                    cv2.putText(display, f"auto-level: {roll_deg:+.1f}deg", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
                elif preview_rotate_deg != 0.0:
                    display = rotate_frame(display, preview_rotate_deg)

                cv2.imshow("iphone extrinsic calibration", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and charuco_corners is not None:
                    T_camera_from_board = solve_board_pose(
                        charuco_corners, charuco_ids, board, K, dist, min_corners=min_corners)
                    if T_camera_from_board is not None:
                        accepted_pose = pose
                        break
                elif key in (27, ord("q")):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
    cv2.destroyAllWindows()

    return T_camera_from_board, accepted_pose


def capture_marker_pose_from_iphone(dev_idx, detector, marker_id, marker_length_m, K, dist,
                                     preview_rotate_deg=0.0, auto_level_preview=False,
                                     capture_rotate90=0):
    """Marker variant of capture_board_pose_from_iphone -- same preview/
    capture_rotate90 semantics, see that function's docstring. Returns
    (T_camera_from_marker, arkit_pose)."""
    print("Live extrinsic calibration: press SPACE to accept the current marker "
          "detection, ESC/Q to abort.")
    T_camera_from_marker = None
    accepted_pose = None
    with IPhoneCamera(dev_idx=dev_idx, capture_rotate90=capture_rotate90) as cam:
        try:
            while True:
                result = cam.get_latest_frame(timeout=2.0)
                if result is None:
                    status = "reconnecting..." if not cam.is_connected else "no frame (timeout)"
                    print(f"Waiting for iPhone ({status}).")
                    continue
                rgb, _depth, _intrinsic_mat, pose = result
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                corners_2d = detect_marker_in_frame(detector, frame, marker_id)
                display = frame.copy()
                if corners_2d is not None:
                    cv2.polylines(display, [corners_2d.astype(np.int32)], True, (0, 255, 0), 2)
                    cv2.putText(display, "marker found - SPACE to accept", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "marker not found", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if auto_level_preview and pose is not None:
                    roll_deg = roll_from_camera_pose(pose)
                    display = rotate_frame(display, roll_deg)
                    cv2.putText(display, f"auto-level: {roll_deg:+.1f}deg", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
                elif preview_rotate_deg != 0.0:
                    display = rotate_frame(display, preview_rotate_deg)

                cv2.imshow("iphone extrinsic calibration", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and corners_2d is not None:
                    T_camera_from_marker = solve_marker_pose(corners_2d, marker_length_m, K, dist)
                    if T_camera_from_marker is not None:
                        accepted_pose = pose
                        break
                elif key in (27, ord("q")):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
    cv2.destroyAllWindows()

    return T_camera_from_marker, accepted_pose


def gravity_correct_rotation(T_world_from_camera, arkit_pose, capture_rotate90=0):
    """Return (T_corrected, correction_deg): T_world_from_camera with its
    rotation nudged by the MINIMAL rotation that makes its implied "up"
    match ARKit's true measured gravity exactly (translation untouched).

    Only fixes the 2-DOF "which way is up" error -- yaw (heading around
    gravity) is whatever shortest_rotation_aligning leaves it at, i.e.
    essentially unchanged from the board-derived value, since the minimal
    rotation aligning two vectors doesn't add rotation about the vectors'
    own axis. correction_deg is how much the board-derived orientation was
    off from ARKit's gravity, mainly useful as a diagnostic (a large value
    likely means the board wasn't as level as --board-rpy-deg assumed, or
    the board itself was genuinely mounted at an angle).

    capture_rotate90 MUST match whatever was passed to IPhoneCamera for
    this capture (see its docstring): camera_gravity_direction_cv expresses
    gravity in the ORIGINAL, un-rotated camera axes, but T_camera_from_board
    (folded into T_world_from_camera already) was solved in the ACTUAL,
    capture_rotate90'd working frame -- rotate_camera_vector_90 re-expresses
    the gravity vector in that same rotated frame before combining them.
    Passing the wrong value here (or leaving it 0 when capture_rotate90 was
    actually nonzero) silently misapplies a ~90deg-class error disguised as
    a "correction".
    """
    gravity_camera_cv = camera_gravity_direction_cv(arkit_pose)
    if capture_rotate90:
        gravity_camera_cv = rotate_camera_vector_90(gravity_camera_cv, capture_rotate90)
    R_world_from_camera = T_world_from_camera[:3, :3]
    gravity_world_estimated = R_world_from_camera @ gravity_camera_cv
    true_gravity_world = np.array([0.0, 0.0, -1.0])  # world +Z is up (ground_z convention)

    correction_deg = float(np.degrees(np.arccos(
        np.clip(np.dot(gravity_world_estimated, true_gravity_world), -1.0, 1.0))))
    R_corr = tf.shortest_rotation_aligning(gravity_world_estimated, true_gravity_world)
    R_corrected = R_corr @ R_world_from_camera
    T_corrected = tf.make_transform(R_corrected, T_world_from_camera[:3, 3])
    return T_corrected, correction_deg


def run_iphone_extrinsic_calibration(intrinsics_path, dev_idx, squares_x, squares_y,
                                      square_length_mm, marker_length_mm, aruco_dict, min_corners,
                                      board_xyz, board_rpy_deg, robot_base_xyz, robot_base_rpy_deg,
                                      ground_z, preview_rotate_deg, auto_level_preview,
                                      auto_gravity_correct, capture_rotate90, output):
    """Shared entry point used by both this module's CLI and the
    calibrate_camera app. Returns T_world_from_camera, or None on failure."""
    K, dist, _ = load_intrinsics(intrinsics_path)
    board, detector = make_charuco_board(
        squares_x, squares_y,
        square_length_mm / 1000.0, marker_length_mm / 1000.0,
        aruco_dict)

    T_camera_from_board, arkit_pose = capture_board_pose_from_iphone(
        dev_idx, board, detector, K, dist, min_corners=min_corners,
        preview_rotate_deg=preview_rotate_deg, auto_level_preview=auto_level_preview,
        capture_rotate90=capture_rotate90)

    if T_camera_from_board is None:
        print("Failed to solve board pose. Aborting.")
        return None

    T_world_from_board = tf.make_transform(
        tf.rpy_deg_to_matrix(*board_rpy_deg), board_xyz)
    T_world_from_camera = tf.compose_transforms(
        T_world_from_board, tf.invert_transform(T_camera_from_board))

    if auto_gravity_correct:
        if arkit_pose is None:
            print("--auto-gravity-correct requested but no ARKit pose was available on the "
                  "accepted frame; using the board-derived orientation as-is.")
        else:
            T_world_from_camera, correction_deg = gravity_correct_rotation(
                T_world_from_camera, arkit_pose, capture_rotate90=capture_rotate90)
            print(f"Auto-gravity-correct: rotated T_world_from_camera by {correction_deg:.1f}deg "
                  f"to match ARKit's measured gravity (this is how far off the board-assumed "
                  f"orientation, i.e. --board-rpy-deg vs. how the board was actually placed, "
                  f"was from true level -- large values are worth investigating physically).")

    T_world_from_robot_base = tf.make_transform(
        tf.rpy_deg_to_matrix(*robot_base_rpy_deg), robot_base_xyz)

    print(f"T_world_from_camera =\n{T_world_from_camera}")

    save_extrinsics(
        output,
        T_world_from_camera,
        ground_z=ground_z,
        T_world_from_robot_base=T_world_from_robot_base,
        marker_id=None,
        notes=(f"charuco: squares={squares_x}x{squares_y}, square_length_mm={square_length_mm}, "
               f"marker_length_mm={marker_length_mm}, aruco_dict={aruco_dict}, source=Record3D, "
               f"auto_gravity_correct={auto_gravity_correct}, capture_rotate90={capture_rotate90}"),
    )
    print(f"Saved extrinsics to {output}")
    return T_world_from_camera


def run_iphone_extrinsic_calibration_marker(intrinsics_path, dev_idx, marker_id, marker_length_mm,
                                             aruco_dict, board_xyz, board_rpy_deg, robot_base_xyz,
                                             robot_base_rpy_deg, ground_z, preview_rotate_deg,
                                             auto_level_preview, auto_gravity_correct,
                                             capture_rotate90, output):
    """Marker variant of run_iphone_extrinsic_calibration -- solves against a
    single plain ArUco marker instead of a ChArUco board. Returns
    T_world_from_camera, or None on failure."""
    K, dist, _ = load_intrinsics(intrinsics_path)
    detector = make_aruco_detector(aruco_dict)
    marker_length_m = marker_length_mm / 1000.0

    T_camera_from_marker, arkit_pose = capture_marker_pose_from_iphone(
        dev_idx, detector, marker_id, marker_length_m, K, dist,
        preview_rotate_deg=preview_rotate_deg, auto_level_preview=auto_level_preview,
        capture_rotate90=capture_rotate90)

    if T_camera_from_marker is None:
        print("Failed to solve marker pose. Aborting.")
        return None

    T_world_from_marker = tf.make_transform(
        tf.rpy_deg_to_matrix(*board_rpy_deg), board_xyz)
    T_world_from_camera = tf.compose_transforms(
        T_world_from_marker, tf.invert_transform(T_camera_from_marker))

    if auto_gravity_correct:
        if arkit_pose is None:
            print("--auto-gravity-correct requested but no ARKit pose was available on the "
                  "accepted frame; using the marker-derived orientation as-is.")
        else:
            T_world_from_camera, correction_deg = gravity_correct_rotation(
                T_world_from_camera, arkit_pose, capture_rotate90=capture_rotate90)
            print(f"Auto-gravity-correct: rotated T_world_from_camera by {correction_deg:.1f}deg "
                  f"to match ARKit's measured gravity (this is how far off the marker-assumed "
                  f"orientation, i.e. --board-rpy-deg vs. how the marker was actually placed, "
                  f"was from true level -- large values are worth investigating physically).")

    T_world_from_robot_base = tf.make_transform(
        tf.rpy_deg_to_matrix(*robot_base_rpy_deg), robot_base_xyz)

    print(f"T_world_from_camera =\n{T_world_from_camera}")

    save_extrinsics(
        output,
        T_world_from_camera,
        ground_z=ground_z,
        T_world_from_robot_base=T_world_from_robot_base,
        marker_id=marker_id,
        notes=(f"aruco_dict={aruco_dict}, marker_length_mm={marker_length_mm}, "
               f"source=Record3D, auto_gravity_correct={auto_gravity_correct}, "
               f"capture_rotate90={capture_rotate90}"),
    )
    print(f"Saved extrinsics to {output}")
    return T_world_from_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", type=str, choices=("board", "marker"), default="board",
                         help="'board': ChArUco board (default). 'marker': single plain ArUco "
                              "marker -- see extrinsic_calibration.py's module docstring.")
    parser.add_argument("--intrinsics", type=str, required=True)
    parser.add_argument("--dev-idx", type=int, default=0,
                         help="Index into Record3D's connected-device list (see IPhoneCamera.list_devices).")
    parser.add_argument("--capture-rotate90", type=int, default=0, choices=(0, 90, 180, 270),
                         help="Rotate the ACTUAL working frame (not just the preview) by this many "
                              "degrees before target detection ever sees it -- fixes Record3D "
                              "streaming the phone's native sensor orientation regardless of how "
                              "it's physically held/mounted. Must match --intrinsics (i.e. re-run "
                              "iphone_intrinsic_calibration.py with the same --capture-rotate90 "
                              "first). See iphone_connection.py's module docstring for why this "
                              "matters far more than --preview-rotate-deg/--auto-level-preview below.")
    parser.add_argument("--squares-x", type=int, default=7, help="--method board only.")
    parser.add_argument("--squares-y", type=int, default=9, help="--method board only.")
    parser.add_argument("--square-length-mm", type=float, default=25.0, help="--method board only.")
    parser.add_argument("--marker-length-mm", type=float, default=19.0,
                         help="Side length of the embedded ArUco marker for --method board, or "
                              "of the whole marker for --method marker.")
    parser.add_argument("--marker-id", type=int, default=None,
                         help="--method marker only. Expected marker id. If omitted, uses the "
                              "first marker detected.")
    parser.add_argument("--aruco-dict", type=str, default="DICT_5X5_50")
    parser.add_argument("--min-corners", type=int, default=6, help="--method board only.")
    parser.add_argument("--board-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                         help="Board origin in world coords, meters. Default: board == world origin.")
    parser.add_argument("--board-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                         help="Board orientation in world coords, roll pitch yaw degrees.")
    parser.add_argument("--robot-base-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                         help="Robot base origin in world coords, meters.")
    parser.add_argument("--robot-base-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--ground-z", type=float, default=0.0,
                         help="World Z of the ground plane used later for ankle back-projection.")
    parser.add_argument("--preview-rotate-deg", type=float, default=0.0,
                         help="Rotate the cv2.imshow preview by this many degrees (CCW positive) "
                              "-- e.g. Record3D streaming the phone's native sensor orientation "
                              "instead of the orientation you're holding it in. Display-only: "
                              "board detection/solvePnP still run on the raw, un-rotated frame. "
                              "Ignored if --auto-level-preview is set.")
    parser.add_argument("--auto-level-preview", action="store_true",
                         help="Instead of a fixed --preview-rotate-deg, auto-level the preview "
                              "every frame using ARKit's fused camera pose (gravity-aligned) via "
                              "iphone_connection.roll_from_camera_pose -- keeps the horizon level "
                              "even if you tilt/roll the phone during calibration. Display-only, "
                              "same caveat as --preview-rotate-deg.")
    parser.add_argument("--no-auto-gravity-correct", dest="auto_gravity_correct",
                         action="store_false",
                         help="Disable auto-correcting T_world_from_camera's roll/pitch to match "
                              "ARKit's measured gravity (see module docstring) -- use the "
                              "board's assumed orientation (--board-rpy-deg) as-is instead, the "
                              "old behavior. On by default.")
    parser.set_defaults(auto_gravity_correct=True)
    parser.add_argument("--output", type=str,
                         default=str(DEFAULT_CALIB_DIR / IPHONE_EXTRINSICS_FILENAME),
                         help="Where to write the extrinsics JSON file. Defaults to "
                              f"{DEFAULT_CALIB_DIR / IPHONE_EXTRINSICS_FILENAME}.")
    args = parser.parse_args()

    if args.method == "board":
        run_iphone_extrinsic_calibration(
            args.intrinsics, args.dev_idx, args.squares_x, args.squares_y,
            args.square_length_mm, args.marker_length_mm, args.aruco_dict, args.min_corners,
            args.board_xyz, args.board_rpy_deg, args.robot_base_xyz, args.robot_base_rpy_deg,
            args.ground_z, args.preview_rotate_deg, args.auto_level_preview,
            args.auto_gravity_correct, args.capture_rotate90, args.output)
    else:
        run_iphone_extrinsic_calibration_marker(
            args.intrinsics, args.dev_idx, args.marker_id, args.marker_length_mm, args.aruco_dict,
            args.board_xyz, args.board_rpy_deg, args.robot_base_xyz, args.robot_base_rpy_deg,
            args.ground_z, args.preview_rotate_deg, args.auto_level_preview,
            args.auto_gravity_correct, args.capture_rotate90, args.output)


if __name__ == "__main__":
    main()
