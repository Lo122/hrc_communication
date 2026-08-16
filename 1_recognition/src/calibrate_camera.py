"""Camera calibration app: run ChArUco intrinsic and/or extrinsic
calibration and save the results as JSON into data_proc_3d/dataset/calib_data.

Uses the board printed by camera_utils/generate_charuco_board.py -- keep
--squares-x/--squares-y/--square-length-mm/--marker-length-mm/--aruco-dict
identical to whatever you passed that script.

Subcommands:

  intrinsic          Solve camera matrix K + distortion coeffs, from a
                     folder of images or a live webcam feed. Writes
                     intrinsics.json.

  extrinsic          Solve T_world_from_camera against a target placed at a
                     known world pose, using an existing intrinsics.json,
                     from a live webcam feed. Writes extrinsics.json.
                     --method board (default) uses a ChArUco board;
                     --method marker uses a single plain ArUco marker.

  full               Run intrinsic then extrinsic back to back on a
                     webcam (extrinsic re-uses the intrinsics.json this
                     run just produced).

  iphone-intrinsic   Same as intrinsic, but reads frames from an iPhone
                     connected via Record3D instead of a webcam. Writes
                     iphone_intrinsics.json.

  iphone-extrinsic   Same as extrinsic, but reads frames from an iPhone
                     connected via Record3D and (by default) auto-corrects
                     roll/pitch against ARKit's fused gravity sensing.
                     Writes iphone_extrinsics.json.

  iphone-full        Run iphone-intrinsic then iphone-extrinsic back to
                     back.

Examples:
    uv run python 1_recognition/src/calibrate_camera.py intrinsic --camera-index 0 `
        --squares-x 7 --squares-y 9 --square-length-mm 25 --marker-length-mm 19

    uv run python 1_recognition/src/calibrate_camera.py extrinsic --camera-index 0 `
        --squares-x 7 --squares-y 9 --square-length-mm 25 --marker-length-mm 19 `
        --ground-z 0.0

    uv run python 1_recognition/src/calibrate_camera.py extrinsic --method marker --camera-index 0 `
        --marker-id 0 --marker-length-mm 100 --ground-z 0.0

    uv run python 1_recognition/src/calibrate_camera.py full --images-dir path/to/imgs `
        --squares-x 7 --squares-y 9 --square-length-mm 25 --marker-length-mm 19

    uv run python 1_recognition/src/calibrate_camera.py iphone-full --capture-rotate90 90 `
        --squares-x 7 --squares-y 9 --square-length-mm 25 --marker-length-mm 19
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_utils.intrinsic_calibration import run_intrinsic_calibration
from camera_utils.extrinsic_calibration import (
    run_extrinsic_calibration, run_extrinsic_calibration_marker,
)
from camera_utils.iphone_intrinsic_calibration import run_iphone_intrinsic_calibration
from camera_utils.iphone_extrinsic_calibration import (
    run_iphone_extrinsic_calibration, run_iphone_extrinsic_calibration_marker,
)

CALIB_DATA_DIR = (Path(__file__).resolve().parents[1] / "calib_data")
DEFAULT_INTRINSICS_PATH = CALIB_DATA_DIR / "intrinsics.json"
DEFAULT_EXTRINSICS_PATH = CALIB_DATA_DIR / "extrinsics.json"
DEFAULT_IPHONE_INTRINSICS_PATH = CALIB_DATA_DIR / "iphone_intrinsics.json"
DEFAULT_IPHONE_EXTRINSICS_PATH = CALIB_DATA_DIR / "iphone_extrinsics.json"


def add_board_args(parser, required=True):
    parser.add_argument("--squares-x", type=int, required=required, default=None if required else 7,
                         help="Number of full checkerboard squares along the board's width.")
    parser.add_argument("--squares-y", type=int, required=required, default=None if required else 9,
                         help="Number of full checkerboard squares along the board's height.")
    parser.add_argument("--square-length-mm", type=float, required=required,
                         default=None if required else 25.0)
    parser.add_argument("--marker-length-mm", type=float, required=required,
                         default=None if required else 19.0,
                         help="Must be smaller than --square-length-mm.")
    parser.add_argument("--aruco-dict", type=str, default="DICT_5X5_50")
    parser.add_argument("--min-corners", type=int, default=6,
                         help="Minimum ChArUco corners required to accept a view/pose.")


def add_intrinsic_args(parser):
    parser.add_argument("--images-dir", type=str, default=None,
                         help="Folder of ChArUco board images (jpg/png). If omitted, "
                              "captures live from --camera-index instead.")
    parser.add_argument("--camera-index", type=int, default=0)
    add_board_args(parser)
    parser.add_argument("--output", type=str, default=DEFAULT_INTRINSICS_PATH,
                         help=f"Where to write intrinsics.json (default: {DEFAULT_INTRINSICS_PATH}).")


def add_extrinsic_args(parser):
    parser.add_argument("--method", type=str, choices=("board", "marker"), default="board",
                         help="'board': ChArUco board (default). 'marker': single plain ArUco "
                              "marker -- see camera_utils/extrinsic_calibration.py's docstring.")
    parser.add_argument("--intrinsics", type=str, default=DEFAULT_INTRINSICS_PATH,
                         help=f"Path to intrinsics.json (default: {DEFAULT_INTRINSICS_PATH}).")
    parser.add_argument("--image", type=str, default=None,
                         help="Single still image to calibrate from instead of live capture.")
    parser.add_argument("--camera-index", type=int, default=0)
    add_board_args(parser, required=False)
    parser.add_argument("--marker-id", type=int, default=None,
                         help="--method marker only. Expected marker id. If omitted, uses the "
                              "first marker detected.")
    parser.add_argument("--board-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                         help="Board origin in world coords, meters. Default: board == world origin.")
    parser.add_argument("--board-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                         help="Board orientation in world coords, roll pitch yaw degrees.")
    parser.add_argument("--robot-base-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                         help="Robot base origin in world coords, meters.")
    parser.add_argument("--robot-base-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--ground-z", type=float, default=0.0,
                         help="World Z of the ground plane used later for ankle back-projection.")
    parser.add_argument("--output", type=str, default=DEFAULT_EXTRINSICS_PATH,
                         help=f"Where to write extrinsics.json (default: {DEFAULT_EXTRINSICS_PATH}).")


def add_iphone_intrinsic_args(parser):
    parser.add_argument("--dev-idx", type=int, default=0,
                         help="Index into Record3D's connected-device list.")
    parser.add_argument("--capture-rotate90", type=int, default=0, choices=(0, 90, 180, 270),
                         help="Rotate the working frame at the source before it's used at all -- "
                              "must match what you pass to iphone-extrinsic afterward.")
    parser.add_argument("--use-reported-intrinsics", action="store_true",
                         help="Skip the board and just average Record3D's own per-frame reported "
                              "intrinsic matrix instead.")
    parser.add_argument("--num-samples", type=int, default=60,
                         help="Number of frames to average when --use-reported-intrinsics is set.")
    add_board_args(parser, required=False)
    parser.add_argument("--output", type=str, default=DEFAULT_IPHONE_INTRINSICS_PATH,
                         help=f"Where to write iphone_intrinsics.json (default: "
                              f"{DEFAULT_IPHONE_INTRINSICS_PATH}).")


def add_iphone_extrinsic_args(parser):
    parser.add_argument("--method", type=str, choices=("board", "marker"), default="board",
                         help="'board': ChArUco board (default). 'marker': single plain ArUco "
                              "marker -- see camera_utils/extrinsic_calibration.py's docstring.")
    parser.add_argument("--intrinsics", type=str, default=DEFAULT_IPHONE_INTRINSICS_PATH,
                         help=f"Path to iphone_intrinsics.json (default: "
                              f"{DEFAULT_IPHONE_INTRINSICS_PATH}).")
    parser.add_argument("--dev-idx", type=int, default=0,
                         help="Index into Record3D's connected-device list.")
    parser.add_argument("--capture-rotate90", type=int, default=0, choices=(0, 90, 180, 270),
                         help="Must match --intrinsics' --capture-rotate90.")
    add_board_args(parser, required=False)
    parser.add_argument("--marker-id", type=int, default=None,
                         help="--method marker only. Expected marker id. If omitted, uses the "
                              "first marker detected.")
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
                         help="Rotate the cv2.imshow preview only (display-only, CCW positive).")
    parser.add_argument("--auto-level-preview", action="store_true",
                         help="Auto-level the preview every frame using ARKit's gravity-aligned "
                              "pose instead of a fixed --preview-rotate-deg. Display-only.")
    parser.add_argument("--no-auto-gravity-correct", dest="auto_gravity_correct",
                         action="store_false",
                         help="Disable auto-correcting T_world_from_camera's roll/pitch to match "
                              "ARKit's measured gravity -- use the board's assumed orientation "
                              "(--board-rpy-deg) as-is instead. On by default.")
    parser.set_defaults(auto_gravity_correct=True)
    parser.add_argument("--output", type=str, default=DEFAULT_IPHONE_EXTRINSICS_PATH,
                         help=f"Where to write iphone_extrinsics.json (default: "
                              f"{DEFAULT_IPHONE_EXTRINSICS_PATH}).")


def do_intrinsic(args):
    run_intrinsic_calibration(
        args.images_dir, args.camera_index, args.squares_x, args.squares_y,
        args.square_length_mm, args.marker_length_mm, args.aruco_dict,
        args.min_corners, args.output)


def do_extrinsic(args):
    if args.method == "board":
        if args.squares_x is None or args.squares_y is None or args.square_length_mm is None:
            raise RuntimeError("--squares-x, --squares-y and --square-length-mm are required "
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


def do_full(args):
    print("=== Step 1/2: intrinsic calibration ===")
    run_intrinsic_calibration(
        args.images_dir, args.camera_index, args.squares_x, args.squares_y,
        args.square_length_mm, args.marker_length_mm, args.aruco_dict,
        args.min_corners, args.intrinsics_output)

    print("\n=== Step 2/2: extrinsic calibration ===")
    if args.images_dir:
        print("(--images-dir was used for intrinsics; extrinsic still needs a live camera "
              "or --image since it requires a single current board/marker placement.)")
    if args.method == "board":
        run_extrinsic_calibration(
            args.intrinsics_output, args.image, args.camera_index, args.squares_x, args.squares_y,
            args.square_length_mm, args.marker_length_mm, args.aruco_dict, args.min_corners,
            args.board_xyz, args.board_rpy_deg, args.robot_base_xyz, args.robot_base_rpy_deg,
            args.ground_z, args.extrinsics_output)
    else:
        run_extrinsic_calibration_marker(
            args.intrinsics_output, args.image, args.camera_index, args.marker_id,
            args.marker_length_mm, args.aruco_dict, args.board_xyz, args.board_rpy_deg,
            args.robot_base_xyz, args.robot_base_rpy_deg, args.ground_z, args.extrinsics_output)


def do_iphone_intrinsic(args):
    run_iphone_intrinsic_calibration(
        args.use_reported_intrinsics, args.dev_idx, args.num_samples,
        args.squares_x, args.squares_y, args.square_length_mm, args.marker_length_mm,
        args.aruco_dict, args.min_corners, args.capture_rotate90, args.output)


def do_iphone_extrinsic(args):
    if args.method == "board":
        if args.squares_x is None or args.squares_y is None or args.square_length_mm is None:
            raise RuntimeError("--squares-x, --squares-y and --square-length-mm are required "
                               "for --method board.")
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


def do_iphone_full(args):
    print("=== Step 1/2: iPhone intrinsic calibration ===")
    run_iphone_intrinsic_calibration(
        args.use_reported_intrinsics, args.dev_idx, args.num_samples,
        args.squares_x, args.squares_y, args.square_length_mm, args.marker_length_mm,
        args.aruco_dict, args.min_corners, args.capture_rotate90, args.intrinsics_output)

    print("\n=== Step 2/2: iPhone extrinsic calibration ===")
    if args.method == "board":
        run_iphone_extrinsic_calibration(
            args.intrinsics_output, args.dev_idx, args.squares_x, args.squares_y,
            args.square_length_mm, args.marker_length_mm, args.aruco_dict, args.min_corners,
            args.board_xyz, args.board_rpy_deg, args.robot_base_xyz, args.robot_base_rpy_deg,
            args.ground_z, args.preview_rotate_deg, args.auto_level_preview,
            args.auto_gravity_correct, args.capture_rotate90, args.extrinsics_output)
    else:
        run_iphone_extrinsic_calibration_marker(
            args.intrinsics_output, args.dev_idx, args.marker_id, args.marker_length_mm,
            args.aruco_dict, args.board_xyz, args.board_rpy_deg, args.robot_base_xyz,
            args.robot_base_rpy_deg, args.ground_z, args.preview_rotate_deg,
            args.auto_level_preview, args.auto_gravity_correct, args.capture_rotate90,
            args.extrinsics_output)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_intrinsic = sub.add_parser("intrinsic", help="Solve camera matrix + distortion.")
    add_intrinsic_args(p_intrinsic)
    p_intrinsic.set_defaults(func=do_intrinsic)

    p_extrinsic = sub.add_parser("extrinsic", help="Solve T_world_from_camera.")
    add_extrinsic_args(p_extrinsic)
    p_extrinsic.set_defaults(func=do_extrinsic)

    p_full = sub.add_parser("full", help="Run intrinsic then extrinsic back to back.")
    p_full.add_argument("--images-dir", type=str, default=None,
                         help="Folder of ChArUco board images for the intrinsic step. If "
                              "omitted, both steps capture live from --camera-index.")
    p_full.add_argument("--camera-index", type=int, default=0)
    p_full.add_argument("--image", type=str, default=None,
                         help="Single still image for the extrinsic step (instead of live capture).")
    p_full.add_argument("--method", type=str, choices=("board", "marker"), default="board",
                         help="Extrinsic step's target: 'board' (default, ChArUco) or 'marker' "
                              "(single plain ArUco marker). The intrinsic step always uses the "
                              "ChArUco board.")
    add_board_args(p_full)
    p_full.add_argument("--marker-id", type=int, default=None,
                         help="--method marker only. Expected marker id. If omitted, uses the "
                              "first marker detected.")
    p_full.add_argument("--board-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p_full.add_argument("--board-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p_full.add_argument("--robot-base-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p_full.add_argument("--robot-base-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p_full.add_argument("--ground-z", type=float, default=0.0)
    p_full.add_argument("--intrinsics-output", type=str, default=DEFAULT_INTRINSICS_PATH)
    p_full.add_argument("--extrinsics-output", type=str, default=DEFAULT_EXTRINSICS_PATH)
    p_full.set_defaults(func=do_full)

    p_iphone_intrinsic = sub.add_parser(
        "iphone-intrinsic", help="Solve camera matrix + distortion from an iPhone/Record3D feed.")
    add_iphone_intrinsic_args(p_iphone_intrinsic)
    p_iphone_intrinsic.set_defaults(func=do_iphone_intrinsic)

    p_iphone_extrinsic = sub.add_parser(
        "iphone-extrinsic", help="Solve T_world_from_camera from an iPhone/Record3D feed.")
    add_iphone_extrinsic_args(p_iphone_extrinsic)
    p_iphone_extrinsic.set_defaults(func=do_iphone_extrinsic)

    p_iphone_full = sub.add_parser(
        "iphone-full", help="Run iphone-intrinsic then iphone-extrinsic back to back.")
    p_iphone_full.add_argument("--dev-idx", type=int, default=0)
    p_iphone_full.add_argument("--capture-rotate90", type=int, default=0, choices=(0, 90, 180, 270))
    p_iphone_full.add_argument("--use-reported-intrinsics", action="store_true")
    p_iphone_full.add_argument("--num-samples", type=int, default=60)
    p_iphone_full.add_argument("--method", type=str, choices=("board", "marker"), default="board",
                                help="Extrinsic step's target: 'board' (default, ChArUco) or "
                                     "'marker' (single plain ArUco marker). The intrinsic step "
                                     "always uses the ChArUco board.")
    add_board_args(p_iphone_full, required=False)
    p_iphone_full.add_argument("--marker-id", type=int, default=None,
                                help="--method marker only. Expected marker id. If omitted, uses "
                                     "the first marker detected.")
    p_iphone_full.add_argument("--board-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p_iphone_full.add_argument("--board-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p_iphone_full.add_argument("--robot-base-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p_iphone_full.add_argument("--robot-base-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p_iphone_full.add_argument("--ground-z", type=float, default=0.0)
    p_iphone_full.add_argument("--preview-rotate-deg", type=float, default=0.0)
    p_iphone_full.add_argument("--auto-level-preview", action="store_true")
    p_iphone_full.add_argument("--no-auto-gravity-correct", dest="auto_gravity_correct",
                                action="store_false")
    p_iphone_full.set_defaults(auto_gravity_correct=True)
    p_iphone_full.add_argument("--intrinsics-output", type=str, default=DEFAULT_IPHONE_INTRINSICS_PATH)
    p_iphone_full.add_argument("--extrinsics-output", type=str, default=DEFAULT_IPHONE_EXTRINSICS_PATH)
    p_iphone_full.set_defaults(func=do_iphone_full)

    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as e:
        print(f"Calibration failed: {e}")


if __name__ == "__main__":
    main()
