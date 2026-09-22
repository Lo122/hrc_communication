"""Camera calibration app: run ChArUco intrinsic and/or extrinsic
calibration and save the results as JSON into 1_recognition/calib_data.

Uses the board printed by generate_charuco_board.py in the data-processing
repo (LSTM_HRC/data_proc_3d/src/camera_utils/) -- this repo only DETECTS the
board and solves from it. Keep
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

K is only valid at the resolution it was solved at. Set it on the intrinsic
step with --capture-width/--capture-height (setup/video_source.py --list shows
what each index can deliver); the extrinsic step reads it back out of the
intrinsics file. iphone-* has no such flag -- Record3D streams whatever the app
is set to, and --capture-rotate90 must match on every command including
run_recognition.py's --iphone-rotate.

--ground-z is the FLOOR's height relative to the target: 0.0 on the floor,
-0.75 on a 0.75 m table.

Webcam:
    # intrinsic, 1080p (use 3840x2160 for 4K)
    uv run python 1_recognition/setup/calibrate_camera.py intrinsic --camera-index 6 `
        --capture-width 640 --capture-height 480 `
        --squares-x 7 --squares-y 9 --square-length-mm 38 --marker-length-mm 28 `
        --output 1_recognition/calib_data/intrinsics_640x480_obs.json

    # extrinsic, ArUco marker
    uv run python 1_recognition/setup/calibrate_camera.py extrinsic --method marker `
        --intrinsics 1_recognition/calib_data/intrinsics_3840x2160_obs.json `
        --camera-index 6 --marker-id 0 --marker-length-mm 200 `
        --aruco-dict DICT_4X4_50 `
        --capture-width 3840 --capture-height 2160 `
        --output 1_recognition/calib_data/extrinsics_3840x2160_obs.json `
        --ground-z -0.70

    # both back to back
    uv run python 1_recognition/setup/calibrate_camera.py full --camera-index 0 `
        --capture-width 1920 --capture-height 1080 `
        --squares-x 7 --squares-y 9 --square-length-mm 25 --marker-length-mm 19 `
        --output 1_recognition/calib_data/intrinsics_640x480_obs.json

Record3D / iPhone:
    # Record3D's own reported K -- no board; image_size in the JSON tells you
    # the stream resolution (post-rotation)
    uv run python 1_recognition/setup/calibrate_camera.py iphone-intrinsic `
        --capture-rotate90 270 --use-reported-intrinsics `
        --output 1_recognition/calib_data/iphone_intrinsics_reported.json

    # board-measured K, to compare against it -- needs its own --output
    uv run python 1_recognition/setup/calibrate_camera.py iphone-intrinsic `
        --capture-rotate90 270 `
        --capture-width 1920 --capture-height 1440 `
        --squares-x 7 --squares-y 9 --square-length-mm 25 --marker-length-mm 19 `
        --output 1_recognition/calib_data/iphone_intrinsics_board.json

    # extrinsic
    uv run python 1_recognition/setup/calibrate_camera.py iphone-extrinsic `
        --capture-rotate90 270 --method marker `
        --intrinsics 1_recognition/calib_data/iphone_intrinsics.json `
        --capture-width 1920 --capture-height 1440 `
        --output 1_recognition/calib_data/iphone_extrinsics.json `
        --aruco-dict DICT_4X4_50 `
        --marker-id 0 --marker-length-mm 200 --ground-z -0.70

    # both back to back (add --use-reported-intrinsics to skip the board)
    $env:UV_PROJECT_ENVIRONMENT = "C:\\Users\\Owner\\.venvs\\hrc_communication"
    uv run python 1_recognition/setup/calibrate_camera.py iphone-full `
        --capture-rotate90 270 `
        --use-reported-intrinsics --num-samples 60 `
        --method marker --marker-id 0 --marker-length-mm 200 --aruco-dict DICT_4X4_50 `
        --board-rpy-deg 90 0 0 `
        --ground-z -0.70 `
        --intrinsics-output 1_recognition/calib_data/iphone_intrinsics.json `
        --extrinsics-output 1_recognition/calib_data/iphone_extrinsics.json
        """

import argparse
import sys
from pathlib import Path

# src/ for camera_utils (runtime camera code), and setup/ so the sibling
# calibration/ package resolves when this is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration.intrinsic_calibration import run_intrinsic_calibration
from calibration.extrinsic_calibration import (
    run_extrinsic_calibration, run_extrinsic_calibration_marker,
)
from calibration.iphone_intrinsic_calibration import run_iphone_intrinsic_calibration
from calibration.iphone_extrinsic_calibration import (
    run_iphone_extrinsic_calibration, run_iphone_extrinsic_calibration_marker,
)
from logging_setup import configure_logging, get_logger
from video_source import add_capture_args

logger = get_logger(__name__)

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
    # Live capture only. The resolution captured here is the resolution the
    # resulting K is valid for, so it is effectively part of the output.
    add_capture_args(parser)
    parser.add_argument("--output", type=str, default=DEFAULT_INTRINSICS_PATH,
                         help=f"Where to write intrinsics.json (default: {DEFAULT_INTRINSICS_PATH}).")


def add_extrinsic_args(parser):
    parser.add_argument("--method", type=str, choices=("board", "marker"), default="board",
                         help="'board': ChArUco board (default). 'marker': single plain ArUco "
                              "marker -- see calibration/extrinsic_calibration.py's docstring.")
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
                         help="World Z of the ground plane used later for ankle back-projection. "
                              "This is the FLOOR's height relative to the target, not the "
                              "camera's: 0.0 only if the target lies on the floor, -0.75 for a "
                              "target on a 0.75 m table.")
    # Defaults to the resolution --intrinsics was solved at, which is almost always
    # what is wanted -- K does not transfer between resolutions.
    add_capture_args(parser)
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
                              "marker -- see calibration/extrinsic_calibration.py's docstring.")
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
        args.min_corners, args.output,
        capture_width=args.capture_width, capture_height=args.capture_height,
        backend=args.backend, preview_width=args.preview_width)


def do_extrinsic(args):
    if args.method == "board":
        if args.squares_x is None or args.squares_y is None or args.square_length_mm is None:
            raise RuntimeError("--squares-x, --squares-y and --square-length-mm are required "
                               "for --method board.")
        run_extrinsic_calibration(
            args.intrinsics, args.image, args.camera_index, args.squares_x, args.squares_y,
            args.square_length_mm, args.marker_length_mm, args.aruco_dict, args.min_corners,
            args.board_xyz, args.board_rpy_deg, args.robot_base_xyz, args.robot_base_rpy_deg,
            args.ground_z, args.output,
            capture_width=args.capture_width, capture_height=args.capture_height,
            backend=args.backend, preview_width=args.preview_width)
    else:
        run_extrinsic_calibration_marker(
            args.intrinsics, args.image, args.camera_index, args.marker_id,
            args.marker_length_mm, args.aruco_dict, args.board_xyz, args.board_rpy_deg,
            args.robot_base_xyz, args.robot_base_rpy_deg, args.ground_z, args.output,
            capture_width=args.capture_width, capture_height=args.capture_height,
            backend=args.backend, preview_width=args.preview_width)


def do_full(args):
    print("=== Step 1/2: intrinsic calibration ===")
    run_intrinsic_calibration(
        args.images_dir, args.camera_index, args.squares_x, args.squares_y,
        args.square_length_mm, args.marker_length_mm, args.aruco_dict,
        args.min_corners, args.intrinsics_output,
        capture_width=args.capture_width, capture_height=args.capture_height,
        backend=args.backend, preview_width=args.preview_width)

    print("\n=== Step 2/2: extrinsic calibration ===")
    if args.images_dir:
        logger.info("(--images-dir was used for intrinsics; extrinsic still needs a live camera "
              "or --image since it requires a single current board/marker placement.)")
    # No capture size passed on purpose: the extrinsic step reads it back from the
    # intrinsics file step 1 just wrote, so the two cannot disagree even if the
    # camera refused the requested mode.
    if args.method == "board":
        run_extrinsic_calibration(
            args.intrinsics_output, args.image, args.camera_index, args.squares_x, args.squares_y,
            args.square_length_mm, args.marker_length_mm, args.aruco_dict, args.min_corners,
            args.board_xyz, args.board_rpy_deg, args.robot_base_xyz, args.robot_base_rpy_deg,
            args.ground_z, args.extrinsics_output, backend=args.backend,
            preview_width=args.preview_width)
    else:
        run_extrinsic_calibration_marker(
            args.intrinsics_output, args.image, args.camera_index, args.marker_id,
            args.marker_length_mm, args.aruco_dict, args.board_xyz, args.board_rpy_deg,
            args.robot_base_xyz, args.robot_base_rpy_deg, args.ground_z, args.extrinsics_output,
            backend=args.backend, preview_width=args.preview_width)


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
    configure_logging("calibrate_camera")
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
    # Applies to both steps, which is the point: they must agree, and here they
    # cannot disagree because the same value drives each.
    add_capture_args(p_full)
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
        logger.error("Calibration failed: %s", e)


if __name__ == "__main__":
    main()
