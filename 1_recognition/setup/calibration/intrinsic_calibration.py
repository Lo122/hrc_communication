"""ChArUco-board intrinsic calibration (camera matrix K + distortion coeffs).

Two ways to get calibration images:

1. Live capture from a camera:
   uv run python 1_recognition/setup/calibration/intrinsic_calibration.py `
       --camera-index 0 --squares-x 7 --squares-y 9 `
       --square-length-mm 25 --marker-length-mm 19 `
       --output 1_recognition/calib_data/intrinsics.json
   Press SPACE to capture a frame once the board is detected, ESC/Q to stop
   capturing and run the calibration.

2. From a folder of already-captured images:
   uv run python 1_recognition/setup/calibration/intrinsic_calibration.py `
       --images-dir path/to/imgs --squares-x 7 --squares-y 9 `
       --square-length-mm 25 --marker-length-mm 19 `
       --output 1_recognition/calib_data/intrinsics.json

``--squares-x``/``--squares-y`` are the number of full checkerboard squares
along each side (including the black ones) -- must match the board used to
generate the print, i.e. generate_charuco_board.py's --squares-x/--squares-y
(that script lives in the data-processing repo, LSTM_HRC/data_proc_3d/src/camera_utils/).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    # src/ for camera_utils, setup/ for the sibling calibration package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_utils.calibration_io import save_intrinsics
from calibration.charuco_board import detect_charuco, draw_charuco_detection, make_charuco_board
from logging_setup import configure_logging, get_logger
from video_source import (
    DEFAULT_PREVIEW_WIDTH, add_capture_args, forget_preview_windows, open_camera,
    preview_scale, show_preview)

logger = get_logger(__name__)


def imread_unicode(path):
    """cv2.imread(path) silently fails on Windows if path has non-ASCII chars
    (e.g. this repo's own "Universität Stuttgart" parent folder) -- decode
    via numpy instead, which goes through Python's own (Unicode-safe) file I/O."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """See imread_unicode -- same non-ASCII-path issue, write side."""
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if ok:
        buf.tofile(path)
    return ok


def calibrate_from_images(image_paths, board, detector, min_corners=6):
    all_object_points, all_image_points = [], []
    image_size = None
    used, skipped = 0, 0

    for path in image_paths:
        img = imread_unicode(path)
        if img is None:
            skipped += 1
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)

        charuco_corners, charuco_ids = detect_charuco(detector, gray, min_corners=min_corners)
        if charuco_corners is None:
            logger.debug("  [skip] fewer than %d ChArUco corners found in %s",
                         min_corners, Path(path).name)
            skipped += 1
            continue

        object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
        if object_points is None or len(object_points) < min_corners:
            skipped += 1
            continue

        all_object_points.append(object_points)
        all_image_points.append(image_points)
        used += 1

    if used < 5:
        raise RuntimeError(
            f"Only found the board in {used} image(s); need at least ~5-10 "
            f"varied views for a stable calibration ({skipped} images skipped)."
        )

    logger.info("Calibrating from %d views (%d skipped)...", used, skipped)
    reprojection_error, K, dist, _, _ = cv2.calibrateCamera(
        all_object_points, all_image_points, image_size, None, None)

    return K, dist, image_size, reprojection_error


def capture_from_camera(camera_index, detector, min_corners=6,
                         capture_width=None, capture_height=None, backend="auto",
                         preview_width=DEFAULT_PREVIEW_WIDTH):
    """Live ChArUco capture. The resolution requested here is the resolution
    the resulting K is valid for, and nothing downstream can recover from
    getting it wrong -- on Windows/DirectShow an unasked camera negotiates
    640x480 however capable it is, so a 1080p camera would silently produce a
    640x480 calibration. open_camera requests a mode and VERIFIES it against a
    decoded frame (the capture properties lie; see video_source.py)."""
    cap, actual = open_camera(camera_index, capture_width, capture_height, backend)
    # Overlays are drawn full-size but shown downscaled; scale them to match.
    s = preview_scale(actual[0], preview_width)
    logger.info("Calibrating at %dx%d -- the intrinsics will only be valid at this "
                "resolution.", actual[0], actual[1])

    print("Live capture: press SPACE to capture a frame when the ChArUco board "
          "is highlighted, ESC or Q to finish and calibrate.")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids = detect_charuco(detector, gray, min_corners=min_corners)
            display = frame.copy()
            draw_charuco_detection(display, charuco_corners, charuco_ids)
            n_corners = 0 if charuco_ids is None else len(charuco_ids)
            cv2.putText(display, f"captured: {len(frames)}  (corners this frame: {n_corners})",
                        (int(10 * s), int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                        (0, 255, 0), max(int(2 * s), 1))
            show_preview("intrinsic calibration - SPACE=capture, Q=done", display, preview_width)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and charuco_corners is not None:
                frames.append(frame.copy())
                print(f"  captured frame {len(frames)} ({n_corners} corners)")
            elif key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        forget_preview_windows()

    return frames


def run_intrinsic_calibration(images_dir, camera_index, squares_x, squares_y,
                               square_length_mm, marker_length_mm, aruco_dict,
                               min_corners, output, capture_width=None,
                               capture_height=None, backend="auto",
                               preview_width=DEFAULT_PREVIEW_WIDTH):
    """Shared entry point used by both this module's CLI and the
    calibrate_camera app. Returns (K, dist, image_size, reprojection_error).

    capture_width/height apply to live capture only -- images from
    --images-dir are already whatever size they were taken at. Either way the
    image_size written into the JSON is the size actually measured from the
    frames, never the size that was requested."""
    board, detector = make_charuco_board(
        squares_x, squares_y,
        square_length_mm / 1000.0, marker_length_mm / 1000.0,
        aruco_dict)

    if images_dir:
        images_dir = Path(images_dir)
        image_paths = sorted(
            list(images_dir.glob("*.jpg"))
            + list(images_dir.glob("*.jpeg"))
            + list(images_dir.glob("*.png"))
        )
        if not image_paths:
            raise RuntimeError(f"No images found in {images_dir}")
        K, dist, image_size, err = calibrate_from_images(
            image_paths, board, detector, min_corners=min_corners)
    else:
        frames = capture_from_camera(camera_index, detector, min_corners=min_corners,
                                     capture_width=capture_width,
                                     capture_height=capture_height, backend=backend,
                                     preview_width=preview_width)
        if len(frames) < 5:
            raise RuntimeError(f"Only captured {len(frames)} frame(s); need at least ~5-10.")
        tmp_dir = Path(output).resolve().parent / "_capture_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_paths = []
        for i, frame in enumerate(frames):
            p = tmp_dir / f"frame_{i:03d}.png"
            imwrite_unicode(p, frame)
            tmp_paths.append(p)
        K, dist, image_size, err = calibrate_from_images(
            tmp_paths, board, detector, min_corners=min_corners)

    logger.info("Reprojection error: %.4f px", err)
    logger.info("K =\n%s", K)
    logger.info("dist = %s", dist.ravel())

    save_intrinsics(output, K, dist, image_size, reprojection_error=err)
    logger.info("Saved intrinsics to %s", output)
    return K, dist, image_size, err


def main():
    configure_logging("intrinsic_calibration")
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=str, default=None,
                         help="Folder of ChArUco board images (jpg/png). If omitted, "
                              "captures live from --camera-index instead.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--squares-x", type=int, required=True,
                         help="Number of full checkerboard squares along the board's width.")
    parser.add_argument("--squares-y", type=int, required=True,
                         help="Number of full checkerboard squares along the board's height.")
    parser.add_argument("--square-length-mm", type=float, required=True)
    parser.add_argument("--marker-length-mm", type=float, required=True,
                         help="Must be smaller than --square-length-mm (the ArUco marker sits "
                              "inside each black square with a white margin).")
    parser.add_argument("--aruco-dict", type=str, default="DICT_5X5_50")
    parser.add_argument("--min-corners", type=int, default=6,
                         help="Minimum ChArUco corners required to accept a view.")
    add_capture_args(parser)
    parser.add_argument("--output", type=str, required=True,
                         help="Where to write the intrinsics JSON file.")
    args = parser.parse_args()

    try:
        run_intrinsic_calibration(
            args.images_dir, args.camera_index, args.squares_x, args.squares_y,
            args.square_length_mm, args.marker_length_mm, args.aruco_dict,
            args.min_corners, args.output,
            capture_width=args.capture_width, capture_height=args.capture_height,
            backend=args.backend, preview_width=args.preview_width)
    except RuntimeError as e:
        logger.error("%s", e)


if __name__ == "__main__":
    main()
