"""Intrinsic calibration for an iPhone connected via Record3D.

Reuses the SAME ChArUco board/detection code as this folder's
intrinsic_calibration.py (make_charuco_board/detect_charuco/
calibrate_from_images) -- only the frame source changes (Record3D's RGBD
stream over USB instead of cv2.VideoCapture). See iphone_connection.py for
the connection wrapper. Uses the board printed by generate_charuco_board.py
-- keep --squares-x/--squares-y/--square-length-mm/--marker-length-mm/
--aruco-dict identical to whatever you passed that script.

Two modes:

1. ChArUco board calibration (matches intrinsic_calibration.py's method,
   useful to verify/compare against Record3D's own reported intrinsics):
   python -m camera_utils.iphone_intrinsic_calibration `
       --squares-x 7 --squares-y 9 --square-length-mm 25 --marker-length-mm 19 `
       --output ../dataset/calib_data/iphone_intrinsics.json

2. Use Record3D's own per-frame reported intrinsics directly (ARKit
   focus-corrected calibration; no board needed -- recommended for the
   built-in iPhone camera since Apple already calibrates it in software):
   python -m camera_utils.iphone_intrinsic_calibration `
       --use-reported-intrinsics `
       --output ../dataset/calib_data/iphone_intrinsics.json
"""
import argparse
import sys
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_utils.calibration_io import save_intrinsics
from camera_utils.charuco_board import detect_charuco, draw_charuco_detection, make_charuco_board
from camera_utils.intrinsic_calibration import calibrate_from_images, imwrite_unicode
from camera_utils.iphone_connection import IPhoneCamera

DEFAULT_CALIB_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "calib_data"
IPHONE_INTRINSICS_FILENAME = "iphone_intrinsics.json"


def capture_from_iphone(dev_idx, board, detector, min_corners=6, capture_rotate90=0):
    print("Live capture: press SPACE to capture a frame when the ChArUco board "
          "is highlighted, ESC or Q to finish and calibrate.")

    frames = []
    with IPhoneCamera(dev_idx=dev_idx, capture_rotate90=capture_rotate90) as cam:
        try:
            while True:
                result = cam.get_latest_frame(timeout=2.0)
                if result is None:
                    status = "reconnecting..." if not cam.is_connected else "no frame (timeout)"
                    print(f"Waiting for iPhone ({status}).")
                    continue
                rgb, _depth, _intrinsic_mat, _pose = result
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                charuco_corners, charuco_ids = detect_charuco(detector, gray, min_corners=min_corners)
                display = frame.copy()
                draw_charuco_detection(display, charuco_corners, charuco_ids)
                n_corners = 0 if charuco_ids is None else len(charuco_ids)
                cv2.putText(display, f"captured: {len(frames)}  (corners this frame: {n_corners})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("iphone intrinsic calibration - SPACE=capture, Q=done", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and charuco_corners is not None:
                    frames.append(frame.copy())
                    print(f"  captured frame {len(frames)} ({n_corners} corners)")
                elif key in (27, ord("q")):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
    cv2.destroyAllWindows()

    return frames


def average_reported_intrinsics(dev_idx, num_samples, capture_rotate90=0):
    """Average Record3D's own per-frame reported intrinsic matrix over num_samples frames."""
    import numpy as np

    print(f"Reading {num_samples} frames of Record3D-reported intrinsics...")
    Ks = []
    image_size = None
    with IPhoneCamera(dev_idx=dev_idx, capture_rotate90=capture_rotate90) as cam:
        while len(Ks) < num_samples:
            result = cam.get_latest_frame(timeout=2.0)
            if result is None:
                status = "reconnecting..." if not cam.is_connected else "no frame (timeout)"
                print(f"Waiting for iPhone ({status}).")
                continue
            rgb, _depth, intrinsic_mat, _pose = result
            if image_size is None:
                image_size = (rgb.shape[1], rgb.shape[0])  # (width, height)
            Ks.append(np.asarray(intrinsic_mat, dtype=np.float64))

    K = np.mean(Ks, axis=0)
    dist = np.zeros(5, dtype=np.float64)  # Record3D reports an already-undistorted pinhole model
    return K, dist, image_size


def run_iphone_intrinsic_calibration(use_reported_intrinsics, dev_idx, num_samples,
                                      squares_x, squares_y, square_length_mm, marker_length_mm,
                                      aruco_dict, min_corners, capture_rotate90, output):
    """Shared entry point used by both this module's CLI and the
    calibrate_camera app. Returns (K, dist, image_size, reprojection_error)."""
    if use_reported_intrinsics:
        K, dist, image_size = average_reported_intrinsics(
            dev_idx, num_samples, capture_rotate90=capture_rotate90)
        err = None
    else:
        board, detector = make_charuco_board(
            squares_x, squares_y,
            square_length_mm / 1000.0, marker_length_mm / 1000.0,
            aruco_dict)
        frames = capture_from_iphone(
            dev_idx, board, detector, min_corners=min_corners, capture_rotate90=capture_rotate90)
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

    print(f"K =\n{K}")
    print(f"dist = {dist.ravel()}")
    if err is not None:
        print(f"Reprojection error: {err:.4f} px")

    save_intrinsics(output, K, dist, image_size, reprojection_error=err)
    print(f"Saved intrinsics to {output}")
    return K, dist, image_size, err


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-idx", type=int, default=0,
                         help="Index into Record3D's connected-device list (see IPhoneCamera.list_devices).")
    parser.add_argument("--capture-rotate90", type=int, default=0, choices=(0, 90, 180, 270),
                         help="Rotate the working frame at the source before it's used at all -- "
                              "see iphone_connection.py's module docstring. Must match what you "
                              "pass to iphone_extrinsic_calibration.py afterward -- this K is only "
                              "valid for that same rotation.")
    parser.add_argument("--use-reported-intrinsics", action="store_true",
                         help="Skip the board and just average Record3D's own per-frame reported "
                              "intrinsic matrix instead.")
    parser.add_argument("--num-samples", type=int, default=60,
                         help="Number of frames to average when --use-reported-intrinsics is set.")
    parser.add_argument("--squares-x", type=int, default=7,
                         help="Number of full checkerboard squares along the board's width. "
                              "Ignored if --use-reported-intrinsics is set.")
    parser.add_argument("--squares-y", type=int, default=9,
                         help="Number of full checkerboard squares along the board's height. "
                              "Ignored if --use-reported-intrinsics is set.")
    parser.add_argument("--square-length-mm", type=float, default=25.0,
                         help="Ignored if --use-reported-intrinsics is set.")
    parser.add_argument("--marker-length-mm", type=float, default=19.0,
                         help="Must be smaller than --square-length-mm. Ignored if "
                              "--use-reported-intrinsics is set.")
    parser.add_argument("--aruco-dict", type=str, default="DICT_5X5_50")
    parser.add_argument("--min-corners", type=int, default=6,
                         help="Minimum ChArUco corners required to accept a view.")
    parser.add_argument("--output", type=str,
                         default=str(DEFAULT_CALIB_DIR / IPHONE_INTRINSICS_FILENAME),
                         help="Where to write the intrinsics JSON file. Defaults to "
                              f"{DEFAULT_CALIB_DIR / IPHONE_INTRINSICS_FILENAME}.")
    args = parser.parse_args()

    try:
        run_iphone_intrinsic_calibration(
            args.use_reported_intrinsics, args.dev_idx, args.num_samples,
            args.squares_x, args.squares_y, args.square_length_mm, args.marker_length_mm,
            args.aruco_dict, args.min_corners, args.capture_rotate90, args.output)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
