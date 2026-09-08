"""ChArUco-board intrinsic calibration (camera matrix K + distortion coeffs).

Two ways to get calibration images:

1. Live capture from a camera:
   python -m camera_utils.intrinsic_calibration `
       --camera-index 0 --squares-x 7 --squares-y 9 `
       --square-length-mm 25 --marker-length-mm 19 `
       --output ../dataset/calib_data/intrinsics.json
   Press SPACE to capture a frame once the board is detected, ESC/Q to stop
   capturing and run the calibration.

2. From a folder of already-captured images:
   python -m camera_utils.intrinsic_calibration `
       --images-dir path/to/imgs --squares-x 7 --squares-y 9 `
       --square-length-mm 25 --marker-length-mm 19 `
       --output ../dataset/calib_data/intrinsics.json

``--squares-x``/``--squares-y`` are the number of full checkerboard squares
along each side (including the black ones) -- must match the board used to
generate the print, i.e. generate_charuco_board.py's --squares-x/--squares-y.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_utils.calibration_io import save_intrinsics
from camera_utils.charuco_board import detect_charuco, draw_charuco_detection, make_charuco_board


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
            print(f"  [skip] fewer than {min_corners} ChArUco corners found in "
                  f"{Path(path).name}")
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

    print(f"Calibrating from {used} views ({skipped} skipped)...")
    reprojection_error, K, dist, _, _ = cv2.calibrateCamera(
        all_object_points, all_image_points, image_size, None, None)

    return K, dist, image_size, reprojection_error


def capture_from_camera(camera_index, detector, min_corners=6):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Could not open camera index {camera_index}")

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
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("intrinsic calibration - SPACE=capture, Q=done", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and charuco_corners is not None:
                frames.append(frame.copy())
                print(f"  captured frame {len(frames)} ({n_corners} corners)")
            elif key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return frames


def run_intrinsic_calibration(images_dir, camera_index, squares_x, squares_y,
                               square_length_mm, marker_length_mm, aruco_dict,
                               min_corners, output):
    """Shared entry point used by both this module's CLI and the
    calibrate_camera app. Returns (K, dist, image_size, reprojection_error)."""
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
        frames = capture_from_camera(camera_index, detector, min_corners=min_corners)
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

    print(f"Reprojection error: {err:.4f} px")
    print(f"K =\n{K}")
    print(f"dist = {dist.ravel()}")

    save_intrinsics(output, K, dist, image_size, reprojection_error=err)
    print(f"Saved intrinsics to {output}")
    return K, dist, image_size, err


def main():
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
    parser.add_argument("--output", type=str, required=True,
                         help="Where to write the intrinsics JSON file.")
    args = parser.parse_args()

    try:
        run_intrinsic_calibration(
            args.images_dir, args.camera_index, args.squares_x, args.squares_y,
            args.square_length_mm, args.marker_length_mm, args.aruco_dict,
            args.min_corners, args.output)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
