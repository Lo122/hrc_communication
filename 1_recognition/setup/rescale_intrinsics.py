"""Convert an intrinsics calibration to a different frame resolution.

K is only valid at the resolution it was solved for. When frames cannot be
captured at that size -- a camera that refuses the mode, a recording made at
another one -- the calibration has to be converted rather than ignored, or
fx, fy, cx and cy are all out by the ratio between the two sizes and every
world position the pipeline reports is wrong by that factor.

This writes a NEW intrinsics file with the rescaled K and the new image_size
recorded in it, so FrameSource's resolution check passes against the new file
instead of being silenced.

    uv run python 1_recognition/setup/rescale_intrinsics.py `
        --input  1_recognition/calib_data/intrinsics_3840x2160_obs.json `
        --size   640x360 `
        --output 1_recognition/calib_data/intrinsics_640x360.json

Then point a run at it: run_recognition.py --intrinsics-file intrinsics_640x360.json

Converting assumes the source RESIZES the image. A camera that changes
resolution by cropping its sensor keeps fx and fy unchanged, so the result
would be wrong; scale_intrinsics warns when the aspect ratio changes, which is
the readable symptom. Recalibrating at the target resolution is always the
better answer when it is available.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from camera_utils.calibration_io import load_intrinsics, save_intrinsics, scale_intrinsics
from logging_setup import configure_logging, get_logger
from video_source import parse_size

logger = get_logger(__name__)


def rescale_file(input_path, new_size, output_path, *, force=False):
    """Read an intrinsics JSON, rescale K to new_size, write it to
    output_path. Returns (K_new, new_size)."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.resolve() == input_path.resolve() and not force:
        raise ValueError(
            f"Refusing to overwrite {input_path}: it is the only record of what was "
            "actually measured, and a rescale cannot be undone exactly. Write to a new "
            "file, or pass --force.")

    K, dist, image_size = load_intrinsics(input_path)
    if tuple(image_size) == tuple(new_size):
        logger.info("%s is already calibrated at %dx%d; copying unchanged.",
                    input_path, *new_size)

    K_new, new_size = scale_intrinsics(K, image_size, new_size)

    # dist is carried through unchanged -- the coefficients act on normalized
    # coordinates, which a scaled K leaves alone. reprojection_error belongs to the
    # original solve and is not re-measurable here, so it travels with a note.
    save_intrinsics(output_path, K_new, dist, new_size,
                    reprojection_error=None)

    logger.info("Rescaled %s (%dx%d) -> %s (%dx%d)",
                input_path, image_size[0], image_size[1], output_path, *new_size)
    logger.info("  fx %.2f -> %.2f   fy %.2f -> %.2f",
                K[0, 0], K_new[0, 0], K[1, 1], K_new[1, 1])
    logger.info("  cx %.2f -> %.2f   cy %.2f -> %.2f",
                K[0, 2], K_new[0, 2], K[1, 2], K_new[1, 2])
    logger.info("  reprojection_error dropped: it belongs to the original %dx%d solve.",
                image_size[0], image_size[1])
    return K_new, new_size


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rescale a camera intrinsics calibration to a different resolution.")
    parser.add_argument("--input", required=True,
                        help="Existing intrinsics JSON (from setup/calibrate_camera.py).")
    parser.add_argument("--size", required=True, metavar="WIDTHxHEIGHT", type=parse_size,
                        help="Resolution to convert the calibration to, e.g. 1280x720.")
    parser.add_argument("--output", required=True,
                        help="Where to write the converted intrinsics.")
    parser.add_argument("--force", action="store_true",
                        help="Allow --output to be --input. The original measurement is "
                             "then gone, so prefer a new file.")
    args = parser.parse_args(argv)

    configure_logging("rescale_intrinsics")
    try:
        rescale_file(args.input, args.size, args.output, force=args.force)
    except (ValueError, OSError, KeyError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
