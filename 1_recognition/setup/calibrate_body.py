"""Capture one subject's body calibration: hold a T-pose, then turn slowly
through 180deg while holding it. Writes the JSON that VisionConfig's
body_calibration_file points at.

Produces two things (see 1_recognition/src/skeleton_utils/body_calibration.py
for the reasoning behind both, including why the turn is sampled throughout
rather than at its endpoints):

  - per-bone target lengths, which seed BoneLengthConstraintFilter so it
    starts converged instead of being seeded by whatever the first live
    frame happened to measure, then defending that guess via its ratio gate;
  - a metric stature, replacing VisionConfig.user_height_m's 1.70 m
    assumption. MetricDepthEstimator's depth scales linearly with it, so this
    is a proportional correction to every absolute world position.

THE TWO STATURES. If you know the subject's height, MEASURE IT AND PASS
--measured-height: a tape is ~+/-0.6%, while the ground-plane estimate carries
NOSE_HEIGHT_RATIO's couple-of-percent error and needs calibrated extrinsics
with a correct --ground-z. Both are stored; the tape wins. Keeping both is the
point -- their difference is this subject's true nose-height ratio, and a
consistent bias across subjects means that constant needs retuning for your
population. Without extrinsics the bone half still works and the ground-plane
stature is simply left null.

Usage (from the repo root):
    uv run python 1_recognition/setup/calibrate_body.py --device cuda:0 --camera `
        --subject uid-08 --measured-height 1.62 `
        --camera-index 6 `
        --intrinsics 1_recognition/calib_data/intrinsics_3840x2160_obs.json `
        --extrinsics 1_recognition/calib_data/extrinsics_3840x2160_obs.json `
        --output 1_recognition/calib_data/body_uid-09.json

    # --intrinsics/--extrinsics default to calib_data/intrinsics.json and
    # extrinsics.json. Pass the pair this camera was actually calibrated with; the
    # capture then defaults to the resolution that intrinsics file was solved at,
    # since K does not transfer between resolutions (setup/rescale_intrinsics.py
    # converts one if you need to capture at a different size).


    uv run python 1_recognition/setup/calibrate_body.py --device cuda:0 --fp16 `
        --source "path/to/a/recorded/tpose.mp4" --subject uid-02

    # add a height to a calibration you already captured -- no camera, no T-pose
    uv run python 1_recognition/setup/calibrate_body.py `
        --update 1_recognition/calib_data/body_uid-01.json --measured-height 1.82

    # then point the live run at it
    uv run python run_recognition.py --camera --body-calibration `
        1_recognition/calib_data/body_uid-01.json

Hold the T-pose until the on-screen counter reaches its target, keeping arms
straight and level, then rotate on the spot. Frames that fail validation are
rejected with the reason shown, so a pose the calibrator will not accept is
visible immediately rather than silently producing a bad calibration.
Press q to finish early, r to restart the capture.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_RECOGNITION_DIR = Path(__file__).resolve().parents[1]
# This file's own directory too, so video_source resolves when it is imported
# as a module rather than run as a script.
for _p in (_RECOGNITION_DIR, _RECOGNITION_DIR / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from skeleton3d_pipeline import RealtimeSkeleton3DPipeline, CALIB_DATA_DIR  # noqa: E402
from skeleton_utils.body_calibration import (  # noqa: E402
    NOSE_HEIGHT_RATIO, BodyCalibration, TPoseCalibrator, validate_stature_m)
from vision_model.vision_config import VisionConfig  # noqa: E402
from logging_setup import configure_logging, get_logger
from video_source import (  # noqa: E402
    DEFAULT_PREVIEW_WIDTH, add_capture_args, forget_preview_windows, open_camera,
    preview_scale, resolve_capture_size, show_preview, verify_frame_size)

logger = get_logger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=None,
                        help="Video file or stream URL. Omit with --camera for a live camera.")
    parser.add_argument("--camera", action="store_true", help="Use the realtime camera.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--subject", default="",
                        help="Subject id recorded in the calibration, e.g. uid-01. Also the "
                             "default output filename (body_<subject>.json).")
    parser.add_argument("--output", default=None,
                        help=f"Output JSON path. Default: {CALIB_DATA_DIR}/body_<subject>.json")
    parser.add_argument("--intrinsics", default=None, metavar="JSON",
                        help=f"Intrinsics file for this camera. Default: "
                             f"{CALIB_DATA_DIR}/intrinsics.json. Pass the one solved at the "
                             "resolution this capture runs at -- K does not transfer between "
                             "resolutions (convert with setup/rescale_intrinsics.py instead).")
    parser.add_argument("--extrinsics", default=None, metavar="JSON",
                        help=f"Extrinsics file for this camera. Default: "
                             f"{CALIB_DATA_DIR}/extrinsics.json. Only the STATURE half needs "
                             "these -- it back-projects ankle and nose pixels onto the ground "
                             "plane -- so bone lengths still calibrate without them.")
    parser.add_argument("--device", default="cpu", help="'cpu', 'cuda:0', ...")
    parser.add_argument("--fp16", action="store_true",
                        help="fp16 MotionBERT (CUDA only) -- see MotionBERTStreamingLifter.")
    parser.add_argument("--min-frames", type=int, default=60,
                        help="Accepted T-pose frames required before a calibration is written. "
                             "Default 60 -- ~6 s at the 10 Hz this pipeline runs live, enough to "
                             "cover a slow turn. result() refuses below this, because a "
                             "calibration from a handful of frames is worse than none: the bone "
                             "filter's ratio gate will defend whatever it is given.")
    parser.add_argument("--measured-height", type=float, default=None, metavar="METRES",
                        help="The subject's real height, measured with a tape, in METRES "
                             "(1.82, not 182). Stored as measured_stature_m and used in "
                             "preference to the ground-plane estimate, which carries "
                             "NOSE_HEIGHT_RATIO's couple-of-percent error and needs "
                             "extrinsics. Both are kept, so the difference tells you how "
                             "biased that constant is for your subjects.")
    parser.add_argument("--update", default=None, metavar="CALIBRATION_JSON",
                        help="Write --measured-height into an existing calibration file and "
                             "exit -- no camera, no T-pose. For adding a height you measured "
                             "after the fact, without redoing the capture.")
    # Live capture only. Defaults to the resolution the intrinsics were solved at,
    # which is what the ground-plane stature needs -- a --source clip cannot be
    # asked for a mode and is only checked.
    add_capture_args(parser)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def update_measured_height(path, measured_height_m):
    """Set measured_stature_m on an existing calibration and save it back."""
    calibration = BodyCalibration.load(path)
    calibration.measured_stature_m = validate_stature_m(
        measured_height_m, source="--measured-height")
    calibration.save(path)
    logger.info("Set measured_stature_m = %.3f m in %s",
                calibration.measured_stature_m, path)
    _log_stature_summary(calibration)
    return calibration


def _log_stature_summary(calibration):
    """Report which height will be used, and -- when both are present -- what
    the tape measurement says about NOSE_HEIGHT_RATIO's bias."""
    stature = calibration.effective_stature_m
    if not stature:
        logger.info("  stature          : not measured (no tape height, no extrinsics)")
        return
    logger.info("  stature in use   : %.3f m (%s)", stature, calibration.stature_source)

    ratio = calibration.implied_nose_height_ratio
    if ratio is None or not calibration.stature_m:
        return
    logger.info("  ground-plane estimate: %.3f m (%+.1f%% vs the tape)",
                calibration.stature_m,
                100.0 * (calibration.stature_m / stature - 1.0))
    logger.info("  implied nose-height ratio: %.3f (assumed %.2f). A consistent bias "
                "across several subjects means NOSE_HEIGHT_RATIO is wrong for this "
                "population and is worth changing.", ratio, NOSE_HEIGHT_RATIO)


def main():
    configure_logging("calibrate_body")
    args = parse_args()

    # Validate before opening a camera: a units mistake here would otherwise only
    # surface after the whole T-pose capture.
    if args.measured_height is not None:
        try:
            args.measured_height = validate_stature_m(
                args.measured_height, source="--measured-height")
        except ValueError as exc:
            raise SystemExit(str(exc))

    if args.update:
        if args.measured_height is None:
            raise SystemExit("--update needs --measured-height: it exists only to write "
                             "that value into an existing calibration.")
        update_measured_height(args.update, args.measured_height)
        return

    source = args.camera_index if args.camera else args.source
    if source is None:
        raise SystemExit("Pass --camera or --source.")

    # use_bone_length_filter=False on purpose: calibration must measure the
    # RAW lifter output. With the filter on, the lengths being sampled would
    # be the ones the filter is already rescaling toward its own running
    # estimate -- the calibration would partly measure itself.
    config = VisionConfig(device=args.device, motionbert_fp16=args.fp16,
                          use_bone_length_filter=False)
    # The pipeline resolves each file as camera.calib_dir / <file>, and joining an
    # ABSOLUTE path discards the left side -- so resolving here overrides calib_dir
    # per file. That matters: the two can then live in different directories, and
    # whichever flag was not passed still falls back to calib_dir's default name.
    for flag, value in (("--intrinsics", args.intrinsics), ("--extrinsics", args.extrinsics)):
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"{flag} {value!r} does not exist (looked at {path}).")
        field = "intrinsics_file" if flag == "--intrinsics" else "extrinsics_file"
        setattr(config.camera, field, str(path))
    pipeline = RealtimeSkeleton3DPipeline(config)

    calibrator = TPoseCalibrator(
        K=pipeline.K,
        T_world_from_camera=pipeline.T_world_from_camera,
        ground_z=pipeline.ground_z,
        min_frames=args.min_frames,
        subject_id=args.subject,
    )
    if pipeline.T_world_from_camera is None:
        logger.warning("No extrinsics -- bone lengths will be calibrated, stature will not. "
                       "Run setup/calibrate_camera.py's extrinsic step to enable the "
                       "height half.")

    # Frames here are measured against pipeline.K -- the stature half back-projects
    # ankle and nose pixels through it -- so this capture has to run at the
    # resolution K was solved for. A live camera can be asked; a recorded clip
    # plays at whatever it was written at and can only be checked.
    if args.camera:
        capture_size = resolve_capture_size(
            args.capture_width, args.capture_height, pipeline.image_size)
        width, height = capture_size if capture_size else (None, None)
        cap, _actual = open_camera(source, width, height, args.backend)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"Could not open {source!r}")

    print("Hold a T-pose: arms straight out, level with the shoulders. "
          "Then turn slowly through 180deg, keeping the pose. [q] finish  [r] restart")

    size_verified = False
    t_start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.info("End of stream.")
                break

            if not size_verified:
                verify_frame_size(frame, pipeline.image_size, source_label=str(source))
                size_verified = True

            result = pipeline.process(frame, timestamp=time.time() - t_start)
            status = {"accepted": 0, "last_reason": "no person detected",
                      "fraction": 0.0, "height_samples": 0, "ready": False}
            if result["root_relative"] is not None:
                status = calibrator.update(
                    result["root_relative"], result["keypoints_2d"], result["keypoints_conf"])

            if not args.no_display:
                _draw(frame, status, args.min_frames,
                      preview_scale(frame.shape[1], args.preview_width))
                show_preview("body calibration -- T-pose + 180deg turn", frame,
                             args.preview_width)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    calibrator.reset()
                    print("restarted")
            elif status["ready"]:
                break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
            forget_preview_windows()

    calibration = calibrator.result()
    if calibration is None:
        progress = calibrator.progress()
        raise SystemExit(
            f"Not enough valid T-pose frames: {progress['accepted']}/{args.min_frames} "
            f"(last rejection: {progress['last_reason']}). Nothing written -- a partial "
            "calibration would be actively harmful, see TPoseCalibrator.result().")

    if args.measured_height is not None:
        calibration.measured_stature_m = args.measured_height

    name = args.subject or "default"
    out_path = Path(args.output) if args.output else Path(CALIB_DATA_DIR) / f"body_{name}.json"
    calibration.save(out_path)

    # Logged, not printed: these numbers ARE the calibration's result, and "what stature
    # did we measure for uid-01 back in September" is exactly the question the log exists
    # to answer. They still reach the console, since INFO goes there too.
    logger.info("Wrote %s", out_path)
    logger.info("  bones calibrated : %d", len(calibration.bone_lengths))
    logger.info("  T-pose frames    : %d (rejected %d)",
                calibration.n_pose_frames, calibrator.progress()["rejected"])
    if calibration.stature_m:
        logger.info("  nose height      : %.3f m (%d frames)",
                    calibration.nose_height_m, calibration.n_height_frames)
        logger.info("  ground-plane stature: %.3f m", calibration.stature_m)
    _log_stature_summary(calibration)
    if calibration.measured_stature_m is None:
        logger.info("  No tape measurement. Add one any time with: "
                    "calibrate_body.py --update \"%s\" --measured-height <metres>", out_path)
    print(f"\nUse it:  --body-calibration \"{out_path}\"")


def _draw(frame, status, min_frames, s=1.0):
    """s scales the overlay so it stays readable after the preview downscale."""
    ready = status.get("ready")
    colour = (0, 200, 0) if ready else (0, 165, 255)
    thick = max(int(2 * s), 1)
    cv2.putText(frame, f"T-pose frames: {status['accepted']}/{min_frames}",
                (int(16 * s), int(34 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, colour, thick)
    cv2.putText(frame, f"height samples: {status['height_samples']}",
                (int(16 * s), int(64 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s,
                (220, 220, 220), max(int(1 * s), 1))
    cv2.putText(frame,
                "TURN SLOWLY - done, press q" if ready else status["last_reason"],
                (int(16 * s), int(92 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s,
                (0, 200, 0) if ready else (0, 0, 255), thick)
    width = frame.shape[1] - int(32 * s)
    top, bottom = int(104 * s), int(118 * s)
    cv2.rectangle(frame, (int(16 * s), top), (int(16 * s) + width, bottom), (70, 70, 70), -1)
    cv2.rectangle(frame, (int(16 * s), top),
                  (int(16 * s) + int(width * status["fraction"]), bottom), colour, -1)


if __name__ == "__main__":
    main()
