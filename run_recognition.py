"""Run realtime recognition in its own process and publish trigger events.

Reads frames from a camera, an iPhone (Record3D) or a recorded file, runs them
through RecognitionManager, and sends RECOGNITION_TRIGGER and
HUMAN_LOCATION_UPDATE events to the communication layer over UDP.

This repo's venv lives outside the project tree, so point uv at it first:
    $env:UV_PROJECT_ENVIRONMENT = "C:\\Users\\Owner\\.venvs\\hrc_communication"

Four modes:

  full pipeline       vision + LSTM step classification (the default)
  --fake-recognition  vision runs for real, step classification is typed in
                      -- no LSTM, so no norm-stats file needed
  --manual-trigger    no camera/video/model at all; type trigger events to
                      time how fast the communication layer reacts
  --realtime-playback a recorded file played against the wall clock, so it
                      drops frames the way a live camera does

Usage:
    uv run python run_recognition.py --camera
    uv run python run_recognition.py --video-source clip.mp4
    uv run python run_recognition.py --iphone --fake-recognition
    uv run python run_recognition.py --manual-trigger

    # latency experiment: same clip, frame by frame vs. against the clock
    uv run python run_recognition.py --video-source clip.mp4 `
        --log-dir results --run-name baseline --no-display
    uv run python run_recognition.py --video-source clip.mp4 --realtime-playback `
        --loop-hz 1000 --log-dir results --run-name realtime_1x --no-display

    uv run python run_recognition.py `
        --video-source 6 `
        --model-dir 1_recognition/best_model/3d_skeleton_01 `
        --intrinsics-file intrinsics_640x360_obs.json `
        --extrinsics-file extrinsics_3840x2160_obs.json `
        --body-calibration 1_recognition/calib_data/body_uid-08.json `
        --fp16 `
        --record 1_recognition/results/recognition_test/detection_test.mp4

    
    uv run python run_recognition.py `
            --iphone `
            --fake-recognition `
            --intrinsics-file iphone_intrinsics.json `
            --extrinsics-file iphone_extrinsics.json `
            --body-calibration 1_recognition/calib_data/body_uid-08.json `
            --iphone-rotate 270 `
            --fp16 `
            --loop-hz 10

            
    
    uv run python run_recognition.py `
            --iphone `
            --model-dir 1_recognition/best_model/3d_skeleton_01 `
            --intrinsics-file iphone_intrinsics.json `
            --extrinsics-file iphone_extrinsics.json `
            --body-calibration 1_recognition/calib_data/body_uid-08.json `
            --iphone-rotate 270 `
            --fp16 `
            --loop-hz 10
            
                

    uv run python run_recognition.py `
        --realtime-playback `
        --video-source "G:\\My Drive\\University of Stuttgart\\ITECH_Thesis\\Videos\\raw\\cam-05\\video__cam-05_uid-11_take-01.mp4" `
        --model-dir 1_recognition/best_model/3d_skeleton_01 `
        --intrinsics-file iphone_intrinsics.json `
        --extrinsics-file iphone_extrinsics.json `
        --body-calibration 1_recognition/calib_data/body_uid-08.json `
        --fp16 


    # live line graph of every step's softmax probability and the progress value
    uv run python run_recognition.py --loop-hz 10 --show-probabilities
    uv run python run_recognition.py --iphone --iphone-rotate 270 --show-probabilities `
        --extrinsics-file iphone_extrinsics_synthetic.json

In both trigger-typing modes the prompt takes:
    step_id[,piece_id[,round_id[,progress]]]

A live camera is opened at the resolution its intrinsics were calibrated at,
since K only holds at that size and nothing negotiates it for us -- Windows
settles on 640x480 however capable the device is. --capture-width/--capture-height
override that; 1_recognition/setup/video_source.py --list shows what each camera
index can deliver.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for layer in ["0_core", "1_recognition"]:
    sys.path.insert(0, str(ROOT / layer))
# The recognition layer's logging lives in its src/, alongside the rest of its internals.
sys.path.insert(0, str(ROOT / "1_recognition" / "src"))

import config
from event_transport import UDPEventSender
from events import Event, EventType
from logging_setup import configure_logging, get_logger
from recognition_manager import DEFAULT_MODEL_DIR, RecognitionManager
from step_probability_plot import StepProbabilityPlot
from trigger_manager import TriggerManager
from vision_model.vision_config import VisionConfig

# iPhone/Record3D defaults. The rotation and the calibration files must agree: these
# extrinsics were calibrated at capture_rotate90=90, and their 960x720 K only matches
# frames rotated the same way. Changing one without the other gives a wrong world frame.
IPHONE_SOURCE = "iphone"
IPHONE_CAPTURE_ROTATE90 = 90
IPHONE_INTRINSICS_FILE = "iphone_intrinsics.json"
IPHONE_EXTRINSICS_FILE = "iphone_extrinsics.json"

DEFAULT_MANUAL_PIECE_ID = 1
DEFAULT_MANUAL_ROUND_ID = 0
DEFAULT_MANUAL_PROGRESS = 1.0

logger = get_logger(__name__)


REALTIME_CAMERA_INDEX = 0
LOOP_HZ = 20.0  # target recognition loop rate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recognition and publish trigger events.")
    parser.add_argument("--camera", action="store_true", help="Use realtime camera input.")
    parser.add_argument("--iphone", action="store_true",
                         help="Capture from an iPhone running Record3D over USB ('USB Streaming' "
                              "enabled in the app). Also selects the iPhone calibration "
                              f"({IPHONE_INTRINSICS_FILE}/{IPHONE_EXTRINSICS_FILE}) and the "
                              f"capture rotation ({IPHONE_CAPTURE_ROTATE90} deg) they were "
                              "calibrated with. Requires the 'iphone' extra: uv sync --extra iphone")
    parser.add_argument("--iphone-dev-idx", type=int, default=0,
                         help="Record3D device index when more than one device is connected.")
    parser.add_argument("--iphone-rotate", type=int, default=IPHONE_CAPTURE_ROTATE90,
                         choices=[0, 90, 180, 270],
                         help="Override the iPhone capture rotation. Only change this if you "
                              "recalibrated intrinsics AND extrinsics at the new rotation -- "
                              "otherwise K and the world frame will be wrong.")
    parser.add_argument("--extrinsics-file", default=None,
                         help="Extrinsics file name inside calib_data/, overriding the default for "
                              "the chosen source. Use iphone_extrinsics_synthetic.json to run "
                              "without an ArUco calibration (idealised level camera -- fine for "
                              "checking the pipeline, not for real measurements).")
    parser.add_argument("--intrinsics-file", default=None,
                         help="Intrinsics file name inside calib_data/, overriding the default.")
    parser.add_argument("--calib-dir", default=None,
                         help="Directory holding the calibration files --intrinsics-file and "
                              "--extrinsics-file name. Defaults to VisionConfig's own default "
                              "(1_recognition/calib_data).")
    parser.add_argument("--capture-width", type=int, default=None,
                         help="Live sources only: request this capture width. By default the "
                              "resolution the intrinsics were calibrated at is requested, which "
                              "is what you want -- K only holds at that size. Set this only to "
                              "capture at a size the calibration was NOT solved for; convert the "
                              "calibration with 1_recognition/setup/rescale_intrinsics.py instead "
                              "of living with the mismatch. Use with --capture-height.")
    parser.add_argument("--capture-height", type=int, default=None,
                         help="Live sources only: request this capture height. Use together "
                              "with --capture-width.")
    parser.add_argument("--capture-backend", choices=("auto", "dshow", "msmf", "any"),
                         default=None,
                         help="Capture backend for a live camera (default auto: DirectShow on "
                              "Windows, which is markedly more reliable for virtual cameras). "
                              "Run 1_recognition/setup/video_source.py --list to see what each "
                              "backend finds.")
    parser.add_argument("--video-source", default=None, help="Video path or stream URL. Defaults to config.test_vid_path.")
    parser.add_argument("--realtime-playback", action="store_true",
                         help="Recorded video only: play the file against the wall clock like a "
                              "live camera instead of feeding the model every frame. Frames that "
                              "'arrive' while the model is busy are dropped, so a recorded run "
                              "sees the same latency-induced frame loss a live run does -- use it "
                              "to measure how much of the offline/live accuracy gap comes from "
                              "processing time. Pair with a high --loop-hz so the video clock, "
                              "not the loop timer, is what paces the run.")
    parser.add_argument("--log-dir", default=None,
                         help="Write per-frame results to <LOG_DIR>/<run-name>/ (frames.csv, "
                              "events.csv, run.json) for offline analysis -- see "
                              "1_recognition/eval/run_logger.py, and analyse_runs.py for the plots. "
                              "Default: no logging.")
    parser.add_argument("--run-name", default=None,
                         help="Name of the run directory under --log-dir. Defaults to a "
                              "timestamp, so repeated runs never overwrite each other. Give "
                              "runs you intend to compare meaningful names (e.g. 'baseline', "
                              "'realtime_1x', 'realtime_2x').")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Stop after this many PROCESSED frames. Useful for keeping "
                              "compared runs the same length when the source is a live camera.")
    parser.add_argument("--playback-speed", type=float, default=1.0,
                         help="With --realtime-playback: how fast the video clock runs (default "
                              "1.0 = the recording's own rate). 2.0 drops twice as many frames "
                              "per inference, i.e. simulates a model half as fast, without "
                              "touching the frame timestamps the features are fit against.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing config.json and best_model.pth.")
    parser.add_argument("--host", default=config.EVENT_TRANSPORT_HOST, help="Communication event receiver host.")
    parser.add_argument("--port", type=int, default=config.EVENT_TRANSPORT_PORT, help="Communication event receiver port.")
    parser.add_argument("--no-display", action="store_true", help="Disable the recognition video preview window.")
    parser.add_argument("--render-world-skeleton", action="store_true",
                         help="Draw the 3D panel in WORLD coordinates (posture fused with the "
                              "depth estimate, expressed in the calibration target's frame) "
                              "instead of the default: the pelvis-centred camera-frame posture "
                              "that is actually fed to the model. The world skeleton carries the "
                              "extrinsics' rotation, so it does not line up with the video's own "
                              "axes -- absolute position is already shown by the World XYZ "
                              "overlay, so this is only useful when checking the extrinsics "
                              "themselves.")
    parser.add_argument("--record", default=None, metavar="PATH",
                         help="Also write the render screen -- the 2D overlay beside the "
                              "3D panel, with the model readout burned in, i.e. exactly "
                              "what the preview window shows -- to this .mp4. Works with "
                              "--no-display for a headless capture. The scrolling debug "
                              "plot is a separate window and is not recorded.")
    parser.add_argument("--record-fps", type=float, default=None,
                         help="Frame rate stamped into the --record file. Defaults to "
                              "--loop-hz, which is the rate frames are actually rendered "
                              "at, so the recording plays back at wall-clock speed.")
    parser.add_argument("--show-probabilities", action="store_true",
                         help="Show a live line-graph preview window of each task step's "
                              "softmax probability and the progress value over time -- see "
                              "step_probability_plot.py. Independent of --no-display.")
    parser.add_argument("--probability-history-seconds", type=float, default=12.0,
                         help="--show-probabilities only. How many seconds of history the "
                              "line graph keeps on screen.")
    parser.add_argument("--body-calibration", default=None,
                        help="Path to a body_calibration.py JSON from 1_recognition/setup/calibrate_body.py "
                             "(T-pose + 180deg turn). Seeds BoneLengthConstraintFilter's per-bone "
                             "targets so it starts converged rather than being seeded by the first "
                             "live frame, and replaces the assumed 1.70 m user height, which scales "
                             "every absolute world position proportionally. The height comes from "
                             "the file's measured_stature_m (a tape measurement, preferred) or "
                             "failing that stature_m (the pipeline's own ground-plane estimate, "
                             "which needs extrinsics). Add a tape height to an existing file with "
                             "calibrate_body.py --update <file> --measured-height <metres>.")
    parser.add_argument("--no-bone-filter", action="store_true",
                        help="Disable BoneLengthConstraintFilter live. NOT recommended: "
                             "generate_lstm_training_data.py applies it to every TRAINING frame, so "
                             "turning it off feeds the model a systematically noisier skeleton than "
                             "it was trained on.")
    parser.add_argument("--fp16", action="store_true",
                        help="Run MotionBERT's 2D->3D lift in fp16. Needs a CUDA device (ignored with "
                             "a warning on CPU). Measured 31.0 -> 17.9 ms per lift at clip_len=81 on "
                             "an RTX 3060 Laptop -- the largest single win available in the live loop "
                             "without changing what the model sees, since it does NOT require "
                             "regenerating training data (see skeleton_utils/motionbert_lifter.py's "
                             "MotionBERTStreamingLifter docstring for the fp32-vs-fp16 A/B).")
    parser.add_argument("--fake-recognition", action="store_true",
                         help="Run the real vision pipeline (YOLO 2D pose + MotionBERT 3D lift) "
                              "with live skeleton display, plots and HUMAN_LOCATION_UPDATE "
                              "streaming, but skip the LSTM step classifier -- so no norm-stats "
                              "file is needed. RECOGNITION_TRIGGER events are typed in instead, "
                              "same syntax as --manual-trigger.")
    parser.add_argument("--manual-trigger", action="store_true",
                         help="Skip the camera/video/model pipeline entirely. Instead, prompt on the "
                              "terminal for step_id[,piece_id[,round_id[,progress]]] and send that "
                              "RECOGNITION_TRIGGER event over the real UDP path on Enter -- for timing "
                              "how fast the communication layer reacts without needing real model output.")
    parser.add_argument("--loop-hz", type=float, default=LOOP_HZ,
                         help="Target recognition loop rate (Hz). Each iteration sleeps only "
                              "the remainder of its tick, so the cadence holds regardless of "
                              "how long update() takes.")
    return parser.parse_args()


def _recognition_source(args: argparse.Namespace):
    if args.iphone:
        return IPHONE_SOURCE
    if args.camera:
        return REALTIME_CAMERA_INDEX
    return args.video_source or config.test_vid_path


def _build_vision_config(args: argparse.Namespace) -> VisionConfig | None:
    """Translate the vision-related CLI flags into a VisionConfig.

    Returns None when no flag overrides a default, letting RecognitionManager
    build its own config.
    """
    if not (args.iphone or args.extrinsics_file or args.intrinsics_file or args.calib_dir
            or args.realtime_playback or args.fp16
            or args.body_calibration or args.no_bone_filter
            or args.capture_width or args.capture_height or args.capture_backend):
        return None

    vision_config = VisionConfig()
    vision_config.motionbert_fp16 = args.fp16
    vision_config.use_bone_length_filter = not args.no_bone_filter
    vision_config.body_calibration_file = args.body_calibration
    vision_config.camera.realtime_playback = args.realtime_playback
    vision_config.camera.playback_speed = args.playback_speed
    # Left as None, FrameSource requests the calibrated resolution instead.
    vision_config.camera.capture_width = args.capture_width
    vision_config.camera.capture_height = args.capture_height
    vision_config.camera.capture_backend = args.capture_backend
    if args.iphone:
        vision_config.camera.dev_idx = args.iphone_dev_idx
        vision_config.camera.capture_rotate90 = args.iphone_rotate
        vision_config.camera.intrinsics_file = IPHONE_INTRINSICS_FILE
        vision_config.camera.extrinsics_file = IPHONE_EXTRINSICS_FILE

    # Explicit overrides win over the per-source defaults above.
    if args.calib_dir is not None:
        vision_config.camera.calib_dir = args.calib_dir
    if args.intrinsics_file:
        vision_config.camera.intrinsics_file = args.intrinsics_file
    if args.extrinsics_file:
        vision_config.camera.extrinsics_file = args.extrinsics_file
    return vision_config


def _build_run_logger(args: argparse.Namespace, source):
    """Open per-frame CSV logging, or return None unless --log-dir is given.

    The metadata recorded alongside -- source and playback settings above all --
    is what makes two runs comparable afterwards.
    """
    if not args.log_dir:
        return None

    # eval/ is offline tooling, so it joins sys.path here next to its only import
    # rather than at module load: a normal run never touches it.
    eval_dir = str(ROOT / "1_recognition" / "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from run_logger import RunLogger

    return RunLogger(args.log_dir, args.run_name, metadata={
        "video_source": str(source),
        "realtime_playback": args.realtime_playback,
        "playback_speed": args.playback_speed,
        "loop_hz": args.loop_hz,
        "model_dir": args.model_dir,
        "step_model_enabled": not args.fake_recognition,
        "display": not args.no_display,
    })


def _parse_manual_trigger_line(line: str) -> tuple[int, int, int, float] | None:
    """Parse "step_id[,piece_id[,round_id[,progress]]]" typed at the prompt.

    Missing fields fall back to DEFAULT_MANUAL_*. Returns None for a blank or
    unparseable line, so the caller re-prompts instead of crashing the loop.
    """
    parts = [p.strip() for p in line.split(",")]
    if not parts or not parts[0]:
        return None
    try:
        step_id = int(parts[0])
        piece_id = int(parts[1]) if len(parts) > 1 and parts[1] else DEFAULT_MANUAL_PIECE_ID
        round_id = int(parts[2]) if len(parts) > 2 and parts[2] else DEFAULT_MANUAL_ROUND_ID
        progress = float(parts[3]) if len(parts) > 3 and parts[3] else DEFAULT_MANUAL_PROGRESS
    except ValueError:
        print(f"Could not parse '{line}' as step_id[,piece_id[,round_id[,progress]]] -- try again.")
        return None
    return step_id, piece_id, round_id, progress


def _configured_steps_text() -> str:
    return ", ".join(str(step) for step in sorted(config.TRIGGER_RULES))


def _send_manual_trigger(sender: UDPEventSender, line: str) -> None:
    """Parse one typed line and, if valid, publish it as a RECOGNITION_TRIGGER."""
    parsed = _parse_manual_trigger_line(line)
    if parsed is None:
        return
    step_id, piece_id, round_id, progress = parsed
    if step_id not in config.TRIGGER_RULES:
        print(f"Step {step_id} has no recognition trigger configured. "
              f"Configured human steps: {_configured_steps_text()}.")
        return

    event = Event(
        event_type=EventType.RECOGNITION_TRIGGER,
        source="manual_recognition",
        payload={
            "step_id": step_id,
            "piece_id": piece_id,
            "round_id": round_id,
            "progress": progress,
        },
    )
    send_time = time.time()
    sender.send(event)
    print(f"[manual trigger] sent step_id={step_id} piece_id={piece_id} "
          f"round_id={round_id} progress={progress} at t={send_time:.6f}", flush=True)


def _print_manual_trigger_help(host: str, port: int) -> None:
    print(f"Manual triggers -- sending RECOGNITION_TRIGGER events to {host}:{port}")
    print(f"Configured human steps: {_configured_steps_text()}")
    print(f"Defaults when omitted: piece_id={DEFAULT_MANUAL_PIECE_ID}, "
          f"round_id={DEFAULT_MANUAL_ROUND_ID}, progress={DEFAULT_MANUAL_PROGRESS}")
    print("Enter: step_id[,piece_id[,round_id[,progress]]]  (Ctrl+C or 'q' to quit)")


def _run_manual_trigger_loop(sender: UDPEventSender, host: str, port: int) -> None:
    """Prompt for trigger events and send them over the real UDP path.

    Bypasses RecognitionManager and TriggerManager entirely, so the communication
    layer's reaction time can be measured without recognition inference in the way.
    """
    _print_manual_trigger_help(host, port)
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if line.lower() in ("q", "quit", "exit"):
            break
        _send_manual_trigger(sender, line)


def _start_manual_trigger_thread(sender: UDPEventSender, host: str, port: int) -> threading.Thread:
    """Read manual triggers from stdin on a background thread.

    OpenCV's imshow/waitKey are main-thread only, so the vision loop keeps the main
    thread and the blocking input() moves here. Daemon, so Ctrl+C in the vision loop
    exits without waiting on a pending input().
    """
    _print_manual_trigger_help(host, port)

    def _read_lines() -> None:
        while True:
            try:
                line = input().strip()
            except (EOFError, OSError):
                return
            if line.lower() in ("q", "quit", "exit"):
                return
            _send_manual_trigger(sender, line)

    thread = threading.Thread(target=_read_lines, name="manual-trigger", daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    args = _parse_args()
    # Default destination; re-pointed at the run directory below if --log-dir is given.
    configure_logging("run_recognition")
    sender = UDPEventSender(args.host, args.port)

    if args.manual_trigger:
        try:
            _run_manual_trigger_loop(sender, args.host, args.port)
        except KeyboardInterrupt:
            pass
        finally:
            sender.close()
        sys.exit(0)

    source = _recognition_source(args)
    recognition_manager = RecognitionManager(
        model_dir=args.model_dir, video_source=source, show_video=not args.no_display,
        enable_step_model=not args.fake_recognition,
        render_world_skeleton=args.render_world_skeleton,
        record_path=args.record,
        record_fps=args.record_fps if args.record_fps is not None else args.loop_hz,
        vision_config=_build_vision_config(args))
    trigger_manager = TriggerManager()
    probability_plot: StepProbabilityPlot | None = None
    last_plotted_probabilities_timestamp = None
    last_mistake_id = None  # so the mistake verdict is printed only when it changes

    print(f"Recognition running with source: {source}")
    print(f"Publishing recognition events to {args.host}:{args.port}")
    if args.record:
        print(f"Recording the render screen to {args.record}")

    if args.fake_recognition:
        # Vision runs for real; only step classification comes from stdin.
        _start_manual_trigger_thread(sender, args.host, args.port)

    # Fixed-tick scheduling: accumulate the ideal next tick and sleep only the remainder,
    # so the cadence holds whatever update() costs. An overrun resyncs to now rather than
    # letting the deficit fire a catch-up burst later.
    period_s = 1.0 / args.loop_hz
    next_tick = time.perf_counter()
    overrun_warned_at = 0.0

    run_logger = _build_run_logger(args, source)
    if run_logger is not None:
        # Put run.log beside the frames.csv/events.csv/run.json it explains. Only now is
        # the directory name known, since --run-name defaults to a timestamp.
        configure_logging("run_recognition", log_file=run_logger.run_dir / "run.log")
        logger.info("Logging per-frame results to %s", run_logger.run_dir)

    frame_count = 0
    try:
        while True:
            update_started = time.perf_counter()
            result = recognition_manager.update()
            update_s = time.perf_counter() - update_started

            # A recorded source running out ends the run; a live source returning nothing
            # is just a hiccup.
            if recognition_manager.source_exhausted:
                print("Video source exhausted.")
                break

            if run_logger is not None:
                run_logger.log_frame(recognition_manager, update_s=update_s)

            if result is not None:
                for event in trigger_manager.update(result):
                    print(f"[recognition event] sending {event.event_type.name} {event.payload}", flush=True)
                    sender.send(event)
                    if run_logger is not None:
                        run_logger.log_event(event)
                    print(f"[recognition event] sent {event.event_type.name} {event.payload}", flush=True)

            # Mistake head (models trained with config.json's num_mistakes; None for
            # models without one). Reported on CHANGE rather than every frame: at
            # --loop-hz 10 a per-frame line would bury everything else, while the
            # transitions are the part worth seeing in a terminal.
            mistake_id = recognition_manager.last_mistake_id
            if mistake_id is not None and mistake_id != last_mistake_id:
                score = recognition_manager.last_mistake_score
                verdict = "MISTAKE" if mistake_id else "ok"
                print(f"[recognition] mistake: {verdict} "
                      f"(class {mistake_id}, score {score:.2f})", flush=True)
                last_mistake_id = mistake_id

            # Live line-graph preview of raw per-frame model output (see
            # step_probability_plot.py) -- fed from RecognitionManager.last_step_probabilities/
            # last_progress, which update every frame the LSTM runs on regardless of whether
            # a stable step transition was confirmed (unlike the sparse `result` above).
            if args.show_probabilities:
                probs_timestamp = recognition_manager.last_step_probabilities_timestamp
                if probs_timestamp is not None and probs_timestamp != last_plotted_probabilities_timestamp:
                    if probability_plot is None:
                        step_labels = (config.STEP_NAMES
                                       if len(config.STEP_NAMES) == recognition_manager.num_steps
                                       else None)
                        probability_plot = StepProbabilityPlot(
                            recognition_manager.num_steps,
                            history_seconds=args.probability_history_seconds,
                            step_labels=step_labels)
                    probability_plot.update(
                        probs_timestamp,
                        recognition_manager.last_step_probabilities,
                        recognition_manager.last_progress)
                    last_plotted_probabilities_timestamp = probs_timestamp
                elif probability_plot is None and recognition_manager.window_size is not None \
                        and frame_count % 15 == 0:
                    # The LSTM doesn't run at all -- so last_step_probabilities never gets
                    # set -- until self.buffer holds a full window_size frames. Log progress
                    # so a several-second wait before the window first appears doesn't look
                    # like a hang.
                    print(f"[recognition] buffering for step probabilities: "
                          f"{len(recognition_manager.buffer)}/{recognition_manager.window_size} frames")

            # Human location is a continuous, much-higher-frequency stream than the
            # discrete task events above -- throttled to config.HUMAN_LOCATION_
            # PUBLISH_EVERY_N_FRAMES (see recognition_manager.py's last_world_xyz,
            # updated every frame regardless of step-recognition state).
            frame_count += 1
            if args.max_frames is not None and frame_count >= args.max_frames:
                print(f"Reached --max-frames ({args.max_frames}).")
                break
            if frame_count % config.HUMAN_LOCATION_PUBLISH_EVERY_N_FRAMES == 0:
                world_xyz = recognition_manager.last_world_xyz
                if world_xyz is not None:
                    sender.send(Event(
                        event_type=EventType.HUMAN_LOCATION_UPDATE,
                        source="recognition",
                        payload={
                            "x": world_xyz[0], "y": world_xyz[1], "z": world_xyz[2],
                            "timestamp": recognition_manager.last_location_timestamp,
                            # Posture only: a pelvis-relative H36M-17 skeleton, separate
                            # from the world-frame x/y/z above.
                            "keypoints": recognition_manager.get_last_keypoints(),
                        },
                    ))

            next_tick += period_s
            now = time.perf_counter()
            sleep_s = next_tick - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Overran the budget: resync to now and warn at most every 5s.
                if now - overrun_warned_at > 5.0:
                    logger.warning(f"[recognition] loop overran budget by {-sleep_s * 1000:.0f}ms "
                          f"(target {period_s * 1000:.0f}ms/iteration) -- update() is the "
                          "bottleneck, not the sleep.")
                    overrun_warned_at = now
                next_tick = now
    except KeyboardInterrupt:
        print("Recognition stopped.")
    finally:
        if args.realtime_playback:
            # The headline number of a latency experiment: how much of the recording the
            # model actually got to see.
            read = recognition_manager.playback_frames_read
            dropped = recognition_manager.playback_frames_dropped
            total = read + dropped
            if total:
                print(f"[realtime playback] model saw {read}/{total} frames "
                      f"({100.0 * read / total:.1f}%), dropped {dropped} to processing latency.")
        elif args.iphone:
            # The iPhone capture keeps only the newest frame, so frames arriving while
            # update() runs or the loop sleeps are overwritten, just as with a camera.
            read = recognition_manager.playback_frames_read
            dropped = recognition_manager.playback_frames_dropped
            total = read + dropped
            if total:
                age_s = recognition_manager.live_frame_age_mean_s
                age_text = f", mean frame age {age_s * 1000:.0f} ms" if age_s is not None else ""
                print(f"[iphone] model saw {read}/{total} frames "
                      f"({100.0 * read / total:.1f}%), dropped {dropped} as stale{age_text}.")
        if run_logger is not None:
            summary = run_logger.close(recognition_manager)
            print(f"[run log] {run_logger.run_dir}")
            for key, value in summary.items():
                print(f"  {key}: {value}")
            print("  analyse with: uv run python 1_recognition/eval/analyse_runs.py "
                  f"{run_logger.run_dir.parent}")
        recognition_manager.release()
        if probability_plot is not None:
            probability_plot.release()
        sender.close()
