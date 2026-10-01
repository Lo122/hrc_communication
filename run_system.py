"""Launch communication and recognition together, each in its own console window --
plus the robot's live TCP reader, which feeds the force-based task detectors.

Both need their own window because both read the keyboard: communication's CLI
prompt, and recognition's step prompt with --fake-recognition. The live reader
(4_execution/eval/read_ur_live_data.py --publish-ros) starts first, so the TCP force
is already on rosbridge when communication subscribes; communication then starts
before recognition so its UDP receiver is listening before recognition sends anything.

Arguments:
    --host, --port              event transport, passed to both
    --demo                      passed to run_communication.py: scripted opening
    --reactive                  the reactive system, alone: only communication starts
                                (run_communication.py --reactive) -- no recognition, no
                                live reader, so no recognition flags. The robot acts
                                only on the human's commands (not with --demo)
    --record-camera             with --reactive: record the camera to the run directory
                                (camera.mp4, run_recorder.py) without recognition; its
                                camera flags (--camera, --video-source, --iphone, ...)
                                go to run_recorder.py
    --log-dir, --run-name       both processes log this run to <log-dir>/<run-name>/
                                (default logs/runs/run_<date>_<time>); --no-run-log
                                turns it off
    --debug-trigger, --debug-step-id, --debug-progress, --debug-round-id
                                passed to run_communication.py
    --robot-ip, --robot-live-hz the live reader's robot and rosbridge publish rate
                                (default config.ROBOT_IP, config.ROBOT_LIVE_DATA_ROS_HZ);
                                --no-robot-live skips it (replays, no robot)
    everything else             passed to run_recognition.py unchanged, so every
                                run_recognition.py flag works here as well (with
                                --reactive --record-camera: to run_recorder.py)

Usage:
    uv run python run_system.py --camera
    uv run python run_system.py --demo --camera --robot-ip 192.168.1.10
    uv run python run_system.py --reactive
    uv run python run_system.py --reactive --record-camera --video-source 6
    uv run python run_system.py --video-source clip.mp4 --debug-trigger --no-robot-live

    uv run python run_system.py `
        --iphone `
        --model-dir 1_recognition/best_model/S3_10fps_8s_bg05 `
        --intrinsics-file iphone_intrinsics.json `
        --extrinsics-file iphone_extrinsics.json `
        --body-calibration 1_recognition/calib_data/body_uid-08.json `
        --iphone-rotate 270 `
        --fp16 `
        --loop-hz 10

    uv run python run_system.py `
            --video-source 6 `
            --model-dir 1_recognition/best_model/S3_10fps_8s_bg05 `
            --intrinsics-file intrinsics_640x360_obs.json `
            --extrinsics-file extrinsics_3840x2160_obs.json `
            --body-calibration 1_recognition/calib_data/body_uid-08.json `
            --fp16 `
            --loop-hz 10

    # The robot-camera calibration (setup/calibration/robot_camera_calibration.py) is an
    # extrinsics file as well: the human's position is then in its marker's frame, where
    # the robot base pose is known too.
    uv run python run_system.py `
            --video-source 6 `
            --model-dir 1_recognition/best_model/S3_10fps_8s_bg05 `
            --intrinsics-file intrinsics_640x360_obs.json `
            --extrinsics-file robot_camera_calibration_20261001_083647.json `
            --body-calibration 1_recognition/calib_data/body_uid-08.json `
            --fp16

    uv run python run_system.py --demo `
                --video-source 6 `
                --model-dir 1_recognition/best_model/S3_10fps_8s_bg05 `
                --intrinsics-file intrinsics_640x360_obs.json `
                --extrinsics-file extrinsics_3840x2160_obs.json `
                --body-calibration 1_recognition/calib_data/body_uid-08.json `
                --fp16 `
                --loop-hz 10 `
                --show-probabilities


One run directory then holds what recognition saw and what communication did about
it, on one clock:
    frames.csv, events.csv,     recognition: every processed frame, every task update
    run.json, run.log           it sent (time / epoch_s columns)
    communication_events.jsonl  communication: every event, transition and message
    timeline.csv                the same, readable: when a task update was recognized,
                                when it arrived, and what followed
    ur_live_data.log            the live reader: joint angles, TCP pose and force
    camera.mp4,                 --reactive --record-camera: the camera video, and when
    camera.timestamps.csv       each frame was taken (epoch s, communication's clock)

Both children run on this same interpreter (sys.executable) rather than through
`uv run`, so there is one environment sync, not two racing each other.

Stopping: communication is the system -- closing its window (or Ctrl+C here) ends
the run. Recognition or the live reader exiting on its own (video finished, q in its
preview; robot unreachable) leaves communication running, so a robot task in progress
is not cut off. The live reader only reads the robot, so stopping it is harmless. Stop
recognition with q / Ctrl+C in its own window when using --record/--record-raw: a
process terminated from here cannot finish writing the .mp4. The reactive system's
camera recording is stopped properly when the run ends (its stop file), so its
camera.mp4 is always finished -- as long as its window is not closed with the X.

decision view in live: 
http://127.0.0.1:8770/

"""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent
COMMUNICATION_SCRIPT = ROOT / "run_communication.py"
RECOGNITION_SCRIPT = ROOT / "run_recognition.py"
ROBOT_LIVE_SCRIPT = ROOT / "4_execution" / "eval" / "read_ur_live_data.py"
ROBOT_LIVE_LOG = "ur_live_data.log"
RECORDER_SCRIPT = ROOT / "run_recorder.py"
# Created in the run directory to have the recorder finish its .mp4 and exit.
RECORDER_STOP_FILE = "camera.stop"
RECORDER_STOP_WAIT_S = 5.0

# Hidden first argument: run the named script in this process (see _run_child).
CHILD_FLAG = "--_child"
COMMUNICATION_STARTUP_S = 0.8  # lets the UDP receiver bind before recognition sends
POLL_S = 1.0


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Launcher options, plus every argument left over for run_recognition.py."""
    parser = argparse.ArgumentParser(
        description="Launch communication and recognition in separate windows. Arguments "
                    "not listed here are passed to run_recognition.py.",
        # Off, so a run_recognition.py flag is never read as an abbreviation of one here.
        allow_abbrev=False)
    parser.add_argument("--host", default=None, help="Event transport host, for both processes.")
    parser.add_argument("--port", type=int, default=None, help="Event transport port, for both processes.")
    communication = parser.add_argument_group("passed to run_communication.py")
    mode = communication.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true",
                      help="Open with the scripted demo dialogue for the first panel.")
    mode.add_argument("--reactive", action="store_true",
                      help="Reactive system: the robot offers nothing and acts only on the human's commands. "
                           "Starts communication alone: no recognition, no live TCP reader.")
    communication.add_argument("--debug-trigger", action="store_true",
                               help="Inject one fake human task update at startup.")
    communication.add_argument("--debug-step-id", type=int, default=None,
                               help="Human step id (config.STEP_NAMES index) for --debug-trigger.")
    communication.add_argument("--debug-progress", type=float, default=None,
                               help="Progress value for --debug-trigger.")
    communication.add_argument("--debug-round-id", type=int, default=None,
                               help="Round id for --debug-trigger.")
    run_log = parser.add_argument_group("run log, passed to both")
    run_log.add_argument("--log-dir", default=str(ROOT / config.RUN_LOG_DIR),
                         help="Both processes log this run to <LOG_DIR>/<RUN_NAME>/.")
    run_log.add_argument("--run-name", default=None,
                         help="Run directory name; defaults to run_<date>_<time>.")
    run_log.add_argument("--no-run-log", action="store_true", help="Do not log the run to a directory.")
    robot_live = parser.add_argument_group(
        "live robot data (4_execution/eval/read_ur_live_data.py --publish-ros: the TCP force "
        "the task detectors read)")
    robot_live.add_argument("--no-robot-live", action="store_true",
                            help="Do not start the live TCP reader (no robot, or a replay).")
    robot_live.add_argument("--robot-ip", default=config.ROBOT_IP,
                            help=f"UR controller the live reader connects to (default "
                                 f"config.ROBOT_IP, {config.ROBOT_IP}).")
    robot_live.add_argument("--robot-live-hz", type=float, default=config.ROBOT_LIVE_DATA_ROS_HZ,
                            help="How often the live reader publishes to rosbridge (Hz).")
    recording = parser.add_argument_group(
        "camera recording (run_recorder.py): the reactive system's video, without recognition")
    recording.add_argument("--record-camera", action="store_true",
                           help="With --reactive: record the camera to <run dir>/camera.mp4. Camera "
                                "flags (--camera, --video-source, --iphone, ...) go to run_recorder.py.")
    args, recognition_args = parser.parse_known_args()
    if args.record_camera and not args.reactive:
        parser.error("--record-camera is for --reactive. With recognition, record through it: "
                     "--record-raw <path> (passed to run_recognition.py).")
    if args.record_camera and args.no_run_log:
        parser.error("--record-camera records into the run directory: drop --no-run-log.")
    if args.reactive and recognition_args and not args.record_camera:
        parser.error("--reactive runs without recognition, so these run_recognition.py arguments "
                     f"do not apply: {' '.join(recognition_args)} (--record-camera records the camera)")
    if args.run_name is None:
        args.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return args, recognition_args


def _run_log_args(args: argparse.Namespace) -> list[str]:
    if args.no_run_log:
        return ["--no-run-log"]
    return ["--log-dir", args.log_dir, "--run-name", args.run_name]


def _transport_args(args: argparse.Namespace) -> list[str]:
    result = []
    if args.host is not None:
        result.extend(["--host", args.host])
    if args.port is not None:
        result.extend(["--port", str(args.port)])
    return result


def _communication_args(args: argparse.Namespace) -> list[str]:
    result = _transport_args(args) + _run_log_args(args)
    if args.demo:
        result.append("--demo")
    if args.reactive:
        result.append("--reactive")
    if args.debug_trigger:
        result.append("--debug-trigger")
    # Only what was given, so run_communication.py's own defaults still apply.
    for flag, value in (("--debug-step-id", args.debug_step_id),
                        ("--debug-progress", args.debug_progress),
                        ("--debug-round-id", args.debug_round_id)):
        if value is not None:
            result.extend([flag, str(value)])
    return result


def _robot_live_args(args: argparse.Namespace) -> list[str]:
    """The live reader publishes to rosbridge and, with a run log, logs its samples into
    the run directory, beside what recognition and communication logged."""
    result = ["--ip", args.robot_ip, "--publish-ros", "--ros-hz", str(args.robot_live_hz)]
    if not args.no_run_log:
        result.extend(["--log-file", str(Path(args.log_dir) / args.run_name / ROBOT_LIVE_LOG)])
    return result


def _recorder_stop_file(args: argparse.Namespace) -> Path:
    return Path(args.log_dir) / args.run_name / RECORDER_STOP_FILE


def _recorder_args(args: argparse.Namespace, camera_args: list[str]) -> list[str]:
    """The recorder writes camera.mp4 into the run directory and stops once its stop
    file appears there (_stop_recorder)."""
    return ["--log-dir", args.log_dir, "--run-name", args.run_name,
            "--stop-file", str(_recorder_stop_file(args)), *camera_args]


def _stop_recorder(process: subprocess.Popen, stop_file: Path) -> None:
    """Have the recorder finish its .mp4 and exit -- terminated from here it could not,
    and the video would be unplayable -- then wait for it."""
    if process.poll() is None:
        print("Stopping the camera recording.")
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.touch()
        try:
            process.wait(timeout=RECORDER_STOP_WAIT_S)
        except subprocess.TimeoutExpired:
            print(f"The camera recording did not stop within {RECORDER_STOP_WAIT_S:g} s.")
    stop_file.unlink(missing_ok=True)


def _open_window(script: Path, script_args: list[str]) -> subprocess.Popen:
    command = [sys.executable, str(Path(__file__).resolve()), CHILD_FLAG, str(script), *script_args]
    kwargs = {"cwd": str(ROOT)}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    return subprocess.Popen(command, **kwargs)


def _run_child(script: str, script_args: list[str]) -> None:
    """Run one script as __main__ in this console, and keep the window open if it fails.

    A new console closes the moment its process ends, which would take a mistyped
    flag's argparse error or a model-loading traceback with it. Running the script
    in this same process (rather than as a grandchild) keeps terminate() from the
    launcher reaching the script itself.
    """
    sys.argv = [script, *script_args]
    code = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        if isinstance(exc.code, int):
            code = exc.code
        elif exc.code is not None:
            print(exc.code, file=sys.stderr)
            code = 1
    except KeyboardInterrupt:
        pass
    except BaseException:
        traceback.print_exc()
        code = 1
    if code:
        try:
            input(f"\n{Path(script).name} exited with code {code}. Press Enter to close this window.")
        except (EOFError, KeyboardInterrupt):
            pass
    sys.exit(code)


def main() -> None:
    args, camera_args = _parse_args()
    recognition_args = _transport_args(args) + _run_log_args(args) + camera_args
    # The reactive system needs neither: it has no recognition and no force-based
    # detectors. Its camera is only recorded, if asked.
    start_recognition = not args.reactive
    start_robot_live = not (args.reactive or args.no_robot_live)
    stop_file = _recorder_stop_file(args)

    print(f"Python: {sys.executable}")
    print(f"Communication: run_communication.py {' '.join(_communication_args(args))}")
    if start_recognition:
        print(f"Recognition:   run_recognition.py {' '.join(recognition_args)}")
    if args.record_camera:
        print(f"Recorder:      run_recorder.py {' '.join(_recorder_args(args, camera_args))}")
    if start_robot_live:
        print(f"Robot live:    read_ur_live_data.py {' '.join(_robot_live_args(args))}")
    if not args.no_run_log:
        print(f"Run log:       {Path(args.log_dir) / args.run_name}")

    # Sidecars: exiting on their own leaves communication running (reported once).
    sidecars = {}
    if args.record_camera:
        stop_file.unlink(missing_ok=True)  # left by an earlier run of the same name
        # First, so the video covers the whole run.
        sidecars["camera recording"] = _open_window(RECORDER_SCRIPT, _recorder_args(args, camera_args))
    if start_robot_live:
        sidecars["robot live data"] = _open_window(ROBOT_LIVE_SCRIPT, _robot_live_args(args))
    communication = _open_window(COMMUNICATION_SCRIPT, _communication_args(args))
    if start_recognition:
        time.sleep(COMMUNICATION_STARTUP_S)
        sidecars["recognition"] = _open_window(RECOGNITION_SCRIPT, recognition_args)
    print("Started. " + ", ".join(f"{name} PID {process.pid}" for name, process in
                                  {"communication": communication, **sidecars}.items()) + ".")
    print("Close the communication window or press Ctrl+C here to stop.")

    reported = set()
    try:
        while communication.poll() is None:
            for name, process in sidecars.items():
                if name not in reported and process.poll() is not None:
                    reported.add(name)
                    print(f"{name.capitalize()} exited (code {process.returncode}); "
                          "communication keeps running.")
            time.sleep(POLL_S)
        print(f"Communication exited (code {communication.returncode}).")
    except KeyboardInterrupt:
        print("Stopping HRC runtime.")
    finally:
        if "camera recording" in sidecars:
            _stop_recorder(sidecars["camera recording"], stop_file)
        for name, process in (*sidecars.items(), ("communication", communication)):
            if process.poll() is None:
                print(f"Terminating {name} (PID {process.pid}).")
                process.terminate()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == CHILD_FLAG:
        _run_child(sys.argv[2], sys.argv[3:])
    else:
        main()
