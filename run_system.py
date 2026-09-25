"""Launch communication and recognition together, each in its own console window.

Both need their own window because both read the keyboard: communication's CLI
prompt, and recognition's step prompt with --fake-recognition. Communication
starts first so its UDP receiver is listening before recognition sends anything.

Arguments:
    --host, --port              event transport, passed to both
    --debug-trigger, --debug-step-id, --debug-progress, --debug-round-id
                                passed to run_communication.py
    everything else             passed to run_recognition.py unchanged, so every
                                run_recognition.py flag works here as well

Usage:
    uv run python run_system.py --camera
    uv run python run_system.py --video-source clip.mp4 --debug-trigger

    uv run python run_system.py `
        --iphone `
        --model-dir 1_recognition/best_model/3d_skeleton_01 `
        --intrinsics-file iphone_intrinsics.json `
        --extrinsics-file iphone_extrinsics.json `
        --body-calibration 1_recognition/calib_data/body_uid-08.json `
        --iphone-rotate 270 `
        --fp16 `
        --loop-hz 10

    uv run python run_system.py `
            --video-source 6 `
            --model-dir 1_recognition/best_model/3d_skeleton_01 `
            --intrinsics-file intrinsics_640x360_obs.json `
            --extrinsics-file extrinsics_3840x2160_obs.json `
            --body-calibration 1_recognition/calib_data/body_uid-08.json `
            --fp16 `

Both children run on this same interpreter (sys.executable) rather than through
`uv run`, so there is one environment sync, not two racing each other.

Stopping: communication is the system -- closing its window (or Ctrl+C here) ends
the run. Recognition exiting on its own (video finished, q in its preview) leaves
communication running, so a robot task in progress is not cut off. Stop
recognition with q / Ctrl+C in its own window when using --record/--record-raw: a
process terminated from here cannot finish writing the .mp4.
"""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMUNICATION_SCRIPT = ROOT / "run_communication.py"
RECOGNITION_SCRIPT = ROOT / "run_recognition.py"

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
    communication.add_argument("--debug-trigger", action="store_true",
                               help="Inject one fake human task update at startup.")
    communication.add_argument("--debug-step-id", type=int, default=None,
                               help="Human step id (config.STEP_NAMES index) for --debug-trigger.")
    communication.add_argument("--debug-progress", type=float, default=None,
                               help="Progress value for --debug-trigger.")
    communication.add_argument("--debug-round-id", type=int, default=None,
                               help="Round id for --debug-trigger.")
    return parser.parse_known_args()


def _transport_args(args: argparse.Namespace) -> list[str]:
    result = []
    if args.host is not None:
        result.extend(["--host", args.host])
    if args.port is not None:
        result.extend(["--port", str(args.port)])
    return result


def _communication_args(args: argparse.Namespace) -> list[str]:
    result = _transport_args(args)
    if args.debug_trigger:
        result.append("--debug-trigger")
    # Only what was given, so run_communication.py's own defaults still apply.
    for flag, value in (("--debug-step-id", args.debug_step_id),
                        ("--debug-progress", args.debug_progress),
                        ("--debug-round-id", args.debug_round_id)):
        if value is not None:
            result.extend([flag, str(value)])
    return result


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
    args, recognition_args = _parse_args()
    recognition_args = _transport_args(args) + recognition_args

    print(f"Python: {sys.executable}")
    print(f"Communication: run_communication.py {' '.join(_communication_args(args))}")
    print(f"Recognition:   run_recognition.py {' '.join(recognition_args)}")

    communication = _open_window(COMMUNICATION_SCRIPT, _communication_args(args))
    time.sleep(COMMUNICATION_STARTUP_S)
    recognition = _open_window(RECOGNITION_SCRIPT, recognition_args)
    print(f"Started. Communication PID {communication.pid}, recognition PID {recognition.pid}.")
    print("Close the communication window or press Ctrl+C here to stop.")

    recognition_reported = False
    try:
        while communication.poll() is None:
            if not recognition_reported and recognition.poll() is not None:
                recognition_reported = True
                print(f"Recognition exited (code {recognition.returncode}); communication "
                      "keeps running.")
            time.sleep(POLL_S)
        print(f"Communication exited (code {communication.returncode}).")
    except KeyboardInterrupt:
        print("Stopping HRC runtime.")
    finally:
        for name, process in (("recognition", recognition), ("communication", communication)):
            if process.poll() is None:
                print(f"Terminating {name} (PID {process.pid}).")
                process.terminate()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == CHILD_FLAG:
        _run_child(sys.argv[2], sys.argv[3:])
    else:
        main()
