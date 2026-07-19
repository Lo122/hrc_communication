"""Launch the two-window HRC runtime."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch communication and recognition in separate windows.")
    parser.add_argument("--camera", action="store_true", help="Use realtime camera input for recognition.")
    parser.add_argument("--video-source", default=None, help="Video path or stream URL for recognition.")
    parser.add_argument("--model-dir", default=None, help="Recognition model directory.")
    parser.add_argument("--host", default=None, help="Recognition event transport host.")
    parser.add_argument("--port", type=int, default=None, help="Recognition event transport port.")
    return parser.parse_args()


def _script_command(script_name: str, extra_args: list[str] | None = None) -> list[str]:
    command = [sys.executable, str(ROOT / script_name)]
    if extra_args:
        command.extend(extra_args)
    return command


def _recognition_args(args: argparse.Namespace) -> list[str]:
    result = []
    if args.camera:
        result.append("--camera")
    if args.video_source is not None:
        result.extend(["--video-source", args.video_source])
    if args.model_dir is not None:
        result.extend(["--model-dir", args.model_dir])
    if args.host is not None:
        result.extend(["--host", args.host])
    if args.port is not None:
        result.extend(["--port", str(args.port)])
    return result


def _communication_args(args: argparse.Namespace) -> list[str]:
    result = []
    if args.host is not None:
        result.extend(["--host", args.host])
    if args.port is not None:
        result.extend(["--port", str(args.port)])
    return result


def _open_process(command: list[str]):
    kwargs = {"cwd": str(ROOT)}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    return subprocess.Popen(command, **kwargs)


if __name__ == "__main__":
    args = _parse_args()
    communication = _open_process(_script_command("run_communication.py", _communication_args(args)))
    time.sleep(0.8)
    recognition = _open_process(_script_command("run_recognition.py", _recognition_args(args)))

    print("Started HRC dual-window runtime.")
    print(f"Communication PID: {communication.pid}")
    print(f"Recognition PID: {recognition.pid}")
    print("Close either child window or press Ctrl+C here to stop monitoring.")

    try:
        while True:
            communication_code = communication.poll()
            recognition_code = recognition.poll()
            if communication_code is not None or recognition_code is not None:
                print(f"Communication exited: {communication_code}")
                print(f"Recognition exited: {recognition_code}")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping HRC dual-window runtime.")
    finally:
        for process in [communication, recognition]:
            if process.poll() is None:
                process.terminate()
