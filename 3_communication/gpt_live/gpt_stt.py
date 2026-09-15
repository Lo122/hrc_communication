"""Standalone microphone test using the runtime's task-context templates."""

import base64
import argparse
import json
import os
import sys
import time
from pathlib import Path

import sounddevice as sd
import websocket
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "0_core", ROOT / "3_communication"):
    sys.path.insert(0, str(directory))

from events import RobotTaskState
from voice_context import VoiceContext, STATE_CONTEXTS, TASK_DESCRIPTIONS, build_instructions

load_dotenv(Path(__file__).with_name(".env"))

SAMPLE_RATE = 24_000
TIMEOUT = 8.0
COMMANDS = {
    "yes", "no", "later", "pause", "resume", "restart", "cancel",
    "faster", "slower", "free drive", "return home", "manual recovery", "done",
    "screw done", "screwing done", "finished screwing", "adjustment done",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test GPT voice interpretation in one task stage.")
    parser.add_argument("--state", choices=[state.name for state in STATE_CONTEXTS], default="R_WAITING_RESPONSE")
    parser.add_argument("--task-id", type=int, choices=sorted(TASK_DESCRIPTIONS), default=1)
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("VOICE_MODEL")
    if not api_key or not model:
        raise RuntimeError("OPENAI_API_KEY and VOICE_MODEL must be set")

    instructions = build_instructions(VoiceContext(RobotTaskState[args.state], args.task_id, "voice_test"))
    ws = websocket.create_connection(
        f"wss://api.openai.com/v1/realtime?model={model}",
        header=[f"Authorization: Bearer {api_key}"],
        timeout=5,
    )
    try:
        ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": instructions,
                "output_modalities": ["text"],
                "max_output_tokens": 4096,
                "reasoning": {"effort": "low"},
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                        "turn_detection": {
                            "type": "semantic_vad",
                            "eagerness": "medium",
                            "create_response": True,
                        },
                    }
                },
            },
        }))
        ws.settimeout(0.2)
        deadline = time.monotonic() + 5.0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Voice session setup timed out")
            try:
                event = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            if event.get("type") == "error":
                raise RuntimeError(event["error"]["message"])
            if (event.get("type") == "session.updated"
                    and event.get("session", {}).get("instructions") == instructions):
                break
        ws.settimeout(0.01)
        started = time.monotonic()
        print(f"Task {args.task_id}, state {args.state}. Listening for {TIMEOUT:.0f} seconds...")

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=2400
        ) as stream:
            while time.monotonic() - started < TIMEOUT:
                audio, _ = stream.read(2400)
                ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio).decode("ascii"),
                }))
                try:
                    event = json.loads(ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue

                event_type = event.get("type")
                print("Event:", event_type)
                if event_type == "response.output_text.done":
                    command = event["text"].strip().lower()
                    print("Raw output:", repr(event["text"]))
                    print("Command:", command if command in COMMANDS else "unknown")
                    return
                if event_type == "error":
                    raise RuntimeError(event["error"]["message"])

        print("No command received before timeout.")
    finally:
        ws.close()


if __name__ == "__main__":
    main()
