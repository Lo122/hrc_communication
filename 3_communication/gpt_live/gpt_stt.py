"""Standalone test matching VoiceInterface's GPT configuration."""

import base64
import json
import os
import time
from pathlib import Path

import sounddevice as sd
import websocket
from dotenv import load_dotenv


load_dotenv(Path(__file__).with_name(".env"))

SAMPLE_RATE = 24_000
TIMEOUT = 8.0
COMMANDS = {
    "yes", "no", "later", "pause", "resume", "restart", "cancel",
    "faster", "slower", "free drive", "return home", "manual recovery", "done",
}


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("VOICE_MODEL")
    if not api_key or not model:
        raise RuntimeError("OPENAI_API_KEY and VOICE_MODEL must be set")

    instructions = (
        "Understand the user's spoken intent and output exactly one lowercase "
        f"command from: {', '.join(sorted(COMMANDS))}, unknown. "
        "Map natural expressions to their meaning and output unknown if unclear."
    )
    ws = websocket.create_connection(
        f"wss://api.openai.com/v1/realtime?model={model}",
        header=[f"Authorization: Bearer {api_key}"],
    )
    try:
        ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": instructions,
                "output_modalities": ["text"],
                "max_output_tokens": 4096,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                        "turn_detection": {
                            "type": "semantic_vad",
                            "eagerness": "high",
                            "create_response": True,
                        },
                    }
                },
            },
        }))
        ws.settimeout(0.01)
        started = time.monotonic()
        print(f"Listening for {TIMEOUT:.0f} seconds...")

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
