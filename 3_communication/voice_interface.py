"""Background microphone input using Vosk or OpenAI Realtime."""

import base64
import json
import os
import queue
import threading
import time
import winsound
from pathlib import Path

import sounddevice as sd
import websocket
from dotenv import load_dotenv
from vosk import KaldiRecognizer, Model, SetLogLevel


load_dotenv(Path(__file__).parent / "gpt_live" / ".env")


class VoiceInterface:
    def __init__(self, model_path: str | Path, gpt_enabled: bool, phrases, device_name=None, timeout=8.0):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise RuntimeError(f"Vosk model not found: {model_path}")
        SetLogLevel(-1)
        self._model = Model(str(model_path))
        self._phrases = set(phrases)
        self._grammar = json.dumps(list(self._phrases))
        self._device = self._find_input_device(device_name)
        self._timeout = timeout
        self._stop = threading.Event()
        self._thread = None
        self._ws = None
        self._gpt_enabled = gpt_enabled
        if self._gpt_enabled:
            try:
                self._connect_gpt()
            except Exception:
                self._ws = None

    @staticmethod
    def _find_input_device(name):
        if not name:
            return None
        matches = [
            index for index, device in enumerate(sd.query_devices())
            if device["max_input_channels"] > 0
            and name.casefold() in device["name"].casefold()
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one input device matching {name!r}, found {len(matches)}"
            )
        return matches[0]

    @staticmethod
    def _play_ready_beep() -> None:
        winsound.Beep(1000, 150)

    def _connect_gpt(self) -> bool:
        if self._ws and self._ws.connected:
            return True

        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("VOICE_MODEL")
        if not api_key or not model:
            raise RuntimeError("OPENAI_API_KEY and VOICE_MODEL must be set")

        instructions = (
            "Understand the user's spoken intent and output exactly one lowercase "
            f"command from: {', '.join(sorted(self._phrases))}, unknown. "
            "Map natural expressions to their meaning and output unknown if unclear."
        )
        ws = websocket.create_connection(
            f"wss://api.openai.com/v1/realtime?model={model}",
            header=[f"Authorization: Bearer {api_key}"],
            timeout=5,
        )
        self._ws = ws
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
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "turn_detection": {
                                "type": "semantic_vad",
                                "eagerness": "high",
                                "create_response": True,
                            },
                        }
                    },
                },
            }))
            ws.settimeout(0.2)
            while not self._stop.is_set():
                try:
                    event = json.loads(ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                if event.get("type") == "session.updated":
                    ws.settimeout(0.01)
                    return True
                if event.get("type") == "error":
                    raise RuntimeError(event["error"]["message"])
            ws.close()
            self._ws = None
            return False
        except Exception:
            ws.close()
            self._ws = None
            raise

    def start_listening(self, on_text, on_failure) -> None:
        self.stop_listening()
        self._stop.clear()
        if self._gpt_enabled:
            self._thread = threading.Thread(
                target=self._listen_once_gpt,
                args=(on_text, on_failure),
                daemon=True,
            )
        else:
            self._thread = threading.Thread(
                target=self._listen_once,
                args=(on_text, on_failure),
                daemon=True,
            )
        self._thread.start()

    def _listen_once_gpt(self, on_text, on_failure) -> None:
        try:
            if not self._connect_gpt():
                return

            ws = self._ws
            ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            self._play_ready_beep()
            started = time.monotonic()
            with sd.RawInputStream(
                samplerate=24000,
                blocksize=2400,
                device=self._device,
                dtype="int16",
                channels=1,
            ) as stream:
                while not self._stop.is_set() and time.monotonic() - started < self._timeout:
                    audio, _ = stream.read(2400)
                    ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio).decode("ascii"),
                    }))
                    try:
                        event = json.loads(ws.recv())
                    except websocket.WebSocketTimeoutException:
                        continue
                    #remove print statement in production, it's for debugging
                    # print(f"[gpt] {event}")
                    if event.get("type") == "response.output_text.done":
                        command = event["text"].strip().lower()
                        on_text(command) if command in self._phrases else on_failure()
                        return
                    if event.get("type") == "error":
                        raise RuntimeError(event["error"]["message"])
            if not self._stop.is_set():
                on_failure()
        except Exception as error:
            if self._ws:
                self._ws.close()
                self._ws = None
            if not self._stop.is_set():
                on_failure(error)

    def _listen_once(self, on_text, on_failure) -> None:
        audio = queue.Queue()
        device_info = sd.query_devices(self._device, "input")
        sample_rate = int(device_info["default_samplerate"])
        recognizer = KaldiRecognizer(self._model, sample_rate, self._grammar)
        self._play_ready_beep()

        def callback(data, frames, time_info, status):
            audio.put(bytes(data))

        try:
            started = time.monotonic()
            with sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=4000,
                device=self._device,
                dtype="int16",
                channels=1,
                callback=callback,
            ):
                while not self._stop.is_set() and time.monotonic() - started < self._timeout:
                    try:
                        data = audio.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if recognizer.AcceptWaveform(data):
                        text = json.loads(recognizer.Result()).get("text", "").strip()
                        if text:
                            on_text(text)
                        else:
                            on_failure()
                        return
                if not self._stop.is_set():
                    text = json.loads(recognizer.FinalResult()).get("text", "").strip()
                    on_text(text) if text else on_failure()
        except Exception as error:
            if not self._stop.is_set():
                on_failure(error)

    def stop_listening(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._thread = None

    def close(self) -> None:
        self.stop_listening()
        if self._ws:
            self._ws.close()
            self._ws = None


class NullVoiceInterface:
    def start_listening(self, on_text, on_failure) -> None:
        pass

    def stop_listening(self) -> None:
        pass

    def close(self) -> None:
        pass
