"""Background microphone input using Vosk or OpenAI Realtime."""

import base64
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import websocket
from dotenv import load_dotenv
from vosk import KaldiRecognizer, Model, SetLogLevel
from voice_result import VoiceOutcome


load_dotenv(Path(__file__).parent / "gpt_live" / ".env")

SAMPLE_RATE = 24_000


class VoiceInterface:
    def __init__(self, model_path: str | Path, gpt_enabled: bool, phrases,
                 device_name=None, timeout=8.0, output_device_name=None):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise RuntimeError(f"Vosk model not found: {model_path}")
        SetLogLevel(-1)
        self._model = Model(str(model_path))
        self._phrases = set(phrases)
        self._grammar = json.dumps(list(self._phrases))
        self._device = self._find_input_device(device_name)
        self._output_device_name = output_device_name
        self._timeout = timeout
        self._stop = threading.Event()
        self._thread = None
        self._ws = None
        self._gpt_enabled = gpt_enabled

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
    def _find_output_device(name):
        if not name:
            return sd.default.device[1]
        return next(
            index for index, device in enumerate(sd.query_devices())
            if device["max_output_channels"] > 0
            and name.casefold() in device["name"].casefold()
        )

    def _play_ready_beep(self) -> None:
        device = self._find_output_device(self._output_device_name)
        sample_rate = int(sd.query_devices(device, "output")["default_samplerate"])
        samples = np.arange(int(sample_rate * 0.4))
        tone = (0.4 * np.sin(2 * np.pi * 1000 * samples / sample_rate)).astype("float32")
        sd.play(tone, sample_rate, device=device, blocking=True)

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
            " Output screw done for finished screwing; output done for finished "
            "adjusting. Do not shorten screw done to done."
            " Output only the English command, without explanations. "
            "For unclear audio, output unknown."
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
            while not self._stop.is_set():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Voice session setup timed out")
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

    def start_listening(self, on_text, on_failure, *, beep=False) -> None:
        self.stop_listening()
        if self._thread is not None and self._thread.is_alive():
            on_failure(VoiceOutcome.ERROR, "Previous microphone worker is still stopping")
            return
        self._stop.clear()
        if self._gpt_enabled:
            self._thread = threading.Thread(
                target=self._listen_once_gpt,
                args=(on_text, on_failure, beep),
                daemon=True,
            )
        else:
            self._thread = threading.Thread(
                target=self._listen_once,
                args=(on_text, on_failure, beep),
                daemon=True,
            )
        self._thread.start()

    def _listen_once_gpt(self, on_text, on_failure, beep=False) -> None:
        speech_seen = False
        try:
            if not self._connect_gpt():
                return

            ws = self._ws
            ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            if beep:
                self._play_ready_beep()
            started = time.monotonic()
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
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
                    if event.get("type") in {"input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"}:
                        speech_seen = True
                    if event.get("type") == "response.output_text.done":
                        command = event["text"].strip().lower()
                        if command in self._phrases:
                            on_text(command)
                        else:
                            outcome = VoiceOutcome.UNRECOGNIZED if speech_seen else VoiceOutcome.NO_SPEECH
                            on_failure(outcome, "GPT returned no supported command")
                        return
                    if event.get("type") == "error":
                        raise RuntimeError(event["error"]["message"])
            if not self._stop.is_set():
                outcome = VoiceOutcome.UNRECOGNIZED if speech_seen else VoiceOutcome.NO_SPEECH
                on_failure(outcome, "Listening window expired after speech" if speech_seen else "No speech detected")
                # Discard an unfinished response rather than accepting it in a
                # later listening window after its audio has been cleared.
                if speech_seen and self._ws:
                    self._ws.close()
                    self._ws = None
        except Exception as error:
            if self._ws:
                self._ws.close()
                self._ws = None
            if not self._stop.is_set():
                on_failure(VoiceOutcome.ERROR, str(error))
        finally:
            if self._stop.is_set() and self._ws:
                # A state change may interrupt an in-flight model response.
                # Do not reuse its socket for the next question.
                self._ws.close()
                self._ws = None

    def _listen_once(self, on_text, on_failure, beep=False) -> None:
        audio = queue.Queue()

        def callback(data, frames, time_info, status):
            audio.put(bytes(data))

        try:
            device_info = sd.query_devices(self._device, "input")
            sample_rate = int(device_info["default_samplerate"])
            recognizer = KaldiRecognizer(self._model, sample_rate, self._grammar)
            if beep:
                self._play_ready_beep()
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
                            on_failure(VoiceOutcome.NO_SPEECH, "Vosk returned empty text")
                        return
                if not self._stop.is_set():
                    text = json.loads(recognizer.FinalResult()).get("text", "").strip()
                    if text:
                        on_text(text)
                    else:
                        on_failure(VoiceOutcome.NO_SPEECH, "Vosk listening window ended without text")
        except Exception as error:
            if not self._stop.is_set():
                on_failure(VoiceOutcome.ERROR, str(error))

    def stop_listening(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None

    def close(self) -> None:
        self.stop_listening()
        if self._ws:
            self._ws.close()
            self._ws = None


class NullVoiceInterface:
    def start_listening(self, on_text, on_failure, *, beep=False) -> None:
        pass

    def stop_listening(self) -> None:
        pass

    def close(self) -> None:
        pass
