"""Background microphone input using the offline Vosk recognizer."""

import json
import queue
import threading
import time
from pathlib import Path

import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel


class VoiceInterface:
    def __init__(self, model_path: str | Path, phrases, device_name=None, timeout=8.0):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise RuntimeError(f"Vosk model not found: {model_path}")
        SetLogLevel(-1)
        self._model = Model(str(model_path))
        self._grammar = json.dumps(list(phrases))
        self._device = self._find_input_device(device_name)
        self._timeout = timeout
        self._stop = threading.Event()
        self._thread = None

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

    def start_listening(self, on_text, on_failure) -> None:
        self.stop_listening()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._listen_once,
            args=(on_text, on_failure),
            daemon=True,
        )
        self._thread.start()

    def _listen_once(self, on_text, on_failure) -> None:
        audio = queue.Queue()
        device_info = sd.query_devices(self._device, "input")
        sample_rate = int(device_info["default_samplerate"])
        recognizer = KaldiRecognizer(self._model, sample_rate, self._grammar)

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


class NullVoiceInterface:
    def start_listening(self, on_text, on_failure) -> None:
        pass

    def stop_listening(self) -> None:
        pass

    def close(self) -> None:
        pass
