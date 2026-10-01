"""Serialized, blocking Windows text-to-speech output."""

import gc
import queue
import threading

import pyttsx3


class TTSManager:
    """Keep SAPI5 on one worker thread and prevent overlapping speech."""

    def __init__(self, rate: int = 300, voice: str | None = None):
        """voice: use the first installed voice whose name contains this ("Zira")."""
        self._rate = rate
        self._voice = voice
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            text, finished, errors = item
            engine = None
            try:
                engine = pyttsx3.init("sapi5")
                engine.setProperty("rate", self._rate)
                if self._voice:
                    match = next((voice for voice in engine.getProperty("voices")
                                  if self._voice.lower() in voice.name.lower()), None)
                    if match is not None:
                        engine.setProperty("voice", match.id)
                engine.say(text)
                engine.runAndWait()
            except Exception as error:
                errors.append(error)
            finally:
                if engine is not None:
                    engine.stop()
                    del engine
                    gc.collect()
                finished.set()

    def speak(self, text: str) -> None:
        finished, errors = threading.Event(), []
        self._queue.put((text, finished, errors))
        finished.wait()
        if errors:
            raise errors[0]

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2.0)


class NullTTSManager:
    def speak(self, text: str) -> None:
        pass

    def close(self) -> None:
        pass
