from vosk import Model, KaldiRecognizer
import urllib.request
from pathlib import Path
import sys
import zipfile
import sounddevice as sd
import time
import queue
import json
import pyttsx3

COMMUNICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMUNICATION_DIR))
from cmd_parser import CommandParser

def tts(text: str, rate: int) -> None:
    print(f"Speaking: {text}")
    engine = pyttsx3.init("sapi5")
    engine.setProperty("rate", rate)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
 
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / MODEL_NAME

if not MODEL_PATH.is_dir():
    print(f"Downloading Vosk model from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH.with_suffix(".zip"))
    print("Extracting Vosk model...")

    with zipfile.ZipFile(MODEL_PATH.with_suffix(".zip"), "r") as zip_ref:
        zip_ref.extractall(MODEL_PATH.parent)
    print("Vosk model downloaded and extracted.")


def _find_input_device(name: str | None):
    if not name:
        return None
    matches = [
        index for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
        and name.casefold() in device["name"].casefold()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one input device matching {name!r}, found {len(matches)}")
    return matches[0]

def vosk_stt(device_name: str | None, timeout: float) -> None:
    device = _find_input_device(device_name)

    device_info = sd.query_devices(device, "input")
    sample_rate = int(device_info["default_samplerate"])
    model = Model(str(MODEL_PATH))

    command_grammar = json.dumps(list(CommandParser._ALIASES))
    recognizer = KaldiRecognizer(model, sample_rate, command_grammar)

    audio_queue: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Audio warning: {status}", file=sys.stderr)
        audio_queue.put(bytes(indata))


    print(f"Input device: {device_info['name']}")
    print(f"Speak now. Listening for up to {timeout:.1f} seconds...")
    started = time.monotonic()

    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=4000,
        device=device,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while time.monotonic() - started < timeout:
            try:
                data = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result()).get("text", "").strip()
                if text:
                    print(f"Recognized text: {text}")
                    tts(f"Command {text} acknowledged.", 175)
                    return
        text = json.loads(recognizer.FinalResult()).get("text", "").strip()
        tts(f"Command {text} acknowledged.", 175)
        
    print(f"Recognized text: {text or '[nothing understood]'}")


# vosk_stt(device_name=None, timeout=8.0)
