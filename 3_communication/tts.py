import pyttsx3


def tts(text: str, rate: int) -> None:
    print(f"Speaking: {text}")
    engine = pyttsx3.init("sapi5")
    engine.setProperty("rate", rate)
    engine.say(text)
    engine.runAndWait()
    engine.stop()