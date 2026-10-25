"""Spoken replies with pyttsx3 (Windows SAPI5)."""

import pyttsx3


class Speaker:
    """Initializes the TTS engine once and reuses it."""

    def __init__(self, rate: int) -> None:
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)

    def say(self, text: str) -> None:
        print(f"Jarvis: {text}")
        self._engine.say(text)
        self._engine.runAndWait()
