"""Turning speech or typed text into a command, a spoken reply and UI updates."""

from collections.abc import Callable
from typing import Protocol

import numpy as np

from jarvis.commands import CommandRegistry
from jarvis.confirmation import is_clear_yes
from jarvis.listener import AudioSource, NoSpeech, Utterance, UtteranceRecorder
from jarvis.ui_bridge import ACTIVE, NULL_PRESENTER, WARN, Controller, Orb, Presenter

WAKE_PROMPT = 'Say "Hey Jarvis" followed by a command (Ctrl+C to quit).'


class Transcribes(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class Speaks(Protocol):
    def say(self, text: str) -> None: ...


class Assistant:
    """Transcribes a spoken command, runs it and speaks the reply."""

    def __init__(
        self,
        registry: CommandRegistry,
        transcriber: Transcribes,
        speaker: Speaks,
        presenter: Presenter = NULL_PRESENTER,
    ) -> None:
        self._registry = registry
        self._transcriber = transcriber
        self.speaker = speaker
        self._presenter = presenter

    def respond(self, audio: np.ndarray) -> None:
        self._presenter.state(Orb.THINKING)
        text = self._transcriber.transcribe(audio)
        print(f"You said: {text!r}")
        self.respond_to_text(text)

    def respond_to_text(self, text: str) -> None:
        """Typed commands and quick actions come straight here, through the same
        registry (whitelist, negation, confirmations) as speech."""
        presenter = self._presenter
        presenter.state(Orb.THINKING)
        presenter.phrase(text)
        reply = self._registry.respond(text)
        if reply.brain_online is not None:
            presenter.module("brain", ACTIVE if reply.brain_online else WARN,
                             "ONLINE" if reply.brain_online else "OFFLINE")
        presenter.exchange(text, reply.text, reply.tag)
        if reply.text:
            self.say(reply.text)
        else:
            print("Jarvis: (done)")  # media keys reply silently
        presenter.state(Orb.IDLE)

    def say(self, text: str) -> None:
        self._presenter.state(Orb.SPEAKING)
        self._presenter.phrase(text)
        self.speaker.say(text)


class WakeWordResponder:
    def __init__(
        self, assistant: Assistant, beep: Callable[[], None], presenter: Presenter = NULL_PRESENTER
    ) -> None:
        self._assistant = assistant
        self._beep = beep
        self._presenter = presenter

    def on_wake(self) -> None:
        print("Listening...")
        self._presenter.state(Orb.WAKE)
        self._beep()
        self._presenter.state(Orb.LISTENING)

    def on_utterance(self, audio: np.ndarray) -> None:
        self._assistant.respond(audio)
        print(f"\n{WAKE_PROMPT}")

    def on_no_speech(self) -> None:
        self._presenter.state(Orb.IDLE)
        print(f"No speech heard.\n\n{WAKE_PROMPT}")


class MicState(Protocol):
    @property
    def is_open(self) -> bool: ...


class UiConfirmer:
    """confirm(question) for the UI: shows and speaks the question, then takes a
    typed answer (input box or YES/NO buttons) or, if the mic is on, a spoken one,
    whichever comes first."""

    def __init__(
        self,
        assistant: Assistant,
        transcriber: Transcribes,
        controller: Controller,
        presenter: Presenter,
        mic: AudioSource,
        mic_state: MicState,
        recorder: UtteranceRecorder,
        beep: Callable[[], None],
        typed_timeout_seconds: float,
    ) -> None:
        self._assistant = assistant
        self._transcriber = transcriber
        self._controller = controller
        self._presenter = presenter
        self._mic = mic
        self._mic_state = mic_state
        self._recorder = recorder
        self._beep = beep
        self._typed_timeout = typed_timeout_seconds

    def __call__(self, question: str) -> bool:
        self._controller.begin_confirm()
        self._presenter.confirming(question)
        try:
            self._assistant.say(question)
            answer, typed = self._answer()
        finally:
            self._controller.end_confirm()
            self._presenter.confirming(None)
            self._presenter.state(Orb.THINKING)
        if answer is None:
            print("No answer heard.")
            return False
        print(f"You answered: {answer!r}")
        return is_clear_yes(answer, typed=typed)

    def _answer(self) -> tuple[str | None, bool]:
        self._presenter.state(Orb.LISTENING)
        if self._mic_state.is_open:
            self._beep()
            self._mic.discard_pending()  # drop the question and beep picked up by the mic
            self._recorder.start()
            while True:
                typed = self._controller.typed_answer(timeout=0)
                if typed is not None:
                    return typed, True
                event = self._recorder.feed(self._mic.read())
                if isinstance(event, Utterance):
                    self._presenter.state(Orb.THINKING)
                    return self._transcriber.transcribe(event.audio), False
                if isinstance(event, NoSpeech):
                    break  # nothing said: still accept a typed answer below
        return self._controller.typed_answer(self._typed_timeout), True
