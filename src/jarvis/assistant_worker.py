"""The assistant's background loop when the UI is running."""

import logging
from collections.abc import Callable
from typing import Protocol

from jarvis.assistant import Assistant
from jarvis.listener import AudioSource, HoldRecorder, Listener, Responder, listen
from jarvis.ui_bridge import ACTIVE, ERROR, IDLE_LEVEL, Controller, Orb, Presenter

logger = logging.getLogger(__name__)


class Mic(AudioSource, Protocol):
    """A reopenable microphone: each `with mic:` opens a fresh stream."""

    @property
    def is_open(self) -> bool: ...

    def __enter__(self) -> "Mic": ...

    def __exit__(self, *exc_info: object) -> None: ...


class AssistantWorker:
    """Wake word listening (unless muted), typed commands and hold-to-talk,
    all handled one at a time on this thread."""

    def __init__(
        self,
        *,
        assistant: Assistant,
        controller: Controller,
        presenter: Presenter,
        mic: Mic,
        listener: Listener,
        responder: Responder,
        hold_recorder: HoldRecorder,
        min_hold_samples: int,
        retry_seconds: float,
        on_quit: Callable[[], None] = lambda: None,
    ) -> None:
        self._assistant = assistant
        self._controller = controller
        self._presenter = presenter
        self._mic = mic
        self._listener = listener
        self._responder = responder
        self._hold = hold_recorder
        self._min_hold_samples = min_hold_samples
        self._retry_seconds = retry_seconds
        self._on_quit = on_quit

    def run(self) -> None:
        controller = self._controller
        while not controller.stopped:
            if controller.muted:
                self._presenter.module("microphone", IDLE_LEVEL, "MUTED")
                self._presenter.state(Orb.IDLE)
                self._safely(lambda: self._handle_request(timeout=0.25))
                continue
            try:
                with self._mic:
                    self._presenter.module("microphone", ACTIVE, "LISTENING")
                    self._presenter.state(Orb.IDLE)
                    listen(
                        self._mic,
                        self._listener,
                        self._responder,
                        between_chunks=lambda: self._handle_request(timeout=0),
                        keep_listening=lambda: not controller.muted and not controller.stopped,
                    )
            except Exception:
                logger.exception("Microphone failed; retrying in %g s", self._retry_seconds)
                self._presenter.module("microphone", ERROR, "NO MIC")
                self._listener.reset()
                controller.wait_stopped(self._retry_seconds)
        self._on_quit()

    def _handle_request(self, timeout: float) -> bool:
        """Handle one UI request if there is one; True if something was handled."""
        request = self._controller.next_request(timeout)
        if request is None:
            return False
        kind, text = request
        if kind == "text":
            self._assistant.respond_to_text(text)
        elif kind == "hold":
            self._hold_to_talk()
        return True  # "quit" also counts: the loop then sees controller.stopped

    def _hold_to_talk(self) -> None:
        self._presenter.state(Orb.LISTENING)
        self._hold.start()
        if self._mic.is_open:
            self._record_while_held()
        else:  # muted: open the mic just for this
            with self._mic:
                self._record_while_held()
        audio = self._hold.audio
        if len(audio) < self._min_hold_samples:  # a tap, not a phrase
            self._presenter.state(Orb.IDLE)
            return
        self._assistant.respond(audio)

    def _record_while_held(self) -> None:
        self._mic.discard_pending()
        while self._controller.holding and self._hold.feed(self._mic.read()):
            pass

    def _safely(self, action: Callable[[], object]) -> None:
        try:
            action()
        except Exception:
            logger.exception("Error while handling a request")
            self._presenter.state(Orb.IDLE)
