"""Always-listening loop: wake word -> record until silence -> hand off.

Pure logic: the microphone, wake-word model, VAD and responses are passed in,
so tests drive it with fake audio and a fake detector.
"""

import logging
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

_INT16_SCALE = 1 / 32768


class WakeWordDetector(Protocol):
    def score(self, chunk: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class SpeechDetector(Protocol):
    def speech_probability(self, chunk: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class AudioSource(Protocol):
    def read(self) -> np.ndarray:
        """Block for the next int16 mono chunk; raise OSError if the stream died."""
        ...

    def discard_pending(self) -> None:
        """Drop audio that queued up while we were busy."""
        ...


class Responder(Protocol):
    def on_wake(self) -> None: ...
    def on_utterance(self, audio: np.ndarray) -> None: ...
    def on_no_speech(self) -> None: ...


class State(Enum):
    WAITING = auto()  # scoring chunks for the wake word
    RECORDING = auto()  # collecting the command after the wake word


@dataclass(frozen=True)
class WakeWordHeard:
    score: float


@dataclass(frozen=True)
class Utterance:
    # float32 in [-1, 1) at the sample rate; a view of the listener's buffer,
    # valid until the next recording starts.
    audio: np.ndarray


@dataclass(frozen=True)
class NoSpeech:
    pass


Event = WakeWordHeard | Utterance | NoSpeech


@dataclass(frozen=True)
class RecordSettings:
    sample_rate: int
    vad_threshold: float
    end_silence_seconds: float
    max_record_seconds: float
    no_speech_timeout_seconds: float


@dataclass(frozen=True)
class ListenSettings(RecordSettings):
    wake_threshold: float


class UtteranceRecorder:
    """Records one utterance from chunks, ending it with VAD."""

    def __init__(self, vad: SpeechDetector, settings: RecordSettings) -> None:
        self._vad = vad
        self._vad_threshold = settings.vad_threshold
        rate = settings.sample_rate
        self._end_silence = round(rate * settings.end_silence_seconds)
        self._no_speech_timeout = round(rate * settings.no_speech_timeout_seconds)
        self._max_samples = round(rate * settings.max_record_seconds)
        # Allocated once; each recording is converted to float32 straight into it.
        self._buffer = np.empty(self._max_samples, dtype=np.float32)
        self._recorded = 0
        self._silence = 0
        self._heard_speech = False

    def start(self) -> None:
        self._recorded = 0
        self._silence = 0
        self._heard_speech = False
        self._vad.reset()

    def feed(self, chunk: np.ndarray) -> Utterance | NoSpeech | None:
        """Add a chunk; returns the finished utterance, NoSpeech, or None to continue."""
        count = min(len(chunk), self._max_samples - self._recorded)
        segment = self._buffer[self._recorded : self._recorded + count]
        segment[:] = chunk[:count]
        segment *= _INT16_SCALE
        self._recorded += count

        if self._vad.speech_probability(chunk) >= self._vad_threshold:
            self._heard_speech = True
            self._silence = 0
        else:
            self._silence += len(chunk)

        full = self._recorded >= self._max_samples
        if self._heard_speech:
            if full or self._silence >= self._end_silence:
                return Utterance(self._buffer[: self._recorded])
        elif full or self._recorded >= self._no_speech_timeout:
            return NoSpeech()
        return None


class HoldRecorder:
    """Records for as long as a button is held (no VAD), up to a maximum."""

    def __init__(self, sample_rate: int, max_seconds: float) -> None:
        self._buffer = np.empty(round(sample_rate * max_seconds), dtype=np.float32)
        self._recorded = 0

    def start(self) -> None:
        self._recorded = 0

    def feed(self, chunk: np.ndarray) -> bool:
        """Add a chunk; False once the buffer is full."""
        count = min(len(chunk), len(self._buffer) - self._recorded)
        segment = self._buffer[self._recorded : self._recorded + count]
        segment[:] = chunk[:count]
        segment *= _INT16_SCALE
        self._recorded += count
        return self._recorded < len(self._buffer)

    @property
    def audio(self) -> np.ndarray:
        """A view of what was recorded, valid until the next start()."""
        return self._buffer[: self._recorded]


def record_utterance(source: AudioSource, recorder: UtteranceRecorder) -> np.ndarray | None:
    """Record from source until the utterance ends; None if nobody spoke."""
    recorder.start()
    while (event := recorder.feed(source.read())) is None:
        pass
    return event.audio if isinstance(event, Utterance) else None


class Listener:
    """Turns a stream of chunks into wake / utterance / no-speech events."""

    def __init__(
        self, detector: WakeWordDetector, vad: SpeechDetector, settings: ListenSettings
    ) -> None:
        self._detector = detector
        self._wake_threshold = settings.wake_threshold
        self._recorder = UtteranceRecorder(vad, settings)
        self.state = State.WAITING

    def feed(self, chunk: np.ndarray) -> Event | None:
        if self.state is State.WAITING:
            score = self._detector.score(chunk)
            if score < self._wake_threshold:
                return None
            self.state = State.RECORDING
            self._recorder.start()
            return WakeWordHeard(score)
        event = self._recorder.feed(chunk)
        if event is not None:
            self.reset()
        return event

    def reset(self) -> None:
        """Go back to waiting for the wake word with fresh model state."""
        self.state = State.WAITING
        self._detector.reset()


def listen(
    source: AudioSource,
    listener: Listener,
    responder: Responder,
    between_chunks: Callable[[], bool] | None = None,
    keep_listening: Callable[[], bool] = lambda: True,
) -> None:
    """Process the stream until keep_listening() is False (by default, forever).

    Chunks are handled one at a time, so no wake word can be detected while a
    command is being handled. Afterwards the audio that queued up meanwhile
    (the beep, Jarvis's own reply) is discarded. between_chunks, if given, runs
    after every chunk and returns True if it handled something (e.g. a typed
    command), which counts like a handled command. An error while handling a
    chunk is logged and listening resumes; errors from the source propagate.
    """
    while keep_listening():
        chunk = source.read()
        try:
            if between_chunks is not None and between_chunks():
                listener.reset()
                source.discard_pending()
                continue
            event = listener.feed(chunk)
            if event is None:
                continue
            if isinstance(event, WakeWordHeard):
                responder.on_wake()
            elif isinstance(event, Utterance):
                responder.on_utterance(event.audio)
            else:
                responder.on_no_speech()
        except Exception:
            logger.exception("Error while handling audio; listening again")
            listener.reset()
        source.discard_pending()


def listen_forever(
    open_source: Callable[[], AbstractContextManager[AudioSource]],
    listener: Listener,
    responder: Responder,
    retry_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run listen(), reopening the audio source after any error."""
    while True:
        try:
            with open_source() as source:
                listen(source, listener, responder)
        except Exception:
            logger.exception("Audio input failed; retrying in %g s", retry_seconds)
            listener.reset()
            sleep(retry_seconds)
