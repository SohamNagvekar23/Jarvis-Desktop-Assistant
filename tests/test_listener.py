from collections import deque
from dataclasses import replace
from types import TracebackType

import numpy as np
import pytest

from jarvis.listener import (
    Listener,
    ListenSettings,
    NoSpeech,
    State,
    Utterance,
    WakeWordHeard,
    listen,
    listen_forever,
    record_utterance,
    RecordSettings,
    UtteranceRecorder,
)

RATE = 16_000
CHUNK = 1280  # 80 ms

# Fake audio: every sample of a chunk holds one marker value, which the fake
# detector and fake VAD read back.
SILENCE, SPEECH, WAKE = 0, 2000, 1000


def chunk(kind: int, samples: int = CHUNK) -> np.ndarray:
    return np.full(samples, kind, dtype=np.int16)


def chunks(kind: int, count: int) -> list[np.ndarray]:
    return [chunk(kind) for _ in range(count)]


class FakeDetector:
    def __init__(self, wake_score: float = 0.9) -> None:
        self.wake_score = wake_score
        self.scored: list[int] = []
        self.resets = 0
        self.fail_next = False

    def score(self, audio: np.ndarray) -> float:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("model blew up")
        self.scored.append(int(audio[0]))
        return self.wake_score if audio[0] == WAKE else 0.01

    def reset(self) -> None:
        self.resets += 1


class FakeVad:
    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0

    def speech_probability(self, audio: np.ndarray) -> float:
        self.calls += 1
        return 0.95 if audio[0] == SPEECH else 0.05

    def reset(self) -> None:
        self.resets += 1


SETTINGS = ListenSettings(
    sample_rate=RATE,
    wake_threshold=0.5,
    vad_threshold=0.5,
    end_silence_seconds=0.8,  # 10 chunks
    max_record_seconds=8.0,  # 100 chunks
    no_speech_timeout_seconds=4.0,  # 50 chunks
)


def make_listener(
    settings: ListenSettings = SETTINGS, detector: FakeDetector | None = None
) -> tuple[Listener, FakeDetector, FakeVad]:
    detector = detector or FakeDetector()
    vad = FakeVad()
    return Listener(detector, vad, settings), detector, vad


def feed_all(listener: Listener, audio: list[np.ndarray]) -> list[tuple[int, object]]:
    """Feed chunks; return (index, event) for each chunk that produced an event."""
    return [(i, e) for i, c in enumerate(audio) if (e := listener.feed(c)) is not None]


# --- Listener state machine ---------------------------------------------------


def test_waiting_ignores_everything_but_the_wake_word() -> None:
    listener, detector, vad = make_listener()
    assert feed_all(listener, chunks(SILENCE, 5) + chunks(SPEECH, 5)) == []
    assert listener.state is State.WAITING
    assert len(detector.scored) == 10
    assert vad.calls == 0  # VAD only runs while recording


@pytest.mark.parametrize(("score", "fires"), [(0.49, False), (0.5, True), (0.99, True)])
def test_wake_threshold(score: float, fires: bool) -> None:
    listener, _, vad = make_listener(detector=FakeDetector(wake_score=score))
    event = listener.feed(chunk(WAKE))
    assert (event == WakeWordHeard(score)) is fires
    assert listener.state is (State.RECORDING if fires else State.WAITING)
    assert vad.resets == (1 if fires else 0)


def test_wake_word_not_scored_while_recording() -> None:
    listener, detector, _ = make_listener()
    listener.feed(chunk(WAKE))
    assert feed_all(listener, chunks(WAKE, 5)) == []  # treated as plain non-speech audio
    assert detector.scored == [WAKE]


def test_utterance_ends_after_0_8_s_of_silence() -> None:
    listener, detector, _ = make_listener()
    listener.feed(chunk(WAKE))
    events = feed_all(listener, chunks(SPEECH, 12) + chunks(SILENCE, 10) + chunks(SPEECH, 3))

    assert len(events) == 1
    index, event = events[0]
    assert index == 21  # the 10th silent chunk, not the 9th
    assert isinstance(event, Utterance)
    assert event.audio.dtype == np.float32
    assert len(event.audio) == 22 * CHUNK
    assert event.audio[0] == pytest.approx(SPEECH / 32768)
    assert event.audio[-1] == 0.0
    assert listener.state is State.WAITING
    assert detector.resets == 1  # stale wake-word features cleared


def test_short_pauses_do_not_end_the_utterance() -> None:
    listener, _, _ = make_listener()
    listener.feed(chunk(WAKE))
    audio = chunks(SPEECH, 5) + chunks(SILENCE, 9) + chunks(SPEECH, 5) + chunks(SILENCE, 10)
    events = feed_all(listener, audio)
    assert [i for i, _ in events] == [len(audio) - 1]
    utterance = events[0][1]
    assert isinstance(utterance, Utterance)
    assert len(utterance.audio) == len(audio) * CHUNK


def test_no_speech_timeout_after_4_s() -> None:
    listener, detector, _ = make_listener()
    listener.feed(chunk(WAKE))
    events = feed_all(listener, chunks(SILENCE, 60))
    assert events == [(49, NoSpeech())]  # 50 chunks x 80 ms = 4 s
    assert listener.state is State.WAITING
    assert detector.resets == 1


def test_speech_starting_just_before_the_timeout_is_recorded() -> None:
    listener, _, _ = make_listener()
    listener.feed(chunk(WAKE))
    audio = chunks(SILENCE, 49) + chunks(SPEECH, 3) + chunks(SILENCE, 10)
    events = feed_all(listener, audio)
    assert [i for i, _ in events] == [len(audio) - 1]
    assert isinstance(events[0][1], Utterance)


def test_recording_stops_at_8_s_max() -> None:
    listener, _, _ = make_listener()
    listener.feed(chunk(WAKE))
    events = feed_all(listener, chunks(SPEECH, 150))
    assert [i for i, _ in events] == [99]
    utterance = events[0][1]
    assert isinstance(utterance, Utterance)
    assert len(utterance.audio) == 8 * RATE
    assert listener.state is State.WAITING


def test_last_chunk_is_truncated_at_the_max() -> None:
    settings = replace(SETTINGS, max_record_seconds=0.1)
    listener, _, _ = make_listener(settings)
    listener.feed(chunk(WAKE))
    assert listener.feed(chunk(SPEECH)) is None
    event = listener.feed(chunk(SPEECH))
    assert isinstance(event, Utterance)
    assert len(event.audio) == 1600


def test_silence_at_the_max_without_speech_is_no_speech() -> None:
    settings = replace(SETTINGS, max_record_seconds=2.0)
    listener, _, _ = make_listener(settings)
    listener.feed(chunk(WAKE))
    assert feed_all(listener, chunks(SILENCE, 30)) == [(24, NoSpeech())]


def test_buffer_is_reused_between_recordings() -> None:
    listener, _, vad = make_listener()
    utterances = []
    for _ in range(2):
        listener.feed(chunk(WAKE))
        events = feed_all(listener, chunks(SPEECH, 2) + chunks(SILENCE, 10))
        utterances.append(events[0][1])
    first, second = utterances
    assert isinstance(first, Utterance) and isinstance(second, Utterance)
    assert np.shares_memory(first.audio, second.audio)  # no per-recording allocation
    assert vad.resets == 2  # fresh VAD state for each recording


def test_reset_returns_to_waiting() -> None:
    listener, detector, _ = make_listener()
    listener.feed(chunk(WAKE))
    listener.reset()
    assert listener.state is State.WAITING
    assert detector.resets == 1
    assert listener.feed(chunk(SPEECH)) is None


# --- listen() loop with a fake microphone ---------------------------------------


class StopListening(BaseException):
    """Ends a test run; BaseException so the loop's error recovery can't swallow it."""


class FakeSource:
    """Scripted audio, plus 'pending' audio that arrives while a handler is busy."""

    def __init__(self, script: list[np.ndarray], fail_after: int | None = None) -> None:
        self.script = deque(script)
        self.pending: list[np.ndarray] = []
        self.discards = 0
        self.discarded = 0
        self.reads = 0
        self.fail_after = fail_after
        self.closed = False

    def __enter__(self) -> "FakeSource":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.closed = True

    def read(self) -> np.ndarray:
        if self.fail_after is not None and self.reads >= self.fail_after:
            raise OSError("microphone unplugged")
        self.reads += 1
        if self.pending:
            return self.pending.pop(0)
        if not self.script:
            raise StopListening
        return self.script.popleft()

    def discard_pending(self) -> None:
        self.discards += 1
        self.discarded += len(self.pending)
        self.pending.clear()


class FakeResponder:
    def __init__(self, source: FakeSource) -> None:
        self.source = source
        self.events: list[str] = []
        self.audio_lengths: list[int] = []
        self.fail_on: set[str] = set()

    def _maybe_fail(self, name: str) -> None:
        if name in self.fail_on:
            self.fail_on.discard(name)
            raise RuntimeError(f"{name} failed")

    def on_wake(self) -> None:
        self.events.append("wake")
        # The beep is picked up by the mic while it plays.
        self.source.pending.extend(chunks(SPEECH, 2))
        self._maybe_fail("wake")

    def on_utterance(self, audio: np.ndarray) -> None:
        self.events.append("utterance")
        self.audio_lengths.append(len(audio))
        # Transcribing and speaking the reply takes a while; the mic keeps
        # capturing, including a reply that sounds like the wake word.
        self.source.pending.extend(chunks(WAKE, 3) + chunks(SPEECH, 20))
        self._maybe_fail("utterance")

    def on_no_speech(self) -> None:
        self.events.append("no_speech")


def command() -> list[np.ndarray]:
    """'Hey Jarvis', then a 0.4 s command, then 0.8 s of silence."""
    return [chunk(WAKE)] + chunks(SPEECH, 5) + chunks(SILENCE, 10)


def run(source: FakeSource, listener: Listener, responder: FakeResponder) -> None:
    with pytest.raises(StopListening):
        listen(source, listener, responder)


def test_listen_runs_wake_record_respond() -> None:
    listener, detector, _ = make_listener()
    source = FakeSource(chunks(SILENCE, 3) + command() + chunks(SILENCE, 3))
    responder = FakeResponder(source)
    run(source, listener, responder)
    assert responder.events == ["wake", "utterance"]
    assert responder.audio_lengths == [15 * CHUNK]  # the beep audio was not recorded


def test_audio_queued_while_busy_is_discarded() -> None:
    listener, detector, _ = make_listener()
    source = FakeSource(command() + command())
    responder = FakeResponder(source)
    run(source, listener, responder)

    # Two full commands, no extra wake from the WAKE-like audio queued during replies.
    assert responder.events == ["wake", "utterance", "wake", "utterance"]
    assert source.discarded == 2 * (2 + 23)  # beep audio + audio during each reply
    assert detector.scored.count(WAKE) == 2  # the stale wake chunks never reached it


def test_no_speech_then_listening_again() -> None:
    listener, _, _ = make_listener()
    source = FakeSource([chunk(WAKE)] + chunks(SILENCE, 50) + command())
    responder = FakeResponder(source)
    run(source, listener, responder)
    assert responder.events == ["wake", "no_speech", "wake", "utterance"]


@pytest.mark.parametrize("failing", ["wake", "utterance"])
def test_recovers_from_handler_errors(failing: str, caplog: pytest.LogCaptureFixture) -> None:
    listener, _, _ = make_listener()
    source = FakeSource(command() + command())
    responder = FakeResponder(source)
    responder.fail_on.add(failing)
    run(source, listener, responder)

    assert "Error while handling audio" in caplog.text
    assert responder.events[-2:] == ["wake", "utterance"]  # second command worked
    assert listener.state is State.WAITING


def test_failed_wake_handler_does_not_record() -> None:
    listener, _, _ = make_listener()
    source = FakeSource(command())
    responder = FakeResponder(source)
    responder.fail_on.add("wake")
    run(source, listener, responder)
    assert responder.events == ["wake"]  # back to waiting; the command audio is ignored


def test_recovers_from_detector_errors(caplog: pytest.LogCaptureFixture) -> None:
    detector = FakeDetector()
    detector.fail_next = True
    listener, _, _ = make_listener(detector=detector)
    source = FakeSource([chunk(SILENCE)] + command())
    responder = FakeResponder(source)
    run(source, listener, responder)
    assert "model blew up" in caplog.text
    assert responder.events == ["wake", "utterance"]


def test_keyboard_interrupt_is_not_swallowed() -> None:
    listener, _, _ = make_listener()
    source = FakeSource(command())
    responder = FakeResponder(source)

    def interrupt(audio: np.ndarray) -> None:
        raise KeyboardInterrupt

    responder.on_utterance = interrupt  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        listen(source, listener, responder)


# --- listen_forever(): reopening the microphone --------------------------------


def test_listen_forever_reopens_the_source_after_errors(caplog: pytest.LogCaptureFixture) -> None:
    listener, detector, _ = make_listener()
    # 1st open fails; 2nd source dies mid-recording; 3rd works.
    dying = FakeSource([chunk(WAKE)] + chunks(SPEECH, 3), fail_after=4)
    healthy = FakeSource(command())
    opens: list[object] = [OSError("no microphone"), dying, healthy]
    responder = FakeResponder(healthy)
    sleeps: list[float] = []

    def open_source() -> FakeSource:
        item = opens.pop(0)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, FakeSource)
        return item

    with pytest.raises(StopListening):
        listen_forever(open_source, listener, responder, retry_seconds=2.0, sleep=sleeps.append)

    assert sleeps == [2.0, 2.0]
    assert dying.closed and healthy.closed
    assert "no microphone" in caplog.text and "microphone unplugged" in caplog.text
    # The half-finished recording from the dead stream was dropped, and the
    # next "Hey Jarvis" on the new stream worked normally.
    assert responder.events == ["wake", "wake", "utterance"]
    assert responder.audio_lengths == [15 * CHUNK]


# --- record_utterance(): the yes/no answer after a confirmation question ---------

ANSWER_SETTINGS = RecordSettings(
    sample_rate=RATE,
    vad_threshold=0.5,
    end_silence_seconds=0.6,  # 7.5 -> 8 chunks
    max_record_seconds=3.0,  # 37.5 chunks
    no_speech_timeout_seconds=4.0,
)


def test_record_utterance_needs_no_wake_word() -> None:
    vad = FakeVad()
    source = FakeSource(chunks(SPEECH, 4) + chunks(SILENCE, 8) + chunks(SPEECH, 5))
    audio = record_utterance(source, UtteranceRecorder(vad, ANSWER_SETTINGS))
    assert audio is not None
    assert len(audio) == 12 * CHUNK
    assert vad.resets == 1
    assert len(source.script) == 5  # stopped reading once the answer ended


def test_record_utterance_returns_none_when_nobody_answers() -> None:
    source = FakeSource(chunks(SILENCE, 100))
    # Silence until the 3 s maximum (before the 4 s no-speech timeout).
    assert record_utterance(source, UtteranceRecorder(FakeVad(), ANSWER_SETTINGS)) is None
    assert source.reads == 38


def test_answer_recorder_does_not_overwrite_the_command_audio() -> None:
    listener, _, vad = make_listener()
    listener.feed(chunk(WAKE))
    events = feed_all(listener, chunks(SPEECH, 2) + chunks(SILENCE, 10))
    command_audio = events[0][1]
    assert isinstance(command_audio, Utterance)
    answer = record_utterance(
        FakeSource(chunks(SILENCE, 1) + chunks(SPEECH, 2) + chunks(SILENCE, 8)),
        UtteranceRecorder(vad, ANSWER_SETTINGS),
    )
    assert answer is not None
    assert not np.shares_memory(command_audio.audio, answer)
    assert command_audio.audio[0] == pytest.approx(SPEECH / 32768)
