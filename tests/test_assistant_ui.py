"""The assistant side of the UI, driven with a fake UI (a real EventBus) and fakes."""

import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import pytest

from jarvis.assistant import Assistant, UiConfirmer, WakeWordResponder
from jarvis.assistant_worker import AssistantWorker
from jarvis.commands import Reply
from jarvis.listener import HoldRecorder, Listener, ListenSettings, RecordSettings, UtteranceRecorder
from jarvis.ui_bridge import Controller, EventBus
from tests.test_commands import FakeDesktop

RATE, CHUNK = 16_000, 1280
SILENCE, SPEECH, WAKE = 0, 2000, 1000


def chunk(kind: int) -> np.ndarray:
    return np.full(CHUNK, kind, dtype=np.int16)


def wait_for(condition: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


class Ui:
    """The fake UI: records every event the bus would send to the pages."""

    def __init__(self) -> None:
        self.bus = EventBus()
        self.events: list[dict] = []
        self._lock = threading.Lock()

    def collect(self) -> list[dict]:
        while (event := self.bus.next_event(timeout=0)) is not None:
            with self._lock:
                self.events.append(event)
        return self.events

    def states(self) -> list[str]:
        return [e["value"]["orb"] for e in self.collect() if e["type"] == "state"]

    def of(self, kind: str) -> list:
        return [e["value"] for e in self.collect() if e["type"] == kind]


class FakeRegistry:
    def __init__(self, replies: dict[str, Reply] | None = None) -> None:
        self.replies = replies or {}
        self.texts: list[str] = []

    def respond(self, text: str) -> Reply:
        self.texts.append(text)
        return self.replies.get(text, Reply(f"Did {text}.", "cmd"))


class FakeTranscriber:
    def __init__(self, *texts: str) -> None:
        self.texts = deque(texts)
        self.lengths: list[int] = []

    def transcribe(self, audio: np.ndarray) -> str:
        self.lengths.append(len(audio))
        return self.texts.popleft() if self.texts else ""


class FakeSpeaker:
    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)


class FakeMic:
    """Scripted chunks, then silence in (fast) real time; tracks opening and closing."""

    def __init__(self, script: list[np.ndarray] | None = None, fail_opens: int = 0) -> None:
        self.script = deque(script or [])
        self.opens = 0
        self.closes = 0
        self._open = False
        self.fail_opens = fail_opens
        self.discards = 0

    @property
    def is_open(self) -> bool:
        return self._open

    def __enter__(self) -> "FakeMic":
        if self.fail_opens:
            self.fail_opens -= 1
            raise OSError("no microphone")
        self.opens += 1
        self._open = True
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closes += 1
        self._open = False

    def read(self) -> np.ndarray:
        assert self._open, "read from a closed mic"
        if self.script:
            return self.script.popleft()
        time.sleep(0.002)
        return chunk(SILENCE)

    def discard_pending(self) -> None:
        self.discards += 1


class FakeDetector:
    def score(self, audio: np.ndarray) -> float:
        return 0.9 if audio[0] == WAKE else 0.0

    def reset(self) -> None:
        pass


class FakeVad:
    def speech_probability(self, audio: np.ndarray) -> float:
        return 0.9 if audio[0] == SPEECH else 0.0

    def reset(self) -> None:
        pass


LISTEN = ListenSettings(
    sample_rate=RATE,
    vad_threshold=0.5,
    end_silence_seconds=0.8,
    max_record_seconds=8.0,
    no_speech_timeout_seconds=4.0,
    wake_threshold=0.5,
)
ANSWER = RecordSettings(
    sample_rate=RATE,
    vad_threshold=0.5,
    end_silence_seconds=0.6,
    max_record_seconds=3.0,
    no_speech_timeout_seconds=1.0,
)


# --- Assistant reports its state ----------------------------------------------------


def test_typed_command_states_phrase_and_log() -> None:
    ui, speaker = Ui(), FakeSpeaker()
    assistant = Assistant(FakeRegistry(), FakeTranscriber(), speaker, ui.bus)
    assistant.respond_to_text("open notepad")
    assert ui.states() == ["thinking", "speaking", "idle"]
    assert ui.of("phrase") == ["open notepad", "Did open notepad."]
    (entry,) = ui.of("log")
    assert (entry["said"], entry["reply"], entry["tag"]) == ("open notepad", "Did open notepad.", "cmd")
    assert speaker.said == ["Did open notepad."]


def test_silent_reply_never_enters_speaking() -> None:
    ui, speaker = Ui(), FakeSpeaker()
    assistant = Assistant(FakeRegistry({"next": Reply("", "cmd")}), FakeTranscriber(), speaker, ui.bus)
    assistant.respond_to_text("next")
    assert ui.states() == ["thinking", "idle"]
    assert speaker.said == []
    assert ui.of("log")[0]["reply"] == ""


@pytest.mark.parametrize(
    ("reply", "module"),
    [
        (Reply("Paris.", "brain", brain_online=True), {"name": "brain", "level": "active", "text": "ONLINE"}),
        (Reply("My brain is offline.", "brain", brain_online=False), {"name": "brain", "level": "warn", "text": "OFFLINE"}),
    ],
)
def test_brain_replies_update_the_brain_module(reply: Reply, module: dict) -> None:
    ui = Ui()
    Assistant(FakeRegistry({"q": reply}), FakeTranscriber(), FakeSpeaker(), ui.bus).respond_to_text("q")
    assert ui.of("module") == [module]
    assert ui.of("log")[0]["tag"] == "brain"


def test_spoken_command_is_transcribed_then_handled_the_same_way() -> None:
    ui, registry = Ui(), FakeRegistry()
    transcriber = FakeTranscriber("volume up")
    Assistant(registry, transcriber, FakeSpeaker(), ui.bus).respond(np.zeros(3200, np.float32))
    assert transcriber.lengths == [3200]
    assert registry.texts == ["volume up"]
    assert ui.states() == ["thinking", "speaking", "idle"]


def test_wake_word_flow_states() -> None:
    ui = Ui()
    assistant = Assistant(FakeRegistry(), FakeTranscriber("open notepad"), FakeSpeaker(), ui.bus)
    beeps: list[int] = []
    responder = WakeWordResponder(assistant, lambda: beeps.append(1), ui.bus)
    responder.on_wake()
    responder.on_utterance(np.zeros(1600, np.float32))
    responder.on_wake()
    responder.on_no_speech()
    assert ui.states() == ["wake", "listening", "thinking", "speaking", "idle", "wake", "listening", "idle"]
    assert beeps == [1, 1]


def test_no_ui_mode_prints_exactly_as_before(capsys: pytest.CaptureFixture[str]) -> None:
    # Without a presenter (--no-ui), the console output is unchanged.
    assistant = Assistant(FakeRegistry(), FakeTranscriber("open notepad"), FakeSpeaker())
    responder = WakeWordResponder(assistant, lambda: None)
    responder.on_wake()
    responder.on_utterance(np.zeros(1600, np.float32))
    responder.on_no_speech()
    assert capsys.readouterr().out == (
        "Listening...\n"
        "You said: 'open notepad'\n"
        '\nSay "Hey Jarvis" followed by a command (Ctrl+C to quit).\n'
        'No speech heard.\n\nSay "Hey Jarvis" followed by a command (Ctrl+C to quit).\n'
    )


# --- UiConfirmer: typed or spoken yes/no ------------------------------------------------


def make_confirmer(ui: Ui, mic: FakeMic, transcriber: FakeTranscriber, timeout: float = 2.0):
    controller = Controller(ui.bus)
    speaker = FakeSpeaker()
    assistant = Assistant(FakeRegistry(), transcriber, speaker, ui.bus)
    confirmer = UiConfirmer(
        assistant,
        transcriber,
        controller,
        ui.bus,
        mic,
        mic,
        UtteranceRecorder(FakeVad(), ANSWER),
        lambda: None,
        timeout,
    )
    return confirmer, controller, speaker


def run_in_thread(function: Callable[[], object]) -> tuple[threading.Thread, list]:
    result: list = []
    thread = threading.Thread(target=lambda: result.append(function()), daemon=True)
    thread.start()
    return thread, result


@pytest.mark.parametrize(("answer", "expected"), [("yes", True), ("Y", True), ("no", False), ("maybe", False)])
def test_typed_answer_with_the_mic_off(answer: str, expected: bool) -> None:
    ui = Ui()
    confirmer, controller, speaker = make_confirmer(ui, FakeMic(), FakeTranscriber())
    thread, result = run_in_thread(lambda: confirmer("Close Spotify? Say yes or no."))
    wait_for(lambda: "Close Spotify? Say yes or no." in ui.of("confirm"))
    controller.submit_text(answer)
    thread.join(timeout=2)
    assert result == [expected]
    assert ui.of("confirm") == ["Close Spotify? Say yes or no.", None]  # shown, then cleared
    assert speaker.said == ["Close Spotify? Say yes or no."]


def test_spoken_answer_with_the_mic_on() -> None:
    ui = Ui()
    mic = FakeMic([chunk(SPEECH)] * 4 + [chunk(SILENCE)] * 8)
    transcriber = FakeTranscriber("Yes.")
    confirmer, _, _ = make_confirmer(ui, mic, transcriber)
    with mic:
        assert confirmer("Close Spotify? Say yes or no.") is True
    assert transcriber.lengths == [12 * CHUNK]
    assert "listening" in ui.states()


def test_spoken_y_does_not_count_but_typed_y_does() -> None:
    ui = Ui()
    mic = FakeMic([chunk(SPEECH)] * 3 + [chunk(SILENCE)] * 8)
    confirmer, _, _ = make_confirmer(ui, mic, FakeTranscriber("Y."))
    with mic:
        assert confirmer("Lock the screen? Say yes or no.") is False


def test_typed_answer_wins_while_listening() -> None:
    ui = Ui()
    mic = FakeMic()  # silence forever (until the no-speech timeout)
    transcriber = FakeTranscriber()
    confirmer, controller, _ = make_confirmer(ui, mic, transcriber)
    with mic:
        thread, result = run_in_thread(lambda: confirmer("Close Chrome? Say yes or no."))
        wait_for(lambda: "Close Chrome? Say yes or no." in ui.of("confirm"))
        controller.submit_text("yes")
        thread.join(timeout=2)
    assert result == [True]
    assert transcriber.lengths == []  # nothing was transcribed


def test_no_answer_at_all_is_no(capsys: pytest.CaptureFixture[str]) -> None:
    ui = Ui()
    mic = FakeMic()
    confirmer, _, _ = make_confirmer(ui, mic, FakeTranscriber(), timeout=0.05)
    with mic:
        assert confirmer("Close Chrome? Say yes or no.") is False
    assert "No answer heard." in capsys.readouterr().out
    assert ui.of("confirm")[-1] is None


# --- AssistantWorker: the background loop ------------------------------------------------


class Harness:
    def __init__(self, mic: FakeMic, registry=None, transcriber: FakeTranscriber | None = None) -> None:
        self.ui = Ui()
        self.mic = mic
        self.controller = Controller(self.ui.bus)
        self.registry = registry or FakeRegistry()
        self.transcriber = transcriber or FakeTranscriber()
        self.speaker = FakeSpeaker()
        self.assistant = Assistant(self.registry, self.transcriber, self.speaker, self.ui.bus)
        self.quit_called = threading.Event()
        self.worker = AssistantWorker(
            assistant=self.assistant,
            controller=self.controller,
            presenter=self.ui.bus,
            mic=mic,
            listener=Listener(FakeDetector(), FakeVad(), LISTEN),
            responder=WakeWordResponder(self.assistant, lambda: None, self.ui.bus),
            hold_recorder=HoldRecorder(RATE, 15.0),
            min_hold_samples=round(RATE * 0.3),
            retry_seconds=0.05,
            on_quit=self.quit_called.set,
        )
        self.thread = threading.Thread(target=self.worker.run, daemon=True)

    def __enter__(self) -> "Harness":
        self.thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.controller.quit()
        self.thread.join(timeout=3)
        assert not self.thread.is_alive(), "worker didn't stop"
        assert self.quit_called.is_set()


def test_typed_commands_are_handled_while_listening() -> None:
    with Harness(FakeMic()) as h:
        wait_for(lambda: h.mic.is_open)
        h.controller.submit_text("open notepad")
        wait_for(lambda: h.speaker.said == ["Did open notepad."])
        assert h.mic.is_open  # still listening for the wake word
    assert h.registry.texts == ["open notepad"]
    assert {"name": "microphone", "level": "active", "text": "LISTENING"} in h.ui.of("module")


def test_wake_word_command_through_the_worker() -> None:
    script = [chunk(WAKE)] + [chunk(SPEECH)] * 5 + [chunk(SILENCE)] * 10
    with Harness(FakeMic(script), transcriber=FakeTranscriber("take a screenshot")) as h:
        wait_for(lambda: h.speaker.said == ["Did take a screenshot."])
    states = h.ui.states()
    assert states[states.index("wake"):states.index("wake") + 5] == [
        "wake", "listening", "thinking", "speaking", "idle"
    ]


def test_mute_closes_the_mic_and_unmute_reopens_it() -> None:
    with Harness(FakeMic()) as h:
        wait_for(lambda: h.mic.is_open)
        h.controller.toggle_mute()
        wait_for(lambda: not h.mic.is_open)
        assert {"name": "microphone", "level": "idle", "text": "MUTED"} in h.ui.of("module")
        h.controller.submit_text("volume up")  # typed commands still work while muted
        wait_for(lambda: h.speaker.said == ["Did volume up."])
        assert not h.mic.is_open
        h.controller.toggle_mute()
        wait_for(lambda: h.mic.is_open)
    assert h.mic.opens == 2
    assert h.ui.of("muted") == [False, True, False]


def test_wake_words_are_not_heard_while_muted() -> None:
    mic = FakeMic()
    with Harness(mic) as h:
        h.controller.set_muted(True)
        wait_for(lambda: not mic.is_open and h.mic.opens <= 1)
        mic.script.extend([chunk(WAKE)] + [chunk(SPEECH)] * 3 + [chunk(SILENCE)] * 10)
        time.sleep(0.2)
    assert "wake" not in h.ui.states()
    assert len(mic.script) == 14  # never read


def test_hold_to_talk_records_until_released_even_when_muted() -> None:
    with Harness(FakeMic(), transcriber=FakeTranscriber("open chrome")) as h:
        h.controller.set_muted(True)
        wait_for(lambda: not h.mic.is_open)
        h.controller.hold_start()
        wait_for(lambda: h.mic.is_open)  # opened just for the hold
        h.mic.script.extend([chunk(SPEECH)] * 10)
        wait_for(lambda: not h.mic.script)
        h.controller.hold_end()
        wait_for(lambda: h.speaker.said == ["Did open chrome."])
        wait_for(lambda: not h.mic.is_open)  # closed again: still muted
    assert h.transcriber.lengths and h.transcriber.lengths[0] >= 10 * CHUNK
    assert "listening" in h.ui.states()


def test_a_tap_on_hold_to_talk_is_ignored() -> None:
    with Harness(FakeMic()) as h:
        h.controller.set_muted(True)
        wait_for(lambda: not h.mic.is_open)
        h.controller.hold_start()
        h.controller.hold_end()  # released before anything was recorded
        time.sleep(0.2)
    assert h.transcriber.lengths == []
    assert h.registry.texts == []


def test_microphone_failure_is_reported_and_retried() -> None:
    with Harness(FakeMic(fail_opens=2)) as h:
        wait_for(lambda: h.mic.is_open)
    assert {"name": "microphone", "level": "error", "text": "NO MIC"} in h.ui.of("module")
    assert h.ui.of("module")[-1]["text"] == "LISTENING"


def test_errors_while_handling_a_request_dont_stop_the_worker() -> None:
    class Exploding(FakeRegistry):
        def respond(self, text: str) -> Reply:
            if text == "boom":
                raise RuntimeError("bug")
            return super().respond(text)

    with Harness(FakeMic(), registry=Exploding()) as h:
        h.controller.set_muted(True)
        h.controller.submit_text("boom")
        h.controller.submit_text("open notepad")
        wait_for(lambda: h.speaker.said == ["Did open notepad."])


def test_close_app_confirmation_end_to_end_with_a_typed_yes() -> None:
    """Typed "close spotify" -> question shown in the UI -> typed "yes" -> closed."""
    desktop = FakeDesktop()
    desktop.windows.open.add("Spotify")
    mic = FakeMic()
    ui = Ui()
    controller = Controller(ui.bus)
    speaker = FakeSpeaker()
    holder: list[UiConfirmer] = []
    desktop.confirm = lambda question: holder[0](question)  # type: ignore[method-assign]
    registry = desktop.registry()
    assistant = Assistant(registry, FakeTranscriber(), speaker, ui.bus)
    holder.append(
        UiConfirmer(assistant, FakeTranscriber(), controller, ui.bus, mic, mic,
                    UtteranceRecorder(FakeVad(), ANSWER), lambda: None, 2.0)
    )
    worker = AssistantWorker(
        assistant=assistant, controller=controller, presenter=ui.bus, mic=mic,
        listener=Listener(FakeDetector(), FakeVad(), LISTEN),
        responder=WakeWordResponder(assistant, lambda: None, ui.bus),
        hold_recorder=HoldRecorder(RATE, 15.0), min_hold_samples=4800, retry_seconds=0.05,
    )
    controller.set_muted(True)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        controller.submit_text("Close Spotify.")
        wait_for(lambda: "Close Spotify? Say yes or no." in ui.of("confirm"))
        controller.submit_text("yes")
        wait_for(lambda: "Closed Spotify." in speaker.said)
    finally:
        controller.quit()
        thread.join(timeout=3)
    assert desktop.windows.calls == [("close", "Spotify")]
    (entry,) = ui.of("log")
    assert (entry["said"], entry["reply"], entry["tag"]) == ("Close Spotify.", "Closed Spotify.", "cmd")
    assert ui.of("confirm") == ["Close Spotify? Say yes or no.", None]
