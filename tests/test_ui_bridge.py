import json
import re
import threading
import time
from datetime import datetime, timedelta

import pytest

from jarvis.ui_bridge import (
    ACTIVE,
    ERROR,
    STATUS_LABELS,
    Controller,
    EventBus,
    EventPump,
    NullPresenter,
    Orb,
    to_js,
)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 25, 14, 30)

    def __call__(self) -> datetime:
        return self.now


def drain(bus: EventBus) -> list[dict]:
    events = []
    while (event := bus.next_event(timeout=0)) is not None:
        events.append(event)
    return events


@pytest.fixture
def bus() -> EventBus:
    return EventBus(Clock())


# --- EventBus: only changes are sent -----------------------------------------------


def test_state_events_carry_the_orb_and_label(bus: EventBus) -> None:
    bus.state(Orb.THINKING)
    assert drain(bus) == [{"type": "state", "value": {"orb": "thinking", "label": "PROCESSING"}}]


def test_labels_for_every_state() -> None:
    assert {orb.value: label for orb, label in STATUS_LABELS.items()} == {
        "idle": "STANDBY",
        "wake": "WAKE DETECTED",
        "listening": "LISTENING",
        "thinking": "PROCESSING",
        "speaking": "RESPONDING",
    }


@pytest.mark.parametrize(
    "report",
    [
        lambda b: b.state(Orb.LISTENING),
        lambda b: b.phrase("open notepad"),
        lambda b: b.module("brain", ACTIVE, "ONLINE"),
        lambda b: b.confirming("Close Spotify? Say yes or no."),
        lambda b: b.muted(True),
        lambda b: b.meters({"cpu": {"percent": 12, "text": "12%"}}),
    ],
)
def test_unchanged_values_are_not_resent(bus: EventBus, report) -> None:
    report(bus)
    report(bus)
    report(bus)
    assert len(drain(bus)) == 1


def test_changes_are_sent_in_order(bus: EventBus) -> None:
    for orb in (Orb.WAKE, Orb.LISTENING, Orb.THINKING, Orb.THINKING, Orb.SPEAKING, Orb.IDLE):
        bus.state(orb)
    assert [e["value"]["orb"] for e in drain(bus)] == ["wake", "listening", "thinking", "speaking", "idle"]


def test_returning_to_an_earlier_value_is_a_change(bus: EventBus) -> None:
    bus.muted(True)
    bus.muted(False)
    bus.muted(True)
    assert [e["value"] for e in drain(bus)] == [True, False, True]


def test_modules_are_tracked_separately(bus: EventBus) -> None:
    bus.module("brain", ACTIVE, "ONLINE")
    bus.module("microphone", ERROR, "NO MIC")
    bus.module("brain", ACTIVE, "ONLINE")  # unchanged
    events = drain(bus)
    assert [(e["value"]["name"], e["value"]["text"]) for e in events] == [
        ("brain", "ONLINE"),
        ("microphone", "NO MIC"),
    ]


def test_meters_only_resend_when_a_reading_changes(bus: EventBus) -> None:
    reading = {"cpu": {"percent": 12, "text": "12%"}, "gpu": None}
    bus.meters(reading)
    bus.meters(dict(reading))
    bus.meters({"cpu": {"percent": 13, "text": "13%"}, "gpu": None})
    assert len(drain(bus)) == 2


def test_every_exchange_is_sent_even_if_identical(bus: EventBus) -> None:
    bus.exchange("volume up", "Volume up to 60 percent.", "cmd")
    bus.exchange("volume up", "Volume up to 60 percent.", "cmd")
    events = drain(bus)
    assert [e["type"] for e in events] == ["log", "log"]
    assert events[0]["value"] == {
        "id": 1,
        "time": "14:30",
        "said": "volume up",
        "reply": "Volume up to 60 percent.",
        "tag": "cmd",
    }
    assert events[1]["value"]["id"] == 2  # ids let the page skip duplicates


def test_snapshot_has_current_values_and_todays_log() -> None:
    clock = Clock()
    bus = EventBus(clock)
    bus.state(Orb.SPEAKING)
    bus.module("stt", ACTIVE, "READY")
    bus.exchange("hello", "Hi.", "brain")
    snapshot = bus.snapshot()
    assert {"type": "state", "value": {"orb": "speaking", "label": "RESPONDING"}} in snapshot["events"]
    assert [e["said"] for e in snapshot["log"]] == ["hello"]

    clock.now += timedelta(days=1)  # midnight passed: yesterday's log is gone
    assert bus.snapshot()["log"] == []
    bus.exchange("good morning", "Morning.", "brain")
    assert [e["said"] for e in bus.snapshot()["log"]] == ["good morning"]


def test_the_bus_is_thread_safe(bus: EventBus) -> None:
    def report(n: int) -> None:
        for i in range(200):
            bus.exchange(f"said {n}-{i}", "ok", "cmd")
            bus.phrase(f"phrase {n}-{i}")

    threads = [threading.Thread(target=report, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    events = drain(bus)
    logs = [e for e in events if e["type"] == "log"]
    assert len(logs) == 1600
    assert len({e["value"]["id"] for e in logs}) == 1600  # no id handed out twice
    assert len(bus.snapshot()["log"]) == 1600


def test_null_presenter_accepts_everything() -> None:
    presenter = NullPresenter()
    presenter.state(Orb.WAKE)
    presenter.phrase("x")
    presenter.exchange("a", "b", "cmd")
    presenter.module("brain", ACTIVE, "ONLINE")
    presenter.confirming(None)
    presenter.muted(True)
    presenter.meters({})


# --- to_js: safe JS for evaluate_js ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "plain",
        'quotes " and \' and `backticks` and ${template}',
        "</script><script>alert(1)</script>",
        "line\nbreak and   separators  ",
        "नमस्ते 東京 café 🎵",
        "\\ backslashes \\",
        "); alert(1); (",
    ],
)
def test_to_js_is_a_single_call_with_an_ascii_json_literal(text: str) -> None:
    script = to_js({"type": "phrase", "value": text})
    assert script.isascii()
    match = re.fullmatch(r"window\.jarvis && window\.jarvis\.onEvent\((.*)\)", script)
    assert match is not None
    assert json.loads(match.group(1)) == {"type": "phrase", "value": text}
    assert "\n" not in script and " " not in script


# --- EventPump with a fake UI -----------------------------------------------------------


class FakeWindow:
    def __init__(self) -> None:
        self.scripts: list[str] = []
        self.fail = False

    def evaluate_js(self, script: str) -> None:
        if self.fail:
            raise RuntimeError("window closed")
        self.scripts.append(script)


def wait_for(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_pump_delivers_events_to_the_ui_in_order(bus: EventBus) -> None:
    window = FakeWindow()
    pump = EventPump(bus, window.evaluate_js)
    pump.start()
    try:
        bus.state(Orb.WAKE)
        bus.state(Orb.LISTENING)
        bus.phrase("open notepad")
        wait_for(lambda: len(window.scripts) == 3)
    finally:
        pump.stop()
        pump.join(timeout=2)
    assert '"orb":"wake"' in window.scripts[0]
    assert '"orb":"listening"' in window.scripts[1]
    assert '"open notepad"' in window.scripts[2]
    assert not pump.is_alive()


def test_pump_survives_a_failing_window(bus: EventBus) -> None:
    window = FakeWindow()
    window.fail = True
    pump = EventPump(bus, window.evaluate_js)
    pump.start()
    try:
        bus.phrase("lost")
        time.sleep(0.1)
        window.fail = False
        bus.phrase("delivered")
        wait_for(lambda: len(window.scripts) == 1)
    finally:
        pump.stop()
        pump.join(timeout=2)
    assert '"delivered"' in window.scripts[0]


# --- Controller: UI -> assistant requests -------------------------------------------------


def test_typed_commands_are_queued_for_the_assistant(bus: EventBus) -> None:
    controller = Controller(bus)
    controller.submit_text("  open   notepad ")
    controller.submit_text("   ")  # ignored
    assert controller.next_request(timeout=0) == ("text", "open notepad")
    assert controller.next_request(timeout=0) is None


def test_typed_text_answers_a_pending_confirmation_instead(bus: EventBus) -> None:
    controller = Controller(bus)
    controller.submit_text("yes")  # before any question: an ordinary command
    controller.begin_confirm()
    controller.submit_text("yes")
    assert controller.typed_answer(timeout=0) == "yes"
    assert controller.next_request(timeout=0) == ("text", "yes")
    assert controller.next_request(timeout=0) is None
    controller.end_confirm()
    controller.submit_text("no")
    assert controller.typed_answer(timeout=0) is None  # not confirming any more


def test_stale_answers_are_dropped_when_a_new_question_starts(bus: EventBus) -> None:
    controller = Controller(bus)
    controller.begin_confirm()
    controller.submit_text("yes")
    controller.end_confirm()
    controller.begin_confirm()
    assert controller.typed_answer(timeout=0) is None


def test_hold_to_talk(bus: EventBus) -> None:
    controller = Controller(bus)
    controller.hold_start()
    controller.hold_start()  # a second press while held queues nothing more
    assert controller.holding
    assert controller.next_request(timeout=0) == ("hold", "")
    assert controller.next_request(timeout=0) is None
    controller.hold_end()
    assert not controller.holding


def test_mute_toggles_and_is_reported(bus: EventBus) -> None:
    controller = Controller(bus)
    assert drain(bus) == [{"type": "muted", "value": False}]
    assert controller.toggle_mute() is True
    assert controller.muted
    assert controller.toggle_mute() is False
    assert [e["value"] for e in drain(bus)] == [True, False]


def test_quit_stops_everything_and_unblocks_waiters(bus: EventBus) -> None:
    controller = Controller(bus)
    controller.hold_start()
    controller.begin_confirm()
    controller.quit()
    assert controller.stopped
    assert not controller.holding
    assert controller.typed_answer(timeout=1) == ""  # an empty answer, i.e. "no"
    assert controller.wait_stopped(0)
