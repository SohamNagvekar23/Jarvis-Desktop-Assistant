"""Thread-safe bridge between the assistant (background thread) and the UI.

The assistant reports what it's doing to a Presenter; EventBus turns that into
events for the UI, sending each one only when the value actually changes.
Controller carries requests the other way (typed commands, hold-to-talk, mute,
quit). Neither knows anything about pywebview, so tests use them directly.
"""

import itertools
import json
import queue
import threading
from collections.abc import Callable, Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


class Orb(str, Enum):
    IDLE = "idle"
    WAKE = "wake"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


STATUS_LABELS: dict[Orb, str] = {
    Orb.IDLE: "STANDBY",
    Orb.WAKE: "WAKE DETECTED",
    Orb.LISTENING: "LISTENING",
    Orb.THINKING: "PROCESSING",
    Orb.SPEAKING: "RESPONDING",
}

# Module status levels shown as colored dots.
ACTIVE, IDLE_LEVEL, WARN, ERROR = "active", "idle", "warn", "error"


class Presenter(Protocol):
    def state(self, orb: Orb) -> None: ...
    def phrase(self, text: str) -> None: ...
    def exchange(self, said: str, reply: str, tag: str) -> None: ...
    def module(self, name: str, level: str, text: str) -> None: ...
    def confirming(self, question: str | None) -> None: ...
    def muted(self, muted: bool) -> None: ...
    def meters(self, values: Mapping[str, Any]) -> None: ...
    def error(self, message: str | None) -> None: ...


class NullPresenter:
    """Used without a UI (--no-ui): every report is ignored."""

    def state(self, orb: Orb) -> None:
        pass

    def phrase(self, text: str) -> None:
        pass

    def exchange(self, said: str, reply: str, tag: str) -> None:
        pass

    def module(self, name: str, level: str, text: str) -> None:
        pass

    def confirming(self, question: str | None) -> None:
        pass

    def muted(self, muted: bool) -> None:
        pass

    def meters(self, values: Mapping[str, Any]) -> None:
        pass

    def error(self, message: str | None) -> None:
        pass


NULL_PRESENTER = NullPresenter()


class EventBus:
    """A Presenter that queues UI events, skipping values that didn't change."""

    def __init__(self, clock: Callable[[], datetime] = datetime.now) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._current: dict[str, dict[str, Any]] = {}  # last event per key, for snapshots
        self._log: list[dict[str, Any]] = []
        self._log_day = clock().date()
        self._ids = itertools.count(1)

    # --- Presenter ---

    def state(self, orb: Orb) -> None:
        self._set("state", "state", {"orb": orb.value, "label": STATUS_LABELS[orb]})

    def phrase(self, text: str) -> None:
        self._set("phrase", "phrase", text)

    def exchange(self, said: str, reply: str, tag: str) -> None:
        now = self._clock()
        entry = {
            "id": next(self._ids),
            "time": now.strftime("%H:%M"),
            "said": said,
            "reply": reply,
            "tag": tag,
        }
        with self._lock:
            if now.date() != self._log_day:  # only today's exchanges
                self._log.clear()
                self._log_day = now.date()
            self._log.append(entry)
            self._queue.put({"type": "log", "value": entry})

    def module(self, name: str, level: str, text: str) -> None:
        self._set(f"module:{name}", "module", {"name": name, "level": level, "text": text})

    def confirming(self, question: str | None) -> None:
        self._set("confirm", "confirm", question)

    def muted(self, muted: bool) -> None:
        self._set("muted", "muted", muted)

    def meters(self, values: Mapping[str, Any]) -> None:
        self._set("meters", "meters", dict(values))

    def error(self, message: str | None) -> None:
        """A problem the user must see (e.g. Jarvis couldn't start); None clears it."""
        self._set("error", "error", message)

    # --- UI side ---

    def next_event(self, timeout: float) -> dict[str, Any] | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def snapshot(self) -> dict[str, Any]:
        """Everything a freshly loaded page needs: current values and today's log."""
        with self._lock:
            if self._clock().date() != self._log_day:
                self._log.clear()
                self._log_day = self._clock().date()
            return {"events": list(self._current.values()), "log": list(self._log)}

    def _set(self, key: str, kind: str, value: Any) -> None:
        event = {"type": kind, "value": value}
        with self._lock:
            if self._current.get(key) == event:
                return  # unchanged: don't send anything
            self._current[key] = event
            self._queue.put(event)


def to_js(event: Mapping[str, Any]) -> str:
    """A JS statement delivering one event to the page.

    json.dumps with ensure_ascii escapes everything outside ASCII (including
    U+2028/U+2029), and the result is a plain object literal, so no transcript
    text can break out of it.
    """
    payload = json.dumps(event, ensure_ascii=True, separators=(",", ":"))
    return f"window.jarvis && window.jarvis.onEvent({payload})"


class EventPump(threading.Thread):
    """Forwards bus events to the UI (deliver runs one JS statement)."""

    def __init__(self, bus: EventBus, deliver: Callable[[str], None]) -> None:
        super().__init__(name="ui-events", daemon=True)
        self._bus = bus
        self._deliver = deliver
        self._stopped = threading.Event()

    def run(self) -> None:
        while not self._stopped.is_set():
            event = self._bus.next_event(timeout=0.25)
            if event is not None:
                try:
                    self._deliver(to_js(event))
                except Exception:  # a closing window must not kill the pump
                    if self._stopped.is_set():
                        return

    def stop(self) -> None:
        self._stopped.set()


Request = tuple[str, str]  # ("text", command) | ("hold", "") | ("quit", "")


class Controller:
    """Requests from the UI thread to the assistant thread."""

    def __init__(self, presenter: Presenter) -> None:
        self._presenter = presenter
        self._requests: queue.Queue[Request] = queue.Queue()
        self._answers: queue.Queue[str] = queue.Queue()
        self._confirming = threading.Event()
        self._holding = threading.Event()
        self._muted = threading.Event()
        self._stopped = threading.Event()
        presenter.muted(False)

    # --- UI side ---

    def submit_text(self, text: str) -> None:
        """A typed command, or the answer to a pending confirmation question."""
        text = " ".join(text.split())
        if not text:
            return
        if self._confirming.is_set():
            self._answers.put(text)
        else:
            self._requests.put(("text", text))

    def hold_start(self) -> None:
        if not self._holding.is_set():
            self._holding.set()
            self._requests.put(("hold", ""))

    def hold_end(self) -> None:
        self._holding.clear()

    def toggle_mute(self) -> bool:
        self.set_muted(not self._muted.is_set())
        return self._muted.is_set()

    def set_muted(self, muted: bool) -> None:
        if muted:
            self._muted.set()
        else:
            self._muted.clear()
        self._presenter.muted(muted)

    def quit(self) -> None:
        self._stopped.set()
        self._holding.clear()
        self._requests.put(("quit", ""))
        self._answers.put("")  # unblock a pending confirmation (an empty answer is "no")

    # --- assistant side ---

    @property
    def muted(self) -> bool:
        return self._muted.is_set()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    @property
    def holding(self) -> bool:
        return self._holding.is_set()

    def next_request(self, timeout: float) -> Request | None:
        try:
            if timeout <= 0:
                return self._requests.get_nowait()
            return self._requests.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait_stopped(self, timeout: float) -> bool:
        return self._stopped.wait(timeout)

    def begin_confirm(self) -> None:
        while True:  # drop answers typed before this question was asked
            try:
                self._answers.get_nowait()
            except queue.Empty:
                break
        self._confirming.set()

    def end_confirm(self) -> None:
        self._confirming.clear()

    def typed_answer(self, timeout: float) -> str | None:
        try:
            if timeout <= 0:
                return self._answers.get_nowait()
            return self._answers.get(timeout=timeout)
        except queue.Empty:
            return None
