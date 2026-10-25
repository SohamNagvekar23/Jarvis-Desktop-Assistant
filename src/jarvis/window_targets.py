"""Which processes and windows "close <app>" may act on (pure logic)."""

from collections.abc import Iterable
from dataclasses import dataclass

from jarvis.apps import App

# Store apps (Calculator, Settings) draw inside a frame window owned by the
# shared ApplicationFrameHost.exe; the app's own window is a child of it.
FRAME_WINDOW_CLASS = "ApplicationFrameWindow"


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    name: str
    username: str | None  # None if Windows won't tell us (another user / SYSTEM)


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    pid: int
    class_name: str
    visible: bool
    cloaked: bool  # hidden by Windows (e.g. a suspended Store app) though "visible"
    owned: bool  # a dialog or tool window belonging to another window
    child_pids: frozenset[int] = frozenset()  # only filled in for frame windows


@dataclass(frozen=True)
class Targets:
    pids: frozenset[int]  # processes force_close may end
    windows: tuple[int, ...]  # main windows to ask to close


def select_targets(
    app: App, processes: Iterable[ProcessInfo], windows: Iterable[WindowInfo], user: str
) -> Targets:
    """Only the current user's processes named in app.processes, and their
    visible main windows (or the Store-app frames hosting them)."""
    names = {name.lower() for name in app.processes}
    pids = frozenset(p.pid for p in processes if p.name.lower() in names and p.username == user)
    selected = tuple(
        w.hwnd
        for w in windows
        if w.visible
        and not w.cloaked
        and not w.owned
        and (not app.window_classes or w.class_name in app.window_classes)
        and (w.pid in pids or (w.class_name == FRAME_WINDOW_CLASS and bool(w.child_pids & pids)))
    )
    return Targets(pids, selected)
