"""Whitelisted application descriptions and spoken-name lookup (no OS code)."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

LaunchKind = Literal["exe", "uri", "shortcut"]


@dataclass(frozen=True)
class App:
    name: str  # used in spoken replies
    aliases: tuple[str, ...]  # what you can call it: "open <alias>", "close <alias>"
    # exe: executable name resolved via PATH / App Paths; uri: e.g. "ms-settings:";
    # shortcut: .lnk file name searched for in the Start menu folders.
    launch: str
    launch_kind: LaunchKind
    processes: tuple[str, ...]  # executable names whose windows "close" acts on
    window_classes: tuple[str, ...] = ()  # if set, only close windows of these classes
    force_close: bool = True  # may its processes be ended if closing gracefully fails?


class CloseResult(Enum):
    CLOSED = auto()
    NOT_OPEN = auto()
    STILL_OPEN = auto()  # e.g. the app is asking to save changes
    DENIED = auto()  # e.g. the app runs as administrator


# Words people add around an app or site name: "open up the calculator app please".
_LEADING_FILLER = frozenset({"the", "up", "all"})
_TRAILING_FILLER = frozenset(
    {
        "app",
        "application",
        "program",
        "window",
        "windows",
        "website",
        "site",
        "please",
        "now",
        "for",
        "me",
        "jarvis",
    }
)


def strip_filler(words: str) -> str:
    """Drop filler words around a normalized name; the name itself must remain whole."""
    tokens = words.split()
    while tokens and tokens[0] in _LEADING_FILLER:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _TRAILING_FILLER:
        tokens = tokens[:-1]
    return " ".join(tokens)


class AppCatalog:
    """Maps normalized aliases to apps."""

    def __init__(self, apps: Iterable[App]) -> None:
        self.apps: tuple[App, ...] = tuple(apps)
        self._by_alias: dict[str, App] = {}
        for app in self.apps:
            for alias in app.aliases:
                if alias in self._by_alias:
                    raise ValueError(f"alias {alias!r} is used by two apps")
                self._by_alias[alias] = app

    def find(self, words: str) -> App | None:
        """Find the app named by normalized words, ignoring filler around the name.

        The name must be the whole phrase: "notepad and chrome" matches nothing.
        """
        return self._by_alias.get(strip_filler(words))
