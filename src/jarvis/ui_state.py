"""Remembering where the main window and the mini pill were."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

Position = tuple[int, int]


class WindowPositions:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._positions: dict[str, Position] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # first run, or an unreadable file: use the defaults
        if isinstance(data, dict):
            for name, value in data.items():
                if (
                    isinstance(name, str)
                    and isinstance(value, list)
                    and len(value) == 2
                    and all(type(v) is int for v in value)
                ):
                    self._positions[name] = (value[0], value[1])

    def get(self, name: str, size: tuple[int, int], screen: tuple[int, int]) -> Position | None:
        """The saved position, if the window would still be mostly on screen
        (a monitor may have been disconnected since)."""
        position = self._positions.get(name)
        if position is None:
            return None
        x, y = position
        width, height = size
        screen_width, screen_height = screen
        visible = (
            x > -width // 2
            and y >= 0
            and x + width // 2 < screen_width
            and y + 40 < screen_height  # the title bar must stay reachable
        )
        return position if visible else None

    def set(self, name: str, x: int, y: int) -> None:
        self._positions[name] = (int(x), int(y))

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {name: list(position) for name, position in self._positions.items()}
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as error:
            logger.warning("Couldn't save window positions: %s", error)
