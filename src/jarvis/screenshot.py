"""Full-desktop screenshots saved as PNG, never overwriting existing files."""

import itertools
from datetime import datetime
from pathlib import Path

import mss
import mss.tools


def save_new_file(data: bytes, directory: Path, stem: str, suffix: str) -> Path:
    """Write data to directory/stem+suffix, adding _1, _2, ... if the name is taken.

    Uses exclusive create ("x" mode), so an existing file is never overwritten,
    even if another program creates one at the same moment.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in itertools.count():
        path = directory / (f"{stem}{suffix}" if attempt == 0 else f"{stem}_{attempt}{suffix}")
        try:
            with path.open("xb") as file:
                file.write(data)
        except FileExistsError:
            continue
        return path
    raise AssertionError("unreachable")


class ScreenCapture:
    def __init__(self, directory: Path, prefix: str) -> None:
        self._directory = directory
        self._prefix = prefix

    def capture(self) -> Path:
        """Capture all monitors as one image and return the saved file's path."""
        try:
            with mss.MSS() as screen:
                shot = screen.grab(screen.monitors[0])  # 0 = union of all monitors
                png = mss.tools.to_png(shot.rgb, shot.size)
        except mss.ScreenShotError as error:
            raise OSError(f"screen capture failed: {error}") from error
        assert png is not None  # to_png returns bytes when no output path is given
        stem = f"{self._prefix}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        return save_new_file(png, self._directory, stem, ".png")
