"""Logging to a size-capped, rotating file (pythonw.exe has no console)."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(log_file: Path, max_bytes: int, backup_count: int) -> None:
    """Log INFO and above to log_file (rotated at max_bytes, keeping backup_count
    old files), and warnings to the console when there is one."""
    _replace_missing_streams()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(file_handler)
    except OSError:
        pass  # can't write the log file: carry on with the console only
    if sys.stderr is not None and getattr(sys.stderr, "name", "") != os.devnull:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.WARNING)
        console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(console)


def _replace_missing_streams() -> None:
    """Under pythonw.exe started at login, sys.stdout/stderr are None, and
    libraries that write to them (progress bars, warnings) would crash."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
