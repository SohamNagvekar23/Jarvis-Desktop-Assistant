"""Find and launch whitelisted Windows applications."""

import os
import shutil
import winreg
from collections.abc import Iterable
from pathlib import Path

from jarvis import known_folders
from jarvis.apps import App

_APP_PATHS = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"


def find_launch_target(app: App) -> str | None:
    """What to pass to launch() for app, or None if it isn't installed."""
    if app.launch_kind == "exe":
        found = shutil.which(app.launch) is not None or _in_app_paths(app.launch)
        return app.launch if found else None
    if app.launch_kind == "uri":
        return app.launch if _uri_registered(app.launch.rstrip(":")) else None
    shortcut = find_shortcut(
        app.launch, [known_folders.programs(), known_folders.common_programs()]
    )
    return str(shortcut) if shortcut else None


def find_shortcut(file_name: str, directories: Iterable[Path]) -> Path | None:
    """The least nested Start-menu shortcut called file_name (case-insensitive)."""
    wanted = file_name.lower()
    matches = [
        path
        for directory in directories
        if directory.is_dir()
        for path in directory.rglob("*.lnk")
        if path.name.lower() == wanted
    ]
    return min(matches, key=lambda path: len(path.parts), default=None)


def launch(target: str) -> None:
    """Start an exe name, URI or shortcut path via the Windows shell.

    Only called with targets derived from config.APPS, never with speech text.
    Raises OSError if Windows cannot start it.
    """
    os.startfile(target)


def open_url(url: str) -> None:
    """Open an https URL in the default browser.

    Only called with URLs built from config.py (sites and search templates).
    """
    if not url.startswith("https://"):
        raise ValueError(f"refusing to open non-https URL {url!r}")
    os.startfile(url)


def _in_app_paths(executable: str) -> bool:
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            winreg.CloseKey(winreg.OpenKey(hive, rf"{_APP_PATHS}\{executable}"))
            return True
        except OSError:
            continue
    return False


def _uri_registered(scheme: str) -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, scheme) as key:
            winreg.QueryValueEx(key, "URL Protocol")
            return True
    except OSError:
        return False
