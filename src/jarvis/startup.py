"""'Start with Windows': a per-user value in the registry's Run key (no admin)."""

import sys
import winreg
from enum import Enum
from pathlib import Path
from typing import Protocol


class RunKey(Protocol):
    def get(self, name: str) -> str | None: ...
    def set(self, name: str, value: str) -> None: ...
    def delete(self, name: str) -> None: ...


class WindowsRunKey:
    """A value-level view of HKEY_CURRENT_USER\\<key_path>."""

    def __init__(self, key_path: str) -> None:
        self._key_path = key_path

    def get(self, name: str) -> str | None:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self._key_path) as key:
                value, kind = winreg.QueryValueEx(key, name)
        except FileNotFoundError:
            return None
        return value if kind in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) and isinstance(value, str) else None

    def set(self, name: str, value: str) -> None:
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, self._key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)

    def delete(self, name: str) -> None:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self._key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, name)


class Status(str, Enum):
    OFF = "off"
    ON = "on"
    OUTDATED = "outdated"  # an entry exists but points elsewhere (e.g. the venv moved)


def startup_command(executable: str = sys.executable) -> str:
    """The Run entry: this venv's pythonw.exe (no console window), absolute and quoted.

    Raises FileNotFoundError if there is no pythonw.exe next to the interpreter.
    """
    pythonw = Path(executable).resolve().with_name("pythonw.exe")
    if not pythonw.is_file():
        raise FileNotFoundError(f"pythonw.exe not found next to {executable}")
    return f'"{pythonw}" -m jarvis --startup'


class StartupEntry:
    def __init__(self, run_key: RunKey, value_name: str, command: str) -> None:
        self._run_key = run_key
        self._name = value_name
        self.command = command

    def status(self) -> Status:
        current = self._run_key.get(self._name)
        if current is None:
            return Status.OFF
        return Status.ON if current == self.command else Status.OUTDATED

    def install(self) -> str:
        """Add the entry, or rewrite it if it points somewhere else."""
        before = self.status()
        if before is Status.ON:
            return "unchanged"
        self._run_key.set(self._name, self.command)
        return "updated" if before is Status.OUTDATED else "installed"

    def remove(self) -> bool:
        """Remove the entry; False if there was none."""
        if self._run_key.get(self._name) is None:
            return False
        self._run_key.delete(self._name)
        return True
