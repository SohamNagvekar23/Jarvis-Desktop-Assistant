"""One Jarvis per Windows session: a named mutex, plus a named event a second
launch uses to ask the running copy to come to the front."""

import ctypes
import threading
from collections.abc import Callable
from ctypes import wintypes

_ERROR_ALREADY_EXISTS = 183
_EVENT_MODIFY_STATE = 0x0002
_WAIT_OBJECT_0 = 0
_ASFW_ANY = wintypes.DWORD(-1)

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
_kernel32.CreateMutexW.restype = wintypes.HANDLE
_kernel32.CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
_kernel32.CreateEventW.restype = wintypes.HANDLE
_kernel32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
_kernel32.OpenEventW.restype = wintypes.HANDLE
_kernel32.SetEvent.argtypes = [wintypes.HANDLE]
_kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_kernel32.WaitForSingleObject.restype = wintypes.DWORD
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_user32.AllowSetForegroundWindow.argtypes = [wintypes.DWORD]


class SingleInstance:
    def __init__(self, mutex_name: str, event_name: str) -> None:
        self._mutex_name = mutex_name
        self._event_name = event_name
        self._mutex: int | None = None
        self._event: int | None = None

    def acquire(self) -> bool:
        """True if this is the only running copy (it then owns the name until
        release() or exit); False if another copy already runs."""
        ctypes.set_last_error(0)  # a stale ERROR_ALREADY_EXISTS must not be misread
        handle = _kernel32.CreateMutexW(None, False, self._mutex_name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
            _kernel32.CloseHandle(handle)
            return False
        self._mutex = handle
        # Created now, so a second launch can signal it as soon as possible.
        self._event = _kernel32.CreateEventW(None, False, False, self._event_name)
        return True

    def signal_existing(self) -> bool:
        """Ask the running copy to show its window; False if it can't be reached."""
        _user32.AllowSetForegroundWindow(_ASFW_ANY)  # let it take the foreground
        event = _kernel32.OpenEventW(_EVENT_MODIFY_STATE, False, self._event_name)
        if not event:
            return False
        try:
            return bool(_kernel32.SetEvent(event))
        finally:
            _kernel32.CloseHandle(event)

    def watch_activations(self, on_activate: Callable[[], None], stop: threading.Event) -> None:
        """Call on_activate whenever another launch signals; returns once stop is set."""
        if self._event is None:
            return
        while not stop.is_set():
            if _kernel32.WaitForSingleObject(self._event, 250) == _WAIT_OBJECT_0:
                on_activate()

    def release(self) -> None:
        for handle in (self._event, self._mutex):
            if handle:
                _kernel32.CloseHandle(handle)
        self._mutex = self._event = None
