"""Close whitelisted apps: WM_CLOSE to their windows first, ending processes only on request."""

import ctypes
import time
from ctypes import wintypes

import psutil

from jarvis.apps import App, CloseResult
from jarvis.window_targets import (
    FRAME_WINDOW_CLASS,
    ProcessInfo,
    Targets,
    WindowInfo,
    select_targets,
)

_WM_CLOSE = 0x0010
_GW_OWNER = 4
_DWMWA_CLOAKED = 14
_ERROR_ACCESS_DENIED = 5

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_dwmapi = ctypes.WinDLL("dwmapi")
_ENUM_PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
_user32.EnumWindows.argtypes = [_ENUM_PROC, wintypes.LPARAM]
_user32.EnumChildWindows.argtypes = [wintypes.HWND, _ENUM_PROC, wintypes.LPARAM]
_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
_user32.GetWindow.restype = wintypes.HWND
_user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.PostMessageW.restype = wintypes.BOOL
_dwmapi.DwmGetWindowAttribute.argtypes = [
    wintypes.HWND,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
]


class WindowsAppWindows:
    def __init__(self, grace_seconds: float, force_wait_seconds: float) -> None:
        self._grace_seconds = grace_seconds
        self._force_wait_seconds = force_wait_seconds
        self._user = psutil.Process().username()

    def is_open(self, app: App) -> bool:
        return bool(self._targets(app).windows)

    def close(self, app: App) -> CloseResult:
        processes = _processes(app)
        targets = select_targets(app, processes, _windows(), self._user)
        if not targets.windows:
            return CloseResult.NOT_OPEN
        for hwnd in targets.windows:
            if not _user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0):
                error = ctypes.get_last_error()
                if error == _ERROR_ACCESS_DENIED:  # window of an elevated process
                    return CloseResult.DENIED
                raise ctypes.WinError(error)
        deadline = time.monotonic() + self._grace_seconds
        while time.monotonic() < deadline:
            time.sleep(0.1)
            if not select_targets(app, processes, _windows(), self._user).windows:
                return CloseResult.CLOSED
        return CloseResult.STILL_OPEN

    def force_close(self, app: App) -> CloseResult:
        if not app.force_close:
            return CloseResult.STILL_OPEN
        names = {name.lower() for name in app.processes}
        processes: list[psutil.Process] = []
        denied = False
        for pid in self._targets(app).pids:
            try:
                process = psutil.Process(pid)
                # Re-check right before ending it, in case the pid was reused.
                if process.name().lower() not in names or process.username() != self._user:
                    continue
                process.terminate()
                processes.append(process)
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                denied = True
        _, alive = psutil.wait_procs(processes, timeout=self._force_wait_seconds)
        if denied:
            return CloseResult.DENIED
        return CloseResult.STILL_OPEN if alive else CloseResult.CLOSED

    def _targets(self, app: App) -> Targets:
        return select_targets(app, _processes(app), _windows(), self._user)


def _processes(app: App) -> list[ProcessInfo]:
    names = {name.lower() for name in app.processes}
    found = []
    for process in psutil.process_iter(["name"]):
        name = process.info["name"] or ""
        if name.lower() not in names:
            continue
        try:
            username = process.username()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            username = None
        found.append(ProcessInfo(process.pid, name, username))
    return found


def _windows() -> list[WindowInfo]:
    handles: list[int] = []

    @_ENUM_PROC
    def collect(hwnd: int, _: int) -> bool:
        handles.append(hwnd)
        return True

    _user32.EnumWindows(collect, 0)
    return [_describe(hwnd) for hwnd in handles]


def _describe(hwnd: int) -> WindowInfo:
    class_name = _class_name(hwnd)
    cloaked = ctypes.c_int(0)
    _dwmapi.DwmGetWindowAttribute(hwnd, _DWMWA_CLOAKED, ctypes.byref(cloaked), 4)
    child_pids: frozenset[int] = frozenset()
    if class_name == FRAME_WINDOW_CLASS:
        pids: set[int] = set()

        @_ENUM_PROC
        def collect(child: int, _: int) -> bool:
            pids.add(_pid(child))
            return True

        _user32.EnumChildWindows(hwnd, collect, 0)
        child_pids = frozenset(pids)
    return WindowInfo(
        hwnd=hwnd,
        pid=_pid(hwnd),
        class_name=class_name,
        visible=bool(_user32.IsWindowVisible(hwnd)),
        cloaked=bool(cloaked.value),
        owned=bool(_user32.GetWindow(hwnd, _GW_OWNER)),
        child_pids=child_pids,
    )


def _pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def _class_name(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    _user32.GetClassNameW(hwnd, buffer, 256)
    return buffer.value
