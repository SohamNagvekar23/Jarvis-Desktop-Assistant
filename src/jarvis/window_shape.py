"""Win32 helpers for Jarvis's own windows: capsule shape, bringing to front."""

import ctypes
from ctypes import wintypes

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
_user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
_user32.SetWindowRgn.argtypes = [wintypes.HWND, wintypes.HRGN, wintypes.BOOL]
_gdi32.CreateRoundRectRgn.argtypes = [ctypes.c_int] * 6
_gdi32.CreateRoundRectRgn.restype = wintypes.HRGN
_gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
_user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.SetForegroundWindow.argtypes = [wintypes.HWND]
_user32.SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT
]

_SW_RESTORE = 9
_HWND_TOPMOST = wintypes.HWND(-1)
_HWND_NOTOPMOST = wintypes.HWND(-2)
_SWP_NOMOVE, _SWP_NOSIZE, _SWP_SHOWWINDOW = 0x0002, 0x0001, 0x0040


def bring_to_front(hwnd: int) -> None:
    """Restore the window, raise it above other windows and give it the
    foreground. Focus only moves if the launch that asked for it was in the
    foreground (it calls AllowSetForegroundWindow); the topmost toggle raises
    the window either way."""
    flags = _SWP_NOMOVE | _SWP_NOSIZE | _SWP_SHOWWINDOW
    _user32.ShowWindow(hwnd, _SW_RESTORE)
    _user32.SetWindowPos(hwnd, _HWND_TOPMOST, 0, 0, 0, 0, flags)
    _user32.SetWindowPos(hwnd, _HWND_NOTOPMOST, 0, 0, 0, 0, flags)
    _user32.SetForegroundWindow(hwnd)


def make_capsule(hwnd: int) -> None:
    """Clip the window to a rounded rectangle whose ends are half circles.

    Uses the window's real (physical-pixel) size, so it's right at any display
    scaling. Raises OSError if Windows refuses.
    """
    rect = wintypes.RECT()
    if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())
    width, height = rect.right - rect.left, rect.bottom - rect.top
    region = _gdi32.CreateRoundRectRgn(0, 0, width + 1, height + 1, height, height)
    if not region:
        raise ctypes.WinError(ctypes.get_last_error())
    if not _user32.SetWindowRgn(hwnd, region, True):  # on success Windows owns the region
        _gdi32.DeleteObject(region)
        raise ctypes.WinError(ctypes.get_last_error())
