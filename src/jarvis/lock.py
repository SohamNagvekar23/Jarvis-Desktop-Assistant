"""Lock the Windows session."""

import ctypes

_user32 = ctypes.WinDLL("user32", use_last_error=True)


def lock_screen() -> None:
    """Lock the workstation; raises OSError if Windows refuses."""
    if not _user32.LockWorkStation():
        raise ctypes.WinError(ctypes.get_last_error())
