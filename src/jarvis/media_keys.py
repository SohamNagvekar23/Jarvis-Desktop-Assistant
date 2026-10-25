"""Press the Windows virtual media keys (works with Spotify, browsers and most players)."""

import ctypes
from ctypes import wintypes

MEDIA_KEYS: dict[str, int] = {
    "play_pause": 0xB3,  # VK_MEDIA_PLAY_PAUSE
    "next": 0xB0,  # VK_MEDIA_NEXT_TRACK
    "previous": 0xB1,  # VK_MEDIA_PREV_TRACK
    "stop": 0xB2,  # VK_MEDIA_STOP
}

_INPUT_KEYBOARD = 1
_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    # All members are declared so the union (and INPUT) has the size Windows expects.
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
_user32.SendInput.restype = wintypes.UINT


def _key_event(vk: int, flags: int) -> _INPUT:
    event = _INPUT(type=_INPUT_KEYBOARD)
    event.union.ki = _KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
    return event


def press_media_key(name: str) -> None:
    """Press and release one media key by name (see MEDIA_KEYS); raises OSError on failure."""
    vk = MEDIA_KEYS[name]  # KeyError for anything not whitelisted
    events = (_INPUT * 2)(
        _key_event(vk, _KEYEVENTF_EXTENDEDKEY),
        _key_event(vk, _KEYEVENTF_EXTENDEDKEY | _KEYEVENTF_KEYUP),
    )
    sent = _user32.SendInput(len(events), events, ctypes.sizeof(_INPUT))
    if sent != len(events):
        raise ctypes.WinError(ctypes.get_last_error())
