import ctypes

import pytest

from jarvis import media_keys


class FakeUser32:
    def __init__(self, sent: int | None = None) -> None:
        self.sent = sent
        self.calls: list[tuple[int, list[tuple[int, int, int]], int]] = []

    def SendInput(self, count: int, events: ctypes.Array, size: int) -> int:  # noqa: N802 (Win32 name)
        keys = [(e.type, e.union.ki.wVk, e.union.ki.dwFlags) for e in events[:count]]
        self.calls.append((count, keys, size))
        return count if self.sent is None else self.sent


@pytest.mark.parametrize(
    ("name", "vk"),
    [("play_pause", 0xB3), ("next", 0xB0), ("previous", 0xB1), ("stop", 0xB2)],
)
def test_presses_and_releases_the_media_key(
    monkeypatch: pytest.MonkeyPatch, name: str, vk: int
) -> None:
    user32 = FakeUser32()
    monkeypatch.setattr(media_keys, "_user32", user32)
    media_keys.press_media_key(name)
    ((count, keys, size),) = user32.calls
    assert count == 2
    assert keys == [(1, vk, 0x1), (1, vk, 0x1 | 0x2)]  # keyboard input; down, then up
    assert size == ctypes.sizeof(media_keys._INPUT)


def test_input_struct_has_the_size_windows_expects() -> None:
    # SendInput fails with ERROR_INVALID_PARAMETER if cbSize is wrong.
    assert ctypes.sizeof(media_keys._INPUT) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)


def test_only_whitelisted_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_keys, "_user32", FakeUser32())
    with pytest.raises(KeyError):
        media_keys.press_media_key("VK_LWIN")


def test_failure_raises_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_keys, "_user32", FakeUser32(sent=0))
    with pytest.raises(OSError):
        media_keys.press_media_key("next")
