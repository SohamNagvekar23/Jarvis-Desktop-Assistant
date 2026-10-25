"""Short audio cues."""

import winsound


def beep(frequency_hz: int, duration_ms: int) -> None:
    """Play a tone; blocks for duration_ms."""
    winsound.Beep(frequency_hz, duration_ms)
