from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from jarvis import microphone
from jarvis.microphone import Microphone


class FakeStream:
    created: list["FakeStream"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        FakeStream.created.append(self)

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def closed_mic(monkeypatch: pytest.MonkeyPatch) -> Microphone:
    FakeStream.created = []
    monkeypatch.setattr(microphone.sd, "InputStream", FakeStream)
    return Microphone(16_000, 1280, max_queued_chunks=3, stall_seconds=0.01)


@pytest.fixture
def mic(closed_mic: Microphone) -> Iterator[Microphone]:
    with closed_mic:
        yield closed_mic


def deliver(mic: Microphone, value: int) -> np.ndarray:
    """Simulate PortAudio calling back with one (frames, 1) int16 block."""
    block = np.full((1280, 1), value, dtype=np.int16)
    mic._on_audio(block, 1280, None, None)  # type: ignore[arg-type]
    return block


def test_opens_one_16k_mono_int16_stream_in_80ms_blocks(closed_mic: Microphone) -> None:
    assert FakeStream.created == []  # nothing opened until entered
    with closed_mic:
        (stream,) = FakeStream.created
        assert stream.kwargs["samplerate"] == 16_000
        assert stream.kwargs["blocksize"] == 1280
        assert stream.kwargs["channels"] == 1
        assert stream.kwargs["dtype"] == "int16"
        assert stream.started
    assert stream.closed


def test_reopening_creates_a_fresh_stream_and_drops_old_audio(closed_mic: Microphone) -> None:
    with closed_mic:
        deliver(closed_mic, 1)
    with closed_mic:
        deliver(closed_mic, 2)
        assert closed_mic.read()[0] == 2
    assert len(FakeStream.created) == 2
    assert all(stream.closed for stream in FakeStream.created)


def test_read_when_closed_raises_oserror(closed_mic: Microphone) -> None:
    with pytest.raises(OSError, match="not open"):
        closed_mic.read()


def test_chunks_are_copied_and_one_dimensional(mic: Microphone) -> None:
    block = deliver(mic, 7)
    block[:] = 0  # PortAudio reuses its buffer
    chunk = mic.read()
    assert chunk.shape == (1280,)
    assert chunk.dtype == np.int16
    assert chunk[0] == 7


def test_full_queue_drops_the_oldest_chunk(mic: Microphone) -> None:
    for value in range(5):
        deliver(mic, value)
    assert [int(mic.read()[0]) for _ in range(3)] == [2, 3, 4]


def test_discard_pending_empties_the_queue(mic: Microphone) -> None:
    deliver(mic, 1)
    deliver(mic, 2)
    mic.discard_pending()
    deliver(mic, 3)
    assert mic.read()[0] == 3


def test_stalled_stream_raises_oserror(mic: Microphone) -> None:
    with pytest.raises(OSError, match="stopped sending audio"):
        mic.read()
