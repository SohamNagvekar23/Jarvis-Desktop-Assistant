"""One continuous microphone stream delivered as fixed-size int16 chunks."""

import queue
from types import TracebackType
from typing import Any

import numpy as np
import sounddevice as sd


class Microphone:
    """Audio arrives on PortAudio's thread and waits in a bounded queue.

    Each `with mic:` opens a fresh stream, so one Microphone object can be
    reopened after a failure and shared by everything that listens.
    """

    def __init__(
        self, sample_rate: int, chunk_samples: int, max_queued_chunks: int, stall_seconds: float
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_samples = chunk_samples
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queued_chunks)
        self._stall_seconds = stall_seconds
        self._stream: sd.InputStream | None = None

    def __enter__(self) -> "Microphone":
        self.discard_pending()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._chunk_samples,
            channels=1,
            dtype="int16",
            callback=self._on_audio,
        )
        self._stream.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _on_audio(self, indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags) -> None:
        chunk = indata[:, 0].copy()  # PortAudio reuses indata after the callback returns
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:  # we're far behind: drop the oldest chunk, keep the newest
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk)
            except (queue.Empty, queue.Full):
                pass

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def read(self) -> np.ndarray:
        if self._stream is None:
            raise OSError("microphone is not open")
        try:
            return self._queue.get(timeout=self._stall_seconds)
        except queue.Empty:
            raise OSError("microphone stopped sending audio") from None

    def discard_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
