"""Fixed-length microphone recording."""

import numpy as np
import sounddevice as sd


class Recorder:
    """Records mono float32 audio into a buffer allocated once and reused."""

    def __init__(self, sample_rate: int, seconds: float) -> None:
        self._sample_rate = sample_rate
        self._buffer = np.empty((int(sample_rate * seconds), 1), dtype=np.float32)

    def record(self) -> np.ndarray:
        """Block until the clip is recorded.

        Returns a 1-D view of the internal buffer (no copy); it is overwritten
        by the next call.
        """
        sd.rec(out=self._buffer, samplerate=self._sample_rate)
        sd.wait()
        return self._buffer.reshape(-1)
