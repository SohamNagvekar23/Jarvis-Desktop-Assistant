"""Speech-to-text with faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    """Loads the Whisper model once and reuses it for every clip."""

    def __init__(
        self, model_size: str, device: str, compute_type: str, language: str, beam_size: int
    ) -> None:
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language
        self._beam_size = beam_size

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe 16 kHz mono float32 audio; returns "" if no speech."""
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
