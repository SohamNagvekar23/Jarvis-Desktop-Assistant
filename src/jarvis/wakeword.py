"""openWakeWord wake-word model and its bundled Silero VAD, on ONNX Runtime."""

import sys
import types
from typing import Any, NoReturn

import numpy as np


def _skip_verifier_training_imports() -> None:
    """Keep scipy and scikit-learn (~70 MB resident) out of memory.

    openwakeword/__init__.py imports its custom-verifier *training* module,
    which imports both libraries; wake-word inference never uses them.
    Registering a stand-in first means that import loads nothing.
    """
    name = "openwakeword.custom_verifier_model"
    if name in sys.modules:
        return
    stub = types.ModuleType(name)

    def train_custom_verifier(*args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError("openWakeWord verifier training is disabled in Jarvis")

    stub.train_custom_verifier = train_custom_verifier  # type: ignore[attr-defined]
    sys.modules[name] = stub


_skip_verifier_training_imports()

import openwakeword.utils  # noqa: E402  (must follow the stand-in above)
from openwakeword.model import Model  # noqa: E402
from openwakeword.vad import VAD  # noqa: E402


class OpenWakeWordDetector:
    def __init__(self, model_name: str) -> None:
        # Fetches the model files on first run; skips files already present.
        openwakeword.utils.download_models([model_name])
        self._model = Model(wakeword_models=[model_name], inference_framework="onnx")
        self._name = model_name

    def score(self, chunk: np.ndarray) -> float:
        """Wake-word probability (0..1) for an 80 ms int16 chunk."""
        return float(self._model.predict(chunk)[self._name])

    def reset(self) -> None:
        # Clears buffered audio features; the model also scores the next 5 chunks
        # as 0 while its buffer refills, so stale audio cannot fire it.
        self._model.reset()


class SileroVad:
    def __init__(self) -> None:
        self._vad = VAD(n_threads=1)

    def speech_probability(self, chunk: np.ndarray) -> float:
        """Mean speech probability over the chunk's 40 ms frames."""
        return float(self._vad.predict(chunk, frame_size=640))

    def reset(self) -> None:
        self._vad.reset_states()
