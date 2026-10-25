"""System meters for the UI: CPU, Jarvis's memory, GPU memory and volume."""

import ctypes
import logging
from collections.abc import Callable
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvidiaGpu:
    """GPU memory through NVIDIA's own nvml.dll (installed with the driver).

    Cheap: no extra package and no nvidia-smi process. If the DLL or a GPU is
    missing, memory() returns None from then on.
    """

    def __init__(self, load: Callable[[], Any] = lambda: ctypes.CDLL("nvml.dll")) -> None:
        self._load = load
        self._nvml: Any = None
        self._handle = ctypes.c_void_p()
        self._available: bool | None = None  # unknown until first use

    def memory(self) -> tuple[int, int] | None:
        """(used, total) bytes of the first GPU, or None if unavailable."""
        if self._available is None:
            self._available = self._init()
        if not self._available:
            return None
        info = _NvmlMemory()
        if self._nvml.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(info)) != 0:
            return None
        return info.used, info.total

    def _init(self) -> bool:
        try:
            self._nvml = self._load()
            if self._nvml.nvmlInit_v2() != 0:
                return False
            return self._nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self._handle)) == 0
        except (OSError, AttributeError) as error:
            logger.info("No NVIDIA GPU meter: %s", error)
            return False


def read_meters(
    process: psutil.Process,
    gpu_memory: Callable[[], tuple[int, int] | None],
    volume_level: Callable[[], int],
) -> dict[str, Any]:
    """Meter values rounded so unchanged readings compare equal (and aren't resent).

    Each meter is {"percent": 0-100 for the bar, "text": readout} or None if
    unavailable.
    """
    rss = process.memory_info().rss
    total_ram = psutil.virtual_memory().total
    cpu = psutil.cpu_percent(interval=None)  # since the previous call: exactly one per reading
    meters: dict[str, Any] = {
        "cpu": _meter(cpu, f"{cpu:.0f}%"),
        "memory": _meter(100 * rss / total_ram, f"{rss / 2**20:.0f} MB"),
        "gpu": None,
        "volume": None,
    }
    gpu = gpu_memory()
    if gpu is not None and gpu[1] > 0:
        used, total = gpu
        meters["gpu"] = _meter(100 * used / total, f"{used / 2**30:.1f} / {total / 2**30:.0f} GB")
    try:
        level = volume_level()
        meters["volume"] = _meter(level, f"{level}%")
    except OSError:
        pass
    return meters


def _meter(percent: float, text: str) -> dict[str, Any]:
    return {"percent": round(max(0.0, min(100.0, percent))), "text": text}
