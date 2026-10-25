"""Resolve Windows known folders (follows OneDrive / user redirection)."""

import ctypes
import uuid
from pathlib import Path

_FOLDERID_PICTURES = uuid.UUID("33E28130-4E1E-4676-835A-98395C3BC3BB")
_FOLDERID_PROGRAMS = uuid.UUID("A77F5D77-2E2B-44C3-A6A2-ABA601054A51")
_FOLDERID_COMMON_PROGRAMS = uuid.UUID("0139D44E-6AFE-49F2-8690-3DAFCAE6FFB8")
_FOLDERID_ROAMING_APP_DATA = uuid.UUID("3EB685DB-65F9-4CF6-A03A-E3EF65729F3D")
_FOLDERID_LOCAL_APP_DATA = uuid.UUID("F1B32785-6FBA-4FCF-9D55-7B8E7F157091")


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _known_folder(folder_id: uuid.UUID) -> Path:
    guid = _GUID(
        folder_id.time_low,
        folder_id.time_mid,
        folder_id.time_hi_version,
        (ctypes.c_ubyte * 8).from_buffer_copy(folder_id.bytes[8:]),
    )
    path = ctypes.c_wchar_p()
    result = ctypes.windll.shell32.SHGetKnownFolderPath(
        ctypes.byref(guid), 0, None, ctypes.byref(path)
    )
    try:
        if result != 0:
            raise OSError(f"SHGetKnownFolderPath failed with HRESULT {result & 0xFFFFFFFF:#010x}")
        return Path(path.value or "")
    finally:
        ctypes.windll.ole32.CoTaskMemFree(path)


def pictures() -> Path:
    """The user's Pictures folder, wherever Windows has it."""
    return _known_folder(_FOLDERID_PICTURES)


def programs() -> Path:
    """The user's Start menu Programs folder."""
    return _known_folder(_FOLDERID_PROGRAMS)


def common_programs() -> Path:
    """The all-users Start menu Programs folder."""
    return _known_folder(_FOLDERID_COMMON_PROGRAMS)


def roaming_app_data() -> Path:
    """The user's roaming AppData folder (where apps keep settings)."""
    return _known_folder(_FOLDERID_ROAMING_APP_DATA)


def local_app_data() -> Path:
    """The user's local AppData folder (machine-specific data such as logs)."""
    return _known_folder(_FOLDERID_LOCAL_APP_DATA)
