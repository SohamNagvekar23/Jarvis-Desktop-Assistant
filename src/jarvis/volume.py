"""Master volume of the default playback device (Windows Core Audio via pycaw)."""

from collections.abc import Callable
from typing import Any, TypeVar

from comtypes import COMError
from pycaw.pycaw import AudioUtilities

_T = TypeVar("_T")


def _com_call(call: Callable[[Any], _T]) -> _T:
    """Run call on the current default speakers' endpoint; raise OSError on failure.

    The device is looked up on every call (a few COM calls, well under a
    millisecond) so switching to headphones while Jarvis runs still works.
    """
    try:
        return call(AudioUtilities.GetSpeakers().EndpointVolume)
    except (COMError, AttributeError) as error:
        raise OSError(f"volume control failed: {error}") from error


class MasterVolume:
    def get_level(self) -> int:
        """Current volume, 0 to 100."""
        return round(_com_call(lambda endpoint: endpoint.GetMasterVolumeLevelScalar()) * 100)

    def set_level(self, percent: int) -> None:
        _com_call(lambda endpoint: endpoint.SetMasterVolumeLevelScalar(percent / 100, None))

    def set_muted(self, muted: bool) -> None:
        _com_call(lambda endpoint: endpoint.SetMute(int(muted), None))
