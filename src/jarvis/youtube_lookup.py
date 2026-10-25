"""Find the top YouTube video for a query with yt-dlp (flat search, no download)."""

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Returns yt-dlp's info dict for a "ytsearch1:..." URL.
Extractor = Callable[[str, float], Any]


def _yt_dlp_extract(url: str, timeout_seconds: float) -> Any:
    import yt_dlp  # lazily: ~25 MB, loaded only the first time "play X" is used

    options = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
        "extract_flat": True,  # search results only, no per-video page fetches
        "socket_timeout": timeout_seconds,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.extract_info(url, download=False)


class YouTubeLookup:
    def __init__(
        self,
        timeout_seconds: float,
        extract: Extractor = _yt_dlp_extract,
        on_status: Callable[[bool], None] = lambda ok: None,
    ) -> None:
        self._timeout = timeout_seconds
        self._extract = extract
        self._on_status = on_status  # True: lookups work (even if empty); False: failed

    def top_video_id(self, query: str) -> str | None:
        """The first search result's video id (not validated here), or None if the
        lookup failed, timed out or found nothing."""
        result: list[Any] = []

        def run() -> None:
            try:
                result.append(self._extract(f"ytsearch1:{query}", self._timeout))
            except Exception as error:  # yt-dlp raises many types; any failure falls back
                logger.warning("YouTube lookup failed: %s", error)

        # socket_timeout only bounds each network read, so enforce a total deadline.
        # A lookup still running afterwards is abandoned; it's a daemon thread.
        worker = threading.Thread(target=run, name="youtube-lookup", daemon=True)
        worker.start()
        worker.join(self._timeout)
        if worker.is_alive():
            logger.warning("YouTube lookup timed out after %g s", self._timeout)
            self._on_status(False)
            return None
        self._on_status(bool(result))
        if not result or not isinstance(result[0], dict):
            return None
        entries = result[0].get("entries")
        if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
            return None
        video_id = entries[0].get("id")
        return video_id if isinstance(video_id, str) else None
