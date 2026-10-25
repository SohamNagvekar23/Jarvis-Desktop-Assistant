import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from jarvis.youtube_lookup import YouTubeLookup


class FakeExtractor:
    def __init__(self, result: Any = None, error: Exception | None = None, delay: float = 0.0) -> None:
        self.result = result
        self.error = error
        self.delay = delay
        self.calls: list[tuple[str, float]] = []
        self.finished = threading.Event()

    def __call__(self, url: str, timeout: float) -> Any:
        self.calls.append((url, timeout))
        time.sleep(self.delay)
        self.finished.set()
        if self.error:
            raise self.error
        return self.result


def playlist(*entries: Any) -> dict[str, Any]:
    return {"_type": "playlist", "entries": list(entries)}


def test_returns_the_first_result_id() -> None:
    extract = FakeExtractor(playlist({"id": "kJQP7kiw5Fk", "title": "Despacito"}, {"id": "other123456"}))
    assert YouTubeLookup(5.0, extract).top_video_id("Despacito") == "kJQP7kiw5Fk"
    assert extract.calls == [("ytsearch1:Despacito", 5.0)]  # one result, timeout passed on


def test_query_is_passed_through_as_a_search() -> None:
    extract = FakeExtractor(playlist({"id": "0CACYW9SQMo"}))
    YouTubeLookup(5.0, extract).top_video_id("https://evil.example/x & more")
    assert extract.calls[0][0] == "ytsearch1:https://evil.example/x & more"  # still a search


@pytest.mark.parametrize(
    "result",
    [
        playlist(),  # no results
        {"_type": "playlist"},
        {"entries": None},
        {"entries": "kJQP7kiw5Fk"},
        playlist("kJQP7kiw5Fk"),
        playlist({"title": "no id"}),
        playlist({"id": None}),
        playlist({"id": 12345678901}),
        None,
        [],
        "kJQP7kiw5Fk",
    ],
)
def test_empty_or_unexpected_results_give_none(result: Any) -> None:
    assert YouTubeLookup(5.0, FakeExtractor(result)).top_video_id("x") is None


def test_ids_are_returned_unvalidated_for_the_caller_to_check() -> None:
    # Validation (exactly 11 of [A-Za-z0-9_-]) happens where the URL is built.
    extract = FakeExtractor(playlist({"id": "../../evil"}))
    assert YouTubeLookup(5.0, extract).top_video_id("x") == "../../evil"


@pytest.mark.parametrize("error", [OSError("network down"), RuntimeError("yt-dlp broke"), KeyError("id")])
def test_errors_give_none(error: Exception, caplog: pytest.LogCaptureFixture) -> None:
    assert YouTubeLookup(5.0, FakeExtractor(error=error)).top_video_id("x") is None
    assert "YouTube lookup failed" in caplog.text


def test_timeout_gives_none_without_waiting_for_the_lookup(caplog: pytest.LogCaptureFixture) -> None:
    extract = FakeExtractor(playlist({"id": "kJQP7kiw5Fk"}), delay=2.0)
    started = time.monotonic()
    assert YouTubeLookup(0.2, extract).top_video_id("x") is None
    assert time.monotonic() - started < 1.0
    assert "timed out after 0.2 s" in caplog.text
    assert not extract.finished.is_set()  # abandoned, still running in a daemon thread


def test_yt_dlp_is_imported_lazily() -> None:
    # Fresh interpreter: starting Jarvis must not load yt-dlp (~25 MB).
    code = "import sys, jarvis.app, jarvis.youtube_lookup; print('yt_dlp' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
