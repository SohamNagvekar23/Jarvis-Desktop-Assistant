"""Reply tags, hold-to-talk recording, meters, window positions and the UI files."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.brain import BrainOffline
from jarvis.listener import HoldRecorder, Listener, listen
from jarvis.system_stats import NvidiaGpu, read_meters
from jarvis.ui_state import WindowPositions
from tests.test_commands import FakeDesktop, FakeLlm, FakeLookup
from tests.test_listener import SETTINGS, FakeDetector, FakeResponder, FakeSource, FakeVad, StopListening

UI_DIR = Path(__file__).parent.parent / "src" / "jarvis" / "ui"


# --- reply tags for the log ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "llm_reply", "tag"),
    [
        ("open notepad", None, "cmd"),
        ("volume up", None, "cmd"),
        ("Don't lock the screen.", None, "cmd"),
        ("open youtube", None, "web"),
        ("open github", None, "web"),
        ("search for cats", None, "web"),
        ("play Despacito", None, "web"),
        ("search youtube for cats", None, "web"),
        ("what's the capital of Japan", '{"answer": "Tokyo."}', "brain"),
        ("my ears hurt", '{"command": "volume_down", "args": {}}', "brain"),
        ("show me dog videos", '{"command": "search_youtube", "args": {"query": "dog videos"}}', "web"),
        ("do something weird", '{"command": "format_disk"}', "brain"),
        ("...", None, "cmd"),
    ],
)
def test_reply_tags(text: str, llm_reply: str | None, tag: str) -> None:
    desktop = FakeDesktop()
    desktop.find_top_video = FakeLookup()
    llm = FakeLlm(*([llm_reply] if llm_reply else []))
    assert desktop.registry(llm).respond(text).tag == tag


def test_brain_online_flag() -> None:
    desktop = FakeDesktop()
    assert desktop.registry(FakeLlm()).respond("open notepad").brain_online is None
    assert desktop.registry(FakeLlm('{"answer": "Hi."}')).respond("hello").brain_online is True
    offline = desktop.registry(FakeLlm(BrainOffline("down"))).respond("hello")
    assert (offline.text, offline.brain_online) == ("My brain is offline.", False)


def test_handle_is_respond_text() -> None:
    desktop = FakeDesktop()
    registry = desktop.registry()
    assert registry.handle("open notepad") == registry.respond("open notepad").text == "Opening Notepad."


# --- HoldRecorder and listen() hooks -----------------------------------------------------


def test_hold_recorder_records_until_full_and_reuses_its_buffer() -> None:
    recorder = HoldRecorder(16_000, max_seconds=0.2)  # 3200 samples
    recorder.start()
    assert recorder.feed(np.full(1280, 1000, np.int16))
    assert recorder.feed(np.full(1280, 1000, np.int16))
    assert not recorder.feed(np.full(1280, 1000, np.int16))  # full: truncated
    first = recorder.audio
    assert len(first) == 3200 and first.dtype == np.float32
    assert first[0] == pytest.approx(1000 / 32768)
    recorder.start()
    assert len(recorder.audio) == 0
    recorder.feed(np.zeros(640, np.int16))
    assert np.shares_memory(first, recorder.audio)


def test_listen_between_chunks_hook_handles_requests_and_discards_audio() -> None:
    from tests.test_listener import SILENCE, WAKE, chunk

    calls = {"n": 0}

    def between() -> bool:
        calls["n"] += 1
        return calls["n"] == 2  # a typed command arrived after the second chunk

    source = FakeSource([chunk(SILENCE), chunk(WAKE), chunk(SILENCE)])
    responder = FakeResponder(source)
    listener = Listener(FakeDetector(), FakeVad(), SETTINGS)
    with pytest.raises(StopListening):
        listen(source, listener, responder, between_chunks=between)
    assert responder.events == []  # the WAKE chunk was dropped while the request ran
    assert source.discards == 1


def test_listen_stops_when_keep_listening_is_false() -> None:
    from tests.test_listener import SILENCE, chunk

    source = FakeSource([chunk(SILENCE)] * 100)
    reads = iter(range(3))
    listen(source, Listener(FakeDetector(), FakeVad(), SETTINGS), FakeResponder(source),
           keep_listening=lambda: next(reads, None) is not None)
    assert source.reads == 3


# --- system meters ---------------------------------------------------------------------


class FakeProcess:
    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=412 * 2**20)


def test_meters_with_everything_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 23.4)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=16 * 2**30))
    meters = read_meters(FakeProcess(), lambda: (3 * 2**30, 8 * 2**30), lambda: 40)  # type: ignore[arg-type]
    assert meters == {
        "cpu": {"percent": 23, "text": "23%"},
        "memory": {"percent": 3, "text": "412 MB"},
        "gpu": {"percent": 38, "text": "3.0 / 8 GB"},
        "volume": {"percent": 40, "text": "40%"},
    }


def test_meters_without_gpu_or_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    def no_volume() -> int:
        raise OSError("no audio device")

    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 5.0)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=16 * 2**30))
    meters = read_meters(FakeProcess(), lambda: None, no_volume)  # type: ignore[arg-type]
    assert meters["gpu"] is None and meters["volume"] is None


def test_gpu_meter_is_skipped_without_nvidia() -> None:
    attempts = []

    def missing_dll() -> object:
        attempts.append(1)
        raise OSError("nvml.dll not found")

    gpu = NvidiaGpu(load=missing_dll)
    assert gpu.memory() is None
    assert gpu.memory() is None
    assert attempts == [1]  # tried once, then skipped for good


def test_gpu_meter_with_a_fake_nvml() -> None:
    class FakeNvml:
        def nvmlInit_v2(self) -> int:  # noqa: N802 (NVML names)
            return 0

        def nvmlDeviceGetHandleByIndex_v2(self, index: int, handle: object) -> int:  # noqa: N802
            return 0

        def nvmlDeviceGetMemoryInfo(self, handle: object, info: object) -> int:  # noqa: N802
            info._obj.used, info._obj.total = 2 * 2**30, 6 * 2**30  # type: ignore[attr-defined]
            return 0

    assert NvidiaGpu(load=FakeNvml).memory() == (2 * 2**30, 6 * 2**30)


# --- window positions -----------------------------------------------------------------


def test_window_positions_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "Jarvis" / "ui.json"
    positions = WindowPositions(path)
    assert positions.get("main", (1440, 820), (1920, 1080)) is None  # first run
    positions.set("main", 100, 50)
    positions.set("mini", 1480, 900)
    positions.save()
    again = WindowPositions(path)
    assert again.get("main", (1440, 820), (1920, 1080)) == (100, 50)
    assert again.get("mini", (420, 96), (1920, 1080)) == (1480, 900)


@pytest.mark.parametrize("position", [(5000, 100), (-2000, 100), (100, -50), (100, 1070)])
def test_offscreen_positions_fall_back_to_the_default(tmp_path: Path, position: tuple[int, int]) -> None:
    positions = WindowPositions(tmp_path / "ui.json")
    positions.set("main", *position)
    assert positions.get("main", (1440, 820), (1920, 1080)) is None


@pytest.mark.parametrize(
    "content",
    ["not json", "[1, 2]", '{"main": [1]}', '{"main": ["a", "b"]}', '{"main": [1.5, 2]}', '{"main": [true, 2]}'],
)
def test_bad_position_files_are_ignored(tmp_path: Path, content: str) -> None:
    path = tmp_path / "ui.json"
    path.write_text(content, encoding="utf-8")
    assert WindowPositions(path).get("main", (1440, 820), (1920, 1080)) is None


# --- the UI files ---------------------------------------------------------------------


def ui_text_files() -> list[Path]:
    return [p for p in UI_DIR.iterdir() if p.suffix in {".html", ".css", ".js"}]


def test_ui_files_make_no_internet_requests() -> None:
    for path in ui_text_files():
        text = path.read_text(encoding="utf-8")
        # Only the SVG namespace URI may appear, and it is never fetched.
        urls = set(re.findall(r"https?://[^\s\"')]+", text)) - {"http://www.w3.org/2000/svg"}
        assert not urls, f"{path.name} references {urls}"
        assert "@import" not in text and "fonts.googleapis" not in text


def test_fonts_are_bundled_with_their_licenses() -> None:
    css = (UI_DIR / "style.css").read_text(encoding="utf-8")
    referenced = re.findall(r'url\("fonts/([^"]+)"\)', css)
    assert sorted(referenced) == [
        "jetbrains-mono-latin-400-normal.woff2",
        "jetbrains-mono-latin-500-normal.woff2",
        "space-grotesk-latin-400-normal.woff2",
        "space-grotesk-latin-500-normal.woff2",
        "space-grotesk-latin-700-normal.woff2",
    ]
    for name in referenced:
        assert (UI_DIR / "fonts" / name).read_bytes()[:4] == b"wOF2"
    for license_file in ("OFL-SpaceGrotesk.txt", "OFL-JetBrainsMono.txt"):
        assert "SIL OPEN FONT LICENSE" in (UI_DIR / "fonts" / license_file).read_text(encoding="utf-8")


def test_pages_lock_down_content_and_load_only_local_files() -> None:
    for page in ("index.html", "mini.html"):
        html = (UI_DIR / page).read_text(encoding="utf-8")
        assert "Content-Security-Policy" in html and "connect-src 'none'" in html
        assert re.findall(r'(?:src|href)="([^"]+)"', html) == ["style.css", "app.js"]


def test_page_elements_the_script_needs_exist() -> None:
    main = (UI_DIR / "index.html").read_text(encoding="utf-8")
    for element_id in ("orb", "stage", "status-label", "phrase", "log", "command", "hold", "confirm", "clock", "dial"):
        assert f'id="{element_id}"' in main
    for module in ("microphone", "stt", "brain", "voice", "youtube"):
        assert f'data-module="{module}"' in main
    for meter in ("cpu", "memory", "gpu", "volume"):
        assert f'data-meter="{meter}"' in main
    assert main.count("<span></span>") == 14  # level bars
    for command in ("open vs code", "take a screenshot", "lock the screen"):
        assert f'data-command="{command}"' in main
    mini = (UI_DIR / "mini.html").read_text(encoding="utf-8")
    for element_id in ("pill", "mini-orb", "status-label", "phrase"):
        assert f'id="{element_id}"' in mini


def test_animations_are_css_only() -> None:
    script = (UI_DIR / "app.js").read_text(encoding="utf-8")
    assert "requestAnimationFrame" not in script
    assert "getContext" not in script and "<canvas" not in (UI_DIR / "index.html").read_text(encoding="utf-8")
    assert script.count("setInterval(") == 1  # the clock
    assert not re.search(r"\.innerHTML|\.outerHTML|insertAdjacentHTML", script)  # textContent only


def test_pywebview_api_names_match_the_python_side() -> None:
    from jarvis.ui_app import Api

    script = (UI_DIR / "app.js").read_text(encoding="utf-8")
    called = set(re.findall(r'call\("([a-z_]+)"', script))
    exposed = {name for name in dir(Api) if not name.startswith("_")}
    assert called <= exposed, called - exposed
    assert json.loads(json.dumps(sorted(called)))  # sanity: something is called
