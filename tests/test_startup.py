"""Start with Windows, single instance, the log file and startup errors."""

import logging
import sys
import threading
import time
import uuid
import winreg
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis import app, cli, config
from jarvis.single_instance import SingleInstance
from jarvis.startup import StartupEntry, Status, WindowsRunKey, startup_command
from jarvis.ui_bridge import Controller, EventBus

COMMAND = r'"C:\Jarvis\.venv\Scripts\pythonw.exe" -m jarvis --startup'


class FakeRunKey:
    """The Run key as a dict; other programs' entries must never be touched."""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})
        self.writes = 0

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        self.writes += 1
        self.values[name] = value

    def delete(self, name: str) -> None:
        self.writes += 1
        del self.values[name]


OTHERS = {"OneDrive": '"C:\\Program Files\\Microsoft OneDrive\\OneDrive.exe" /background'}


def entry(values: dict[str, str] | None = None) -> tuple[StartupEntry, FakeRunKey]:
    run_key = FakeRunKey({**OTHERS, **(values or {})})
    return StartupEntry(run_key, "Jarvis", COMMAND), run_key


# --- the Run entry ---------------------------------------------------------------------


def test_install_adds_one_value_and_leaves_others_alone() -> None:
    startup, run_key = entry()
    assert startup.status() is Status.OFF
    assert startup.install() == "installed"
    assert run_key.values == {**OTHERS, "Jarvis": COMMAND}
    assert startup.status() is Status.ON


def test_install_again_changes_nothing() -> None:
    startup, run_key = entry({"Jarvis": COMMAND})
    assert startup.install() == "unchanged"
    assert run_key.writes == 0


def test_install_rewrites_an_entry_from_an_old_location() -> None:
    old = r'"D:\old place\.venv\Scripts\pythonw.exe" -m jarvis --startup'
    startup, run_key = entry({"Jarvis": old})
    assert startup.status() is Status.OUTDATED
    assert startup.install() == "updated"
    assert run_key.values["Jarvis"] == COMMAND
    assert run_key.values["OneDrive"] == OTHERS["OneDrive"]


def test_remove() -> None:
    startup, run_key = entry({"Jarvis": COMMAND})
    assert startup.remove() is True
    assert run_key.values == OTHERS
    assert startup.status() is Status.OFF
    assert startup.remove() is False  # nothing to remove
    assert run_key.values == OTHERS


def test_command_is_this_venvs_pythonw_absolute_and_quoted(tmp_path: Path) -> None:
    scripts = tmp_path / "my venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").touch()
    (scripts / "pythonw.exe").touch()
    command = startup_command(str(scripts / "python.exe"))
    assert command == f'"{scripts / "pythonw.exe"}" -m jarvis --startup'
    assert Path(command.split('"')[1]).is_absolute()


def test_command_for_the_real_venv() -> None:
    command = startup_command()
    executable = Path(command.split('"')[1])
    assert executable.name == "pythonw.exe" and executable.is_file() and executable.is_absolute()
    assert executable.parent == Path(sys.executable).resolve().parent
    assert command.endswith(" -m jarvis --startup")


def test_command_needs_pythonw(tmp_path: Path) -> None:
    (tmp_path / "python.exe").touch()
    with pytest.raises(FileNotFoundError, match="pythonw.exe"):
        startup_command(str(tmp_path / "python.exe"))


def test_windows_run_key_against_a_scratch_registry_key() -> None:
    # A throwaway key under HKCU\Software; the real Run key is never touched here.
    path = rf"Software\JarvisTests\{uuid.uuid4()}"
    run_key = WindowsRunKey(path)
    try:
        assert run_key.get("Jarvis") is None  # key doesn't exist yet
        run_key.set("Jarvis", COMMAND)
        assert run_key.get("Jarvis") == COMMAND
        startup = StartupEntry(run_key, "Jarvis", COMMAND)
        assert startup.status() is Status.ON
        assert startup.remove() is True
        assert run_key.get("Jarvis") is None
    finally:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, path)
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\JarvisTests")
        except OSError:
            pass  # another test run's key is still there


def test_config_uses_the_per_user_run_key() -> None:
    assert config.STARTUP_RUN_KEY == r"Software\Microsoft\Windows\CurrentVersion\Run"
    assert config.STARTUP_VALUE_NAME == "Jarvis"


# --- the command line ----------------------------------------------------------------


@pytest.fixture
def no_side_effects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """main() with logging in tmp_path, a fake Run key and nothing heavy loaded."""
    state = SimpleNamespace(run_key=FakeRunKey(OTHERS), loaded=[], instances=[])
    monkeypatch.setattr(cli, "log_file", lambda: tmp_path / "jarvis.log")
    monkeypatch.setattr(cli, "startup_entry", lambda: StartupEntry(state.run_key, "Jarvis", COMMAND))
    monkeypatch.setattr(app, "load_transcriber", lambda: state.loaded.append("models"))
    monkeypatch.setattr(app, "run_with_ui", lambda **kwargs: state.loaded.append(("ui", kwargs)))
    yield state
    logging.getLogger().handlers.clear()  # release the log file


def test_install_startup_flag(no_side_effects: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--install-startup"])
    assert exit_info.value.code == 0
    assert no_side_effects.run_key.values["Jarvis"] == COMMAND
    assert "Jarvis will now start with Windows." in capsys.readouterr().out
    assert no_side_effects.loaded == []  # no models, no UI


def test_install_startup_flag_rewrites_a_moved_venv(
    no_side_effects: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    no_side_effects.run_key.values["Jarvis"] = '"D:\\old\\pythonw.exe" -m jarvis --startup'
    with pytest.raises(SystemExit):
        cli.main(["--install-startup"])
    assert no_side_effects.run_key.values["Jarvis"] == COMMAND
    assert "Updated the startup entry" in capsys.readouterr().out


def test_remove_startup_flag(no_side_effects: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> None:
    no_side_effects.run_key.values["Jarvis"] = COMMAND
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--remove-startup"])
    assert exit_info.value.code == 0
    assert no_side_effects.run_key.values == OTHERS
    assert "no longer start with Windows" in capsys.readouterr().out


def test_startup_flag_errors_exit_nonzero(
    no_side_effects: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> StartupEntry:
        raise FileNotFoundError("pythonw.exe not found next to python.exe")

    monkeypatch.setattr(cli, "startup_entry", broken)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--install-startup"])
    assert exit_info.value.code == 1


@pytest.mark.parametrize("flags", [["--startup", "--no-ui"], ["--install-startup", "--remove-startup"]])
def test_conflicting_flags_are_rejected(flags: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(flags)
    assert exit_info.value.code == 2


def test_startup_flag_opens_the_ui_at_login(
    no_side_effects: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "SINGLE_INSTANCE_MUTEX", rf"Local\JarvisTest.{uuid.uuid4()}")
    cli.main(["--startup"])
    ((kind, kwargs),) = no_side_effects.loaded
    assert kind == "ui" and kwargs["at_login"] is True
    assert kwargs["startup"].command == COMMAND


def test_second_launch_signals_the_first_and_exits(
    no_side_effects: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    names = (rf"Local\JarvisTest.{uuid.uuid4()}", rf"Local\JarvisTest.{uuid.uuid4()}")
    monkeypatch.setattr(config, "SINGLE_INSTANCE_MUTEX", names[0])
    monkeypatch.setattr(config, "SINGLE_INSTANCE_EVENT", names[1])
    first = SingleInstance(*names)
    assert first.acquire()
    activated = threading.Event()
    stop = threading.Event()
    watcher = threading.Thread(target=first.watch_activations, args=(activated.set, stop))
    watcher.start()
    try:
        cli.main([])  # the second launch
        assert activated.wait(2), "the running copy wasn't asked to come to the front"
    finally:
        stop.set()
        watcher.join(2)
        first.release()
    assert no_side_effects.loaded == []  # it didn't start a second UI
    assert "Jarvis is already running." in capsys.readouterr().out


def test_the_single_instance_check_comes_before_heavy_imports() -> None:
    # Fresh interpreter: everything cli.main needs before the mutex must be light,
    # or two launches close together at a busy login could both start.
    import subprocess

    code = (
        "import sys, jarvis.cli; "
        "print(sorted(m for m in ('jarvis.app', 'faster_whisper', 'webview', 'pyttsx3', 'pycaw', "
        "'sounddevice', 'numpy', 'mss', 'yt_dlp') if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


# --- single instance (real named Win32 objects, unique names per test) -----------------


@pytest.fixture
def names() -> tuple[str, str]:
    return rf"Local\JarvisTest.{uuid.uuid4()}", rf"Local\JarvisTest.{uuid.uuid4()}"


def test_only_one_instance(names: tuple[str, str]) -> None:
    first, second = SingleInstance(*names), SingleInstance(*names)
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()  # free again once the first copy is gone
    second.release()


def test_a_stale_error_code_is_not_mistaken_for_already_running(names: tuple[str, str]) -> None:
    import ctypes

    ctypes.set_last_error(183)  # ERROR_ALREADY_EXISTS left over from something else
    instance = SingleInstance(*names)
    assert instance.acquire()
    instance.release()


def test_signal_without_a_running_copy(names: tuple[str, str]) -> None:
    assert SingleInstance(*names).signal_existing() is False


def test_each_signal_activates_once(names: tuple[str, str]) -> None:
    first = SingleInstance(*names)
    assert first.acquire()
    count: list[int] = []
    stop = threading.Event()
    watcher = threading.Thread(target=first.watch_activations, args=(lambda: count.append(1), stop))
    watcher.start()
    try:
        for _ in range(3):
            assert SingleInstance(*names).signal_existing()
            time.sleep(0.4)
    finally:
        stop.set()
        watcher.join(2)
        first.release()
    assert len(count) == 3
    assert not watcher.is_alive()


# --- the log file ---------------------------------------------------------------------


def test_log_file_rotates_at_the_cap(tmp_path: Path) -> None:
    from jarvis.logs import setup_logging

    log_file = tmp_path / "Jarvis" / "jarvis.log"
    setup_logging(log_file, max_bytes=2_000, backup_count=2)
    try:
        log = logging.getLogger("jarvis.test")
        log.info("first line")
        for i in range(200):
            log.info("line %d %s", i, "x" * 40)
        logging.getLogger().handlers[0].flush()
    finally:
        handlers = list(logging.getLogger().handlers)
        logging.getLogger().handlers.clear()
        for handler in handlers:
            handler.close()
    files = sorted(p.name for p in log_file.parent.iterdir())
    assert files == ["jarvis.log", "jarvis.log.1", "jarvis.log.2"]  # capped: 3 files at most
    assert all(p.stat().st_size <= 2_100 for p in log_file.parent.iterdir())
    assert "line 199" in log_file.read_text(encoding="utf-8")


def test_missing_console_streams_are_replaced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from jarvis.logs import setup_logging

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    setup_logging(tmp_path / "jarvis.log", 10_000, 1)
    try:
        print("pythonw has no console")  # must not crash
        sys.stderr.write("progress bar\n")
        logging.getLogger("jarvis").warning("still logged to the file")
        handlers = logging.getLogger().handlers
        assert len(handlers) == 1  # no console handler for a devnull stderr
        handlers[0].flush()
    finally:
        handlers = list(logging.getLogger().handlers)
        logging.getLogger().handlers.clear()
        for handler in handlers:
            handler.close()
    assert "still logged to the file" in (tmp_path / "jarvis.log").read_text(encoding="utf-8")


# --- starting the assistant behind the UI --------------------------------------------


def events(bus: EventBus) -> list[dict]:
    collected = []
    while (event := bus.next_event(timeout=0)) is not None:
        collected.append(event)
    return collected


def test_startup_errors_are_shown_in_the_ui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def broken() -> None:
        raise RuntimeError("no speech model")

    monkeypatch.setattr(app, "load_transcriber", broken)
    monkeypatch.setattr(cli, "log_file", lambda: tmp_path / "jarvis.log")
    bus = EventBus()
    app.run_assistant_for_ui(bus, Controller(bus))
    sent = events(bus)
    (error,) = [e["value"] for e in sent if e["type"] == "error"]
    assert "Jarvis couldn't start: no speech model" in error
    assert str(tmp_path / "jarvis.log") in error
    assert {"name": "stt", "level": "error", "text": "FAILED"} in [
        e["value"] for e in sent if e["type"] == "module"
    ]


def test_login_delay_comes_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_at: list[float] = []

    def record() -> None:
        loaded_at.append(time.monotonic())
        raise RuntimeError("stop here")

    monkeypatch.setattr(app, "load_transcriber", record)
    bus = EventBus()
    started = time.monotonic()
    app.run_assistant_for_ui(bus, Controller(bus), delay_seconds=0.3)
    assert loaded_at and loaded_at[0] - started >= 0.3


def test_closing_during_the_login_delay_loads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "load_transcriber", lambda: pytest.fail("models loaded after quit"))
    bus = EventBus()
    controller = Controller(bus)
    threading.Timer(0.1, controller.quit).start()
    started = time.monotonic()
    app.run_assistant_for_ui(bus, controller, delay_seconds=30)
    assert time.monotonic() - started < 2


# --- the UI shell: settings toggle and which window opens first ----------------------


def test_settings_toggle() -> None:
    from jarvis.ui_app import Api

    startup, run_key = entry()
    bus = EventBus()
    api = Api(bus, Controller(bus), windows=None, startup=startup)  # type: ignore[arg-type]
    assert api.startup_status() == "off"
    assert api.set_startup(True) == "on"
    assert run_key.values["Jarvis"] == COMMAND
    assert api.set_startup("yes") == "on"  # anything but a real bool changes nothing
    assert api.set_startup(False) == "off"
    assert run_key.values == OTHERS
    run_key.values["Jarvis"] = "old"
    assert api.startup_status() == "outdated"
    assert api.set_startup(True) == "on"  # switching on fixes it


def test_settings_toggle_unavailable_without_pythonw() -> None:
    from jarvis.ui_app import Api

    bus = EventBus()
    api = Api(bus, Controller(bus), windows=None)  # type: ignore[arg-type]
    assert api.startup_status() == "unavailable"
    assert api.set_startup(True) == "unavailable"


class FakeEvent:
    def __init__(self) -> None:
        self.handlers: list = []

    def __iadd__(self, handler):  # type: ignore[no-untyped-def]
        self.handlers.append(handler)
        return self


class FakeWebviewWindow:
    def __init__(self, title: str, **kwargs: object) -> None:
        self.title = title
        self.kwargs = kwargs
        self.events = SimpleNamespace(moved=FakeEvent(), closed=FakeEvent(), shown=FakeEvent())
        self.calls: list[str] = []

    def show(self) -> None:
        self.calls.append("show")

    def hide(self) -> None:
        self.calls.append("hide")


@pytest.mark.parametrize("at_login", [False, True])
def test_which_window_opens_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, at_login: bool) -> None:
    import webview

    from jarvis import ui_app
    from jarvis.ui_state import WindowPositions

    created: dict[str, FakeWebviewWindow] = {}

    def create_window(title: str, **kwargs: object) -> FakeWebviewWindow:
        created[title] = FakeWebviewWindow(title, **kwargs)
        return created[title]

    monkeypatch.setattr(webview, "create_window", create_window)
    monkeypatch.setattr(webview, "screens", [SimpleNamespace(width=1536, height=864)])
    bus = EventBus()
    windows = ui_app.JarvisWindows(bus, Controller(bus), WindowPositions(tmp_path / "ui.json"), at_login)
    windows.create(ui_app.Api(bus, Controller(bus), windows))
    assert created["Jarvis"].kwargs["hidden"] is at_login
    assert created["Jarvis mini"].kwargs["hidden"] is (not at_login)
    assert created["Jarvis"].kwargs["width"] <= 1536 - 32  # clamped to this screen


def test_activation_shows_the_full_window_in_front(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from jarvis import ui_app, window_shape
    from jarvis.ui_state import WindowPositions

    fronted: list[int] = []
    monkeypatch.setattr(window_shape, "bring_to_front", fronted.append)
    bus = EventBus()
    windows = ui_app.JarvisWindows(bus, Controller(bus), WindowPositions(tmp_path / "ui.json"), True)
    main, mini = FakeWebviewWindow("Jarvis"), FakeWebviewWindow("Jarvis mini")
    main.native = SimpleNamespace(Handle=SimpleNamespace(ToInt64=lambda: 4242))  # type: ignore[attr-defined]
    windows._main, windows._mini = main, mini
    windows.activate()
    assert mini.calls == ["hide"] and main.calls == ["show"]
    assert fronted == [4242]
