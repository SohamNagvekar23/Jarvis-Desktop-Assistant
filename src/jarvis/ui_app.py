"""The desktop UI: two frameless pywebview windows on Edge WebView2.

The windows run on the main thread (pywebview requires it). The assistant runs
on a background thread and talks to the pages only through the EventBus
(Python -> JS, via evaluate_js) and the Api object (JS -> Python).
"""

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil
import webview

from jarvis import config
from jarvis.single_instance import SingleInstance
from jarvis.startup import StartupEntry
from jarvis.system_stats import NvidiaGpu, read_meters
from jarvis.ui_bridge import Controller, EventBus, EventPump
from jarvis.ui_state import WindowPositions

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).parent / "ui"


class JarvisWindows:
    def __init__(
        self,
        bus: EventBus,
        controller: Controller,
        positions: WindowPositions,
        start_minimized: bool = False,
    ) -> None:
        self._bus = bus
        self._controller = controller
        self._positions = positions
        self._start_minimized = start_minimized  # --startup: open as the mini pill
        self._main: Any = None
        self._mini: Any = None
        self._maximized = False
        self._quitting = threading.Event()
        self._meters_stop = threading.Event()
        self.pump = EventPump(bus, self._deliver)

    def create(self, api: "Api") -> None:
        screen = webview.screens[0]
        screen_size = (screen.width, screen.height)
        mini_size = config.UI_MINI_SIZE
        # 1440x820 by default, but never bigger than the screen (at 125% scaling a
        # 1920x1080 display is only 1536x864), and centered unless it was moved.
        main_size = (
            min(config.UI_MAIN_SIZE[0], screen.width - 32),
            min(config.UI_MAIN_SIZE[1], screen.height - 72),
        )
        main_at = self._positions.get("main", main_size, screen_size) or (
            (screen.width - main_size[0]) // 2,
            max(0, (screen.height - main_size[1]) // 2 - 20),
        )
        mini_at = self._positions.get("mini", mini_size, screen_size) or (
            screen.width - mini_size[0] - config.UI_MINI_MARGIN[0],
            screen.height - mini_size[1] - config.UI_MINI_MARGIN[1],
        )
        self._main = webview.create_window(
            "Jarvis",
            url=str(UI_DIR / "index.html"),
            js_api=api,
            width=main_size[0],
            height=main_size[1],
            x=main_at[0],
            y=main_at[1],
            min_size=config.UI_MAIN_MIN_SIZE,
            frameless=True,
            easy_drag=False,  # only the title bar drags (.pywebview-drag-region)
            background_color="#060A11",
            hidden=self._start_minimized,
        )
        self._mini = webview.create_window(
            "Jarvis mini",
            url=str(UI_DIR / "mini.html"),
            js_api=api,
            width=mini_size[0],
            height=mini_size[1],
            x=mini_at[0],
            y=mini_at[1],
            resizable=False,
            frameless=True,
            easy_drag=False,
            on_top=True,
            hidden=not self._start_minimized,
            shadow=False,
            background_color="#080E16",
        )
        self._main.events.moved += lambda x, y: self._positions.set("main", x, y)
        self._mini.events.moved += lambda x, y: self._positions.set("mini", x, y)
        self._main.events.closed += self.quit  # e.g. Alt+F4
        self._mini.events.shown += self._shape_mini

    def start_background(
        self, assistant: Callable[[], None], instance: SingleInstance | None = None
    ) -> None:
        """Runs once the GUI loop is up (pywebview's start(func))."""
        self.pump.start()
        threading.Thread(target=self._meters_loop, name="ui-meters", daemon=True).start()
        threading.Thread(target=assistant, name="assistant", daemon=True).start()
        if instance is not None:  # a second launch asks us to come to the front
            threading.Thread(
                target=instance.watch_activations,
                args=(self.activate, self._meters_stop),
                name="single-instance",
                daemon=True,
            ).start()

    # --- window actions (called from the pages through Api) ---

    def to_mini(self) -> None:
        self._main.hide()
        self._mini.show()
        self._shape_mini()

    def _shape_mini(self) -> None:
        """Round the pill's ends (a Windows window region; no transparency needed)."""
        from jarvis.window_shape import make_capsule

        try:
            make_capsule(int(self._mini.native.Handle.ToInt64()))
        except Exception:  # cosmetic: a square pill still works
            logger.exception("Couldn't round the mini window")

    def to_main(self) -> None:
        self._mini.hide()
        self._main.show()

    def activate(self) -> None:
        """Show the full window in front of everything (Jarvis was launched again)."""
        from jarvis.window_shape import bring_to_front

        self.to_main()
        try:
            bring_to_front(int(self._main.native.Handle.ToInt64()))
        except Exception:
            logger.exception("Couldn't bring the window to the front")

    def toggle_maximize(self) -> None:
        if self._maximized:
            self._main.restore()
        else:
            self._main.maximize()
        self._maximized = not self._maximized

    def quit(self) -> None:
        if self._quitting.is_set():
            return
        self._quitting.set()
        self._controller.quit()
        self._meters_stop.set()
        self.pump.stop()
        self._positions.save()
        for window in (self._mini, self._main):
            try:
                window.destroy()
            except Exception:  # already gone
                pass

    # --- background ---

    def _deliver(self, script: str) -> None:
        for window in (self._main, self._mini):
            if not self._quitting.is_set():
                window.evaluate_js(script)

    def _meters_loop(self) -> None:
        import comtypes  # this thread reads the volume through COM

        comtypes.CoInitialize()
        from jarvis.volume import MasterVolume

        volume, gpu, process = MasterVolume(), NvidiaGpu(), psutil.Process()
        psutil.cpu_percent(interval=None)  # the first reading primes the counter
        delay = 0.5
        while not self._meters_stop.wait(delay):
            try:
                self._bus.meters(read_meters(process, gpu.memory, volume.get_level))
            except Exception:
                logger.exception("Couldn't read the system meters")
            delay = config.UI_METER_INTERVAL_SECONDS


class Api:
    """Methods the pages call as window.pywebview.api.<name>(...).

    Only public methods are exposed; everything else is underscore-prefixed so
    pywebview doesn't walk into it.
    """

    def __init__(
        self,
        bus: EventBus,
        controller: Controller,
        windows: JarvisWindows,
        startup: StartupEntry | None = None,
    ) -> None:
        self._bus = bus
        self._controller = controller
        self._windows = windows
        self._startup = startup

    def snapshot(self) -> dict[str, Any]:
        return self._bus.snapshot()

    def settings(self) -> dict[str, Any]:
        return {
            "wakeWord": "Hey Jarvis",
            "wakeThreshold": config.WAKE_THRESHOLD,
            "speechModel": config.WHISPER_MODEL,
            "brainModel": config.BRAIN_MODEL,
            "brainUrl": config.OLLAMA_URL,
            "volumeStep": config.VOLUME_STEP_PERCENT,
            "screenshots": f"Pictures\\{config.SCREENSHOT_SUBFOLDER}",
        }

    def startup_status(self) -> str:
        """"on", "off", "outdated" (the entry points elsewhere) or "unavailable"."""
        if self._startup is None:
            return "unavailable"
        return self._startup.status().value

    def set_startup(self, enabled: object) -> str:
        """Turn "start with Windows" on or off; returns the new status."""
        if self._startup is None:
            return "unavailable"
        try:
            if enabled is True:
                result = self._startup.install()
                logger.info("Start with Windows: %s (%s)", result, self._startup.command)
            elif enabled is False:
                self._startup.remove()
                logger.info("Start with Windows: removed")
        except OSError:
            logger.exception("Couldn't change the startup entry")
        return self._startup.status().value

    def submit(self, text: object) -> None:
        if isinstance(text, str):
            self._controller.submit_text(text[: config.SEARCH_MAX_QUERY_CHARS + 100])

    def hold_start(self) -> None:
        self._controller.hold_start()

    def hold_end(self) -> None:
        self._controller.hold_end()

    def toggle_mute(self) -> bool:
        return self._controller.toggle_mute()

    def minimize(self) -> None:
        self._windows.to_mini()

    def restore(self) -> None:
        self._windows.to_main()

    def maximize(self) -> None:
        self._windows.toggle_maximize()

    def close(self) -> None:
        self._windows.quit()


def run_ui(
    bus: EventBus,
    controller: Controller,
    assistant: Callable[[], None],
    *,
    start_minimized: bool = False,
    startup: StartupEntry | None = None,
    instance: SingleInstance | None = None,
) -> None:
    """Open the UI and block until it's closed. assistant runs on its own thread."""
    from jarvis import known_folders

    positions = WindowPositions(
        known_folders.roaming_app_data() / config.UI_STATE_SUBFOLDER / config.UI_STATE_FILENAME
    )
    windows = JarvisWindows(bus, controller, positions, start_minimized)
    windows.create(Api(bus, controller, windows, startup))
    webview.start(windows.start_background, (assistant, instance), gui="edgechromium")
    windows.quit()  # also when the GUI loop ended some other way
