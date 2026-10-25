import time
from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis import config
from jarvis.apps import App, CloseResult
from jarvis.brain import BrainOffline, BrainTimeout
from jarvis.commands import Command, CommandRegistry, build_registry

APPS_BY_NAME = {app.name: app for app in config.APPS}


class FakeVolume:
    def __init__(self, level: int = 50, muted: bool = True, fail: bool = False) -> None:
        self.level = level
        self.muted = muted
        self.fail = fail
        self.set_calls: list[int] = []

    def get_level(self) -> int:
        if self.fail:
            raise OSError("no audio device")
        return self.level

    def set_level(self, percent: int) -> None:
        if self.fail:
            raise OSError("no audio device")
        self.set_calls.append(percent)
        self.level = percent

    def set_muted(self, muted: bool) -> None:
        if self.fail:
            raise OSError("no audio device")
        self.muted = muted


class FakeAppWindows:
    def __init__(self) -> None:
        self.open: set[str] = set()
        self.close_results: list[CloseResult] = []  # next results of close(); default CLOSED
        self.force_result = CloseResult.CLOSED
        self.calls: list[tuple[str, str]] = []
        self.fail = False

    def is_open(self, app: App) -> bool:
        if self.fail:
            raise OSError("enumeration failed")
        return app.name in self.open

    def close(self, app: App) -> CloseResult:
        self.calls.append(("close", app.name))
        result = self.close_results.pop(0) if self.close_results else CloseResult.CLOSED
        if result is CloseResult.CLOSED:
            self.open.discard(app.name)
        return result

    def force_close(self, app: App) -> CloseResult:
        self.calls.append(("force_close", app.name))
        return self.force_result


class FakeDesktop:
    """Records every OS call made by the command handlers."""

    def __init__(self, volume: FakeVolume | None = None, apps: tuple[App, ...] = config.APPS) -> None:
        self.volume = volume or FakeVolume()
        self.apps = apps
        self.windows = FakeAppWindows()
        self.launched: list[str] = []
        self.urls: list[str] = []
        self.media_keys: list[str] = []
        self.find_top_video: Callable[[str], str | None] | None = None
        self.answers: list[bool] = []  # scripted confirmation answers
        self.questions: list[str] = []
        self.screenshots = 0
        self.locks = 0
        self.fail: set[str] = set()

    def launch(self, app: App) -> None:
        if "launch" in self.fail:
            raise FileNotFoundError(app.launch)
        self.launched.append(app.name)

    def press_media_key(self, name: str) -> None:
        if "media" in self.fail:
            raise OSError("SendInput failed")
        self.media_keys.append(name)

    def open_url(self, url: str) -> None:
        if "browser" in self.fail:
            raise OSError("no browser")
        self.urls.append(url)

    def confirm(self, question: str) -> bool:
        self.questions.append(question)
        return self.answers.pop(0)

    def take_screenshot(self) -> Path:
        if "screenshot" in self.fail:
            raise OSError("capture failed")
        self.screenshots += 1
        return Path("C:/Pictures/Jarvis/screenshot_2026-10-25_12-00-00.png")

    def lock(self) -> None:
        if "lock" in self.fail:
            raise PermissionError("denied")
        self.locks += 1

    def registry(self, llm: "FakeLlm | None" = None) -> CommandRegistry:
        return build_registry(
            llm=llm,
            apps=self.apps,
            launch_app=self.launch,
            app_windows=self.windows,
            confirm=self.confirm,
            volume=self.volume,
            take_screenshot=self.take_screenshot,
            lock_screen=self.lock,
            press_media_key=self.press_media_key,
            sites=config.SITES,
            open_url=self.open_url,
            find_top_video=self.find_top_video,
        )

    def nothing_ran(self) -> bool:
        return (
            not self.launched
            and not self.urls
            and not self.media_keys
            and not self.windows.calls
            and not self.questions
            and not self.volume.set_calls
            and self.screenshots == 0
            and self.locks == 0
        )


@pytest.fixture
def desktop() -> FakeDesktop:
    return FakeDesktop()


# Every trigger phrase, per command. Must match the registry exactly, so adding a
# phrase without a test (or removing one) fails test_phrase_table_is_complete.
PHRASES: dict[str, tuple[str, ...]] = {
    "open_app": ("open", "launch", "start"),
    "close_app": ("close", "quit", "exit"),
    "search_google": ("search for", "google", "search google for", "look up"),
    "search_youtube": ("search youtube for", "youtube", "find * on youtube"),
    "play_youtube": ("play * on youtube", "play *", "put on *"),
    "media_play_pause": ("pause", "pause the music", "resume", "play", "play the music", "continue playing"),
    "media_next": (
        "next",
        "next song",
        "next track",
        "skip",
        "skip this song",
        "change the song",
        "change it",
        "i don't like this song",
        "don't like this song",
        "can't stand this song",
    ),
    "media_previous": ("previous", "previous song", "go back a song", "last track"),
    "media_stop": ("stop the music", "stop playing"),
    "volume_up": (
        "volume up",
        "turn the volume up",
        "turn volume up",
        "turn up the volume",
        "turn up volume",
        "increase the volume",
        "increase volume",
        "raise the volume",
        "raise volume",
        "louder",
        "crank it up",
        "turn it up",
        "pump it up",
    ),
    "volume_down": (
        "volume down",
        "turn the volume down",
        "turn volume down",
        "turn down the volume",
        "turn down volume",
        "decrease the volume",
        "decrease volume",
        "lower the volume",
        "lower volume",
        "quieter",
    ),
    "set_volume": (
        "set volume",
        "set the volume",
        "change volume",
        "change the volume",
        "volume to",
        "volume at",
    ),
    "mute": (
        "mute",
        "mute volume",
        "mute the volume",
        "mute sound",
        "mute the sound",
        "mute audio",
        "mute the audio",
        "silence",
        "silence please",
        "quiet",
    ),
    "unmute": (
        "unmute",
        "unmute volume",
        "unmute the volume",
        "unmute sound",
        "unmute the sound",
        "unmute audio",
        "unmute the audio",
        "mute off",
        "turn off mute",
        "turn mute off",
        "turn off the mute",
        "turn the mute off",
    ),
    "screenshot": (
        "screenshot",
        "take a screenshot",
        "take screenshot",
        "capture the screen",
        "capture screen",
        "print screen",
    ),
    "lock_screen": (
        "lock the screen",
        "lock screen",
        "lock my screen",
        "lock the computer",
        "lock my computer",
        "lock computer",
        "lock the pc",
        "lock my pc",
        "lock pc",
    ),
}
ALL_PHRASES = [(name, phrase) for name, phrases in PHRASES.items() for phrase in phrases]


def test_phrase_table_is_complete(desktop: FakeDesktop) -> None:
    registered = {c.name: set(c.triggers) for c in desktop.registry().commands}
    assert registered == {name: set(phrases) for name, phrases in PHRASES.items()}


@pytest.mark.parametrize(("expected", "phrase"), ALL_PHRASES)
@pytest.mark.parametrize(
    "template",
    ["{}", "{}.", "{}!", "Hey Jarvis, {}, please.", "jarvis {} now"],
    ids=["bare", "period", "bang", "wrapped", "suffix"],
)
def test_every_phrase_matches_its_command(
    desktop: FakeDesktop, expected: str, phrase: str, template: str
) -> None:
    phrase = phrase.replace("*", "cats")  # fill "*" slots with a real query
    spoken = template.format(phrase[0].upper() + phrase[1:])
    found = desktop.registry().match(spoken)
    assert found is not None
    assert found.command.name == expected
    assert not found.negated


@pytest.mark.parametrize(
    "text",
    [
        "Un-mute.",
        "un mute",
        "Un mute the volume, please.",
        "UNMUTE THE AUDIO",
        "Turn off the mute.",
        "Hey Jarvis, turn the mute off please",
        "mute off",
        "unmute sound now",
    ],
)
def test_unmute_variants_never_match_mute(desktop: FakeDesktop, text: str) -> None:
    found = desktop.registry().match(text)
    assert found is not None
    assert found.command.name == "unmute"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "what time is it",
        "reopen chromebook",
        "notepad",
        "volume",
        "what's the volume",
        "volume upgrade",
        "turn up the music",
        "wait a minute",
        "my commute",
        "computer",
        "lock",
        "unlock the screen",
        "padlock screen",
        "screen",
        "print the screenplay",
        "mutex",
        "muted",
        "set",
        "the door is closed",
        "restart",
        "an opener",
        "closet",
        "exited",
    ],
)
def test_near_misses_do_not_match(desktop: FakeDesktop, text: str) -> None:
    assert desktop.registry().match(text) is None


@pytest.mark.parametrize(
    ("text", "expected", "rest"),
    [
        # The phrase that starts earliest wins...
        ("Close the lock screen settings", "close_app", "the lock screen settings"),
        ("Lock the screen and take a screenshot", "lock_screen", "and take a screenshot"),
        ("Open notepad and close chrome", "open_app", "notepad and close chrome"),
        ("Turn off mute", "unmute", ""),
        ("Turn the volume up to fifty", "volume_up", "to fifty"),  # handled as set, see below
        ("Set the volume to 40", "set_volume", "to 40"),  # not "volume to"
        # ...and length breaks ties between phrases starting at the same word.
        ("Mute off", "unmute", ""),
        ("Mute the audio", "mute", ""),
        ("Take a screenshot", "screenshot", ""),
        # "screen shot" -> "screenshot", so "lock screen" isn't there at all.
        ("Lock screen shot", "screenshot", ""),
    ],
)
def test_overlapping_phrases(desktop: FakeDesktop, text: str, expected: str, rest: str) -> None:
    found = desktop.registry().match(text)
    assert found is not None
    assert found.command.name == expected
    assert found.rest == rest


def test_close_the_lock_screen_settings_does_not_lock_the_screen(desktop: FakeDesktop) -> None:
    desktop.windows.open.add("Settings")
    reply = desktop.registry().handle("Close the lock screen settings.")
    assert desktop.locks == 0
    assert reply == "Sorry, I don't know that app."  # "lock screen settings" isn't an app name
    assert desktop.nothing_ran()


def test_duplicate_triggers_rejected() -> None:
    with pytest.raises(ValueError, match="registered twice"):
        CommandRegistry(
            [
                Command("a", ("Open Notepad",), lambda _: "", "", lambda _: None),
                Command("b", ("open note pad",), lambda _: "", "", lambda _: None),
            ]
        )


# --- negation -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Don't open Chrome.",
        "dont open chrome",
        "Do not close Spotify.",
        "Never close VS Code",
        "Jarvis, don't lock the screen!",
        "Don't mute.",
        "do not unmute",
        "Don't take a screenshot.",
        "Never set the volume to 100.",
        "Don't turn the volume up.",
        "Please don't quit Task Manager",
        "Never mind, open Chrome.",
        "I said don't open notepad",
    ],
)
def test_negated_commands_do_nothing(desktop: FakeDesktop, text: str) -> None:
    found = desktop.registry().match(text)
    assert found is not None and found.negated
    assert desktop.registry().handle(text) == "Okay, I won't."
    assert desktop.nothing_ran()


@pytest.mark.parametrize(
    "text",
    [
        "Open notepad.",  # "not" inside "notepad" is not a negation
        "Close notepad and don't save",  # negation after the command doesn't apply
        "Do open Chrome",
        "Open Donkey Kong",  # "don" is not "don't"
    ],
)
def test_words_that_are_not_negations(desktop: FakeDesktop, text: str) -> None:
    found = desktop.registry().match(text)
    assert found is not None
    assert not found.negated


# --- open / close apps ------------------------------------------------------------

ALL_ALIASES = [(app.name, alias) for app in config.APPS for alias in app.aliases]


@pytest.mark.parametrize(("name", "alias"), ALL_ALIASES)
@pytest.mark.parametrize("verb", ["open", "launch", "start"])
def test_open_every_alias(desktop: FakeDesktop, name: str, alias: str, verb: str) -> None:
    reply = desktop.registry().handle(f"{verb.capitalize()} {alias}.")
    assert reply == f"Opening {name}."
    assert desktop.launched == [name]


@pytest.mark.parametrize(("name", "alias"), ALL_ALIASES)
@pytest.mark.parametrize("verb", ["close", "quit", "exit"])
def test_close_every_alias(desktop: FakeDesktop, name: str, alias: str, verb: str) -> None:
    desktop.windows.open.add(name)
    desktop.answers = [True]
    reply = desktop.registry().handle(f"{verb} {alias}")
    assert desktop.questions == [f"Close {name}? Say yes or no."]
    assert reply == f"Closed {name}."
    assert desktop.windows.calls == [("close", name)]


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Hey Jarvis, open VS Code.", "VS Code"),
        ("Open VSCode", "VS Code"),
        ("open V.S. Code please", "VS Code"),
        ("Open Visual Studio Code.", "VS Code"),
        ("Launch the calculator app.", "Calculator"),
        ("start calc", "Calculator"),
        ("Open my files.", "File Explorer"),
        ("Open File Explorer for me", "File Explorer"),
        ("open up spotify", "Spotify"),
        ("Open the settings app please", "Settings"),
        ("Open Task Manager now.", "Task Manager"),
        ("Open note pad.", "Notepad"),
        ("open google", "Chrome"),
        ("Open the browser.", "Chrome"),
    ],
)
def test_open_spoken_variants(desktop: FakeDesktop, text: str, name: str) -> None:
    assert desktop.registry().handle(text) == f"Opening {name}."
    assert desktop.launched == [name]


@pytest.mark.parametrize(
    "text",
    [
        "open photoshop",
        "Open Notepad and Chrome.",
        "open the",
        "open chrome browser",
        "Open my spotify",
        "open codes",
        "start the timer",
        "launch a rocket",
    ],
)
def test_open_unknown_app(desktop: FakeDesktop, text: str) -> None:
    assert desktop.registry().handle(text) == "Sorry, I don't know that app."
    assert desktop.nothing_ran()


@pytest.mark.parametrize("text", ["close photoshop", "Quit the game.", "exit full screen"])
def test_close_unknown_app_asks_nothing(desktop: FakeDesktop, text: str) -> None:
    assert desktop.registry().handle(text) == "Sorry, I don't know that app."
    assert desktop.nothing_ran()


def test_app_not_installed_is_unknown() -> None:
    installed = tuple(app for app in config.APPS if app.name != "Spotify")
    desktop = FakeDesktop(apps=installed)
    desktop.windows.open.add("Spotify")
    registry = desktop.registry()
    assert registry.handle("open spotify") == "Sorry, I don't know that app."
    assert registry.handle("close spotify") == "Sorry, I don't know that app."
    assert desktop.nothing_ran()


@pytest.mark.parametrize(("text", "reply"), [("open", "Open which app?"), ("Close.", "Close which app?")])
def test_missing_app_name(desktop: FakeDesktop, text: str, reply: str) -> None:
    assert desktop.registry().handle(text) == reply
    assert desktop.nothing_ran()


def test_open_failure(desktop: FakeDesktop) -> None:
    desktop.fail.add("launch")
    assert desktop.registry().handle("open chrome") == "Sorry, I couldn't open Chrome."


def test_close_app_that_is_not_open_asks_nothing(desktop: FakeDesktop) -> None:
    assert desktop.registry().handle("close spotify") == "Spotify isn't open."
    assert desktop.questions == []
    assert desktop.windows.calls == []


def test_close_declined(desktop: FakeDesktop) -> None:
    desktop.windows.open.add("Spotify")
    desktop.answers = [False]
    assert desktop.registry().handle("Close Spotify.") == "Okay, I'll leave Spotify open."
    assert desktop.questions == ["Close Spotify? Say yes or no."]
    assert desktop.windows.calls == []


def test_close_still_open_then_force(desktop: FakeDesktop) -> None:
    desktop.windows.open.add("VS Code")
    desktop.windows.close_results = [CloseResult.STILL_OPEN]
    desktop.answers = [True, True]
    assert desktop.registry().handle("close vs code") == "Force closed VS Code."
    assert desktop.questions == [
        "Close VS Code? Say yes or no.",
        "VS Code is still open. It may be asking to save your work. "
        "Force it to close? Say yes or no.",
    ]
    assert desktop.windows.calls == [("close", "VS Code"), ("force_close", "VS Code")]


def test_close_still_open_force_declined(desktop: FakeDesktop) -> None:
    desktop.windows.open.add("Notepad")
    desktop.windows.close_results = [CloseResult.STILL_OPEN]
    desktop.answers = [True, False]
    assert desktop.registry().handle("close notepad") == "Okay, Notepad is still open."
    assert desktop.windows.calls == [("close", "Notepad")]  # never forced


def test_explorer_is_never_force_closed(desktop: FakeDesktop) -> None:
    desktop.windows.open.add("File Explorer")
    desktop.windows.close_results = [CloseResult.STILL_OPEN]
    desktop.answers = [True]
    assert desktop.registry().handle("close explorer") == "Sorry, File Explorer didn't close."
    assert len(desktop.questions) == 1  # no force question
    assert desktop.windows.calls == [("close", "File Explorer")]


@pytest.mark.parametrize(
    ("close_result", "force_result", "reply"),
    [
        (CloseResult.DENIED, None, "Sorry, I'm not allowed to close Task Manager. It may be running as administrator."),
        (CloseResult.NOT_OPEN, None, "Task Manager isn't open."),
        (CloseResult.STILL_OPEN, CloseResult.DENIED, "Sorry, I'm not allowed to close Task Manager. It may be running as administrator."),
        (CloseResult.STILL_OPEN, CloseResult.STILL_OPEN, "Sorry, I couldn't close Task Manager."),
    ],
)
def test_close_outcomes(
    desktop: FakeDesktop,
    close_result: CloseResult,
    force_result: CloseResult | None,
    reply: str,
) -> None:
    desktop.windows.open.add("Task Manager")
    desktop.windows.close_results = [close_result]
    if force_result is not None:
        desktop.windows.force_result = force_result
    desktop.answers = [True, True]
    assert desktop.registry().handle("close task manager") == reply


def test_close_os_error(desktop: FakeDesktop) -> None:
    desktop.windows.fail = True
    assert desktop.registry().handle("close chrome") == "Sorry, I couldn't close Chrome."
    assert desktop.questions == []


# --- other handlers -----------------------------------------------------------------


def test_empty_and_unknown(desktop: FakeDesktop) -> None:
    registry = desktop.registry()
    assert registry.handle("  ...  ") == "Sorry, I didn't catch that."
    assert registry.handle("delete everything") == "Sorry, I don't know that command yet."
    assert desktop.nothing_ran()


@pytest.mark.parametrize(
    ("start", "text", "level", "reply"),
    [
        (50, "volume up", 60, "Volume up to 60 percent."),
        (95, "Turn up the volume.", 100, "Volume up to 100 percent."),
        (100, "louder", 100, "Volume is already at 100 percent."),
        (50, "volume down", 40, "Volume down to 40 percent."),
        (5, "Lower the volume.", 0, "Volume down to 0 percent."),
        (0, "quieter", 0, "Volume is already at 0 percent."),
    ],
)
def test_volume_step(start: int, text: str, level: int, reply: str) -> None:
    desktop = FakeDesktop(FakeVolume(level=start, muted=True))
    assert desktop.registry().handle(text) == reply
    assert desktop.volume.level == level
    assert desktop.volume.muted is False


def test_volume_step_size_comes_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "VOLUME_STEP_PERCENT", 25)
    desktop = FakeDesktop(FakeVolume(level=50))
    desktop.registry().handle("volume up")
    assert desktop.volume.level == 75


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("Set volume to 40.", 40),
        ("Set volume to 40%.", 40),
        ("set volume to 40 percent", 40),
        ("set volume to 40 per cent please", 40),
        ("set volume 75", 75),
        ("set the volume at 0", 0),
        ("Set volume to zero.", 0),
        ("set volume to five", 5),
        ("set volume to fifteen", 15),
        ("set the volume to forty", 40),
        ("Set the volume to forty-five.", 45),
        ("set volume to ninety nine", 99),
        ("change the volume to a hundred", 100),
        ("Change volume to one hundred.", 100),
        ("set volume to hundred", 100),
        ("set volume to 100%", 100),
        ("Turn the volume to twenty.", 20),
        ("Hey Jarvis, volume at 30 please", 30),
        ("Turn the volume up to fifty.", 50),
        ("turn down the volume to 30%", 30),
    ],
)
def test_set_volume_number_forms(text: str, level: int) -> None:
    desktop = FakeDesktop(FakeVolume(level=10, muted=True))
    assert desktop.registry().handle(text) == f"Volume set to {level} percent."
    assert desktop.volume.set_calls == [level]
    assert desktop.volume.muted is False


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("set volume to 101", 101),
        ("Set volume to 150%.", 150),
        ("set volume to 1000", 1000),
        ("set volume to -10", -10),
        ("set volume to minus five", -5),
        ("set volume to negative 1", -1),
        ("set volume to a hundred and one", 101),
        ("set volume to one hundred and fifty", 150),
        ("set volume to two hundred", 200),
        ("turn the volume up to 120", 120),
    ],
)
def test_set_volume_out_of_range_is_refused(text: str, level: int) -> None:
    desktop = FakeDesktop(FakeVolume(level=10))
    reply = desktop.registry().handle(text)
    assert reply == f"Sorry, {level} is out of range. Volume goes from 0 to 100."
    assert desktop.volume.set_calls == []


@pytest.mark.parametrize(
    "text",
    [
        "set volume",
        "set the volume to",
        "set volume to loud",
        "set volume to forty forty",
        "set volume to five forty",
        "set volume to a thousand",
        "set volume to 40.5",
        "set volume up",
        "turn the volume up to max",
    ],
)
def test_set_volume_without_a_valid_number(text: str) -> None:
    desktop = FakeDesktop(FakeVolume(level=10))
    assert desktop.registry().handle(text) == "Sorry, I need a volume level from 0 to 100."
    assert desktop.volume.set_calls == []


def test_mute_and_unmute(desktop: FakeDesktop) -> None:
    registry = desktop.registry()
    desktop.volume.muted = False
    assert registry.handle("Mute the sound.") == "Muted."
    assert desktop.volume.muted is True
    assert registry.handle("Un-mute.") == "Unmuted."
    assert desktop.volume.muted is False
    assert desktop.volume.set_calls == []  # muting never changes the level


@pytest.mark.parametrize("text", ["volume up", "set volume to 20", "mute", "unmute"])
def test_volume_failure(text: str) -> None:
    desktop = FakeDesktop(FakeVolume(fail=True))
    assert desktop.registry().handle(text) == "Sorry, I couldn't change the volume."


def test_screenshot(desktop: FakeDesktop) -> None:
    reply = desktop.registry().handle("Take a screenshot.")
    assert reply == "Screenshot saved as screenshot_2026-10-25_12-00-00.png."
    assert desktop.screenshots == 1


def test_screenshot_failure(desktop: FakeDesktop) -> None:
    desktop.fail.add("screenshot")
    assert desktop.registry().handle("screenshot") == "Sorry, I couldn't take a screenshot."


def test_lock_screen(desktop: FakeDesktop) -> None:
    assert desktop.registry().handle("Lock the screen.") == "Locking the screen."
    assert desktop.locks == 1


def test_lock_screen_failure(desktop: FakeDesktop) -> None:
    desktop.fail.add("lock")
    assert desktop.registry().handle("lock my pc") == "Sorry, I couldn't lock the screen."


def test_commands_module_has_no_windows_imports() -> None:
    # Windows-only libraries stay in their own modules so this one can be faked.
    from jarvis import commands

    source = Path(commands.__file__).read_text(encoding="utf-8")
    for name in ("pycaw", "comtypes", "mss", "ctypes", "psutil", "winreg", "launcher",
                 "volume", "screenshot", "app_windows", "ollama_client", "urllib"):
        assert f"import {name}" not in source and f"jarvis.{name}" not in source


# --- LLM brain (fake client) ----------------------------------------------------


class FakeLlm:
    """Returns scripted raw replies (or raises scripted errors) and records prompts."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.mark.parametrize(
    "text", ["open chrome", "Volume up.", "Don't lock the screen", "open photoshop", " ... "]
)
def test_brain_is_not_asked_when_a_phrase_matches_or_text_is_empty(
    desktop: FakeDesktop, text: str
) -> None:
    llm = FakeLlm()
    desktop.registry(llm).handle(text)
    assert llm.calls == []


def test_brain_answers_unmatched_speech(desktop: FakeDesktop) -> None:
    llm = FakeLlm('{"answer": "Paris is the capital of France."}')
    reply = desktop.registry(llm).handle("What's the capital of France?")
    assert reply == "Paris is the capital of France."
    assert llm.calls[0][1] == "What's the capital of France?"  # the transcribed text, as heard
    assert desktop.nothing_ran()


def test_long_answers_are_cut_to_three_sentences(desktop: FakeDesktop) -> None:
    llm = FakeLlm('{"answer": "One. **Two!** Three? Four. Five."}')
    assert desktop.registry(llm).handle("tell me a story") == "One. Two! Three?"


def test_system_prompt_lists_every_command_and_only_installed_apps() -> None:
    installed = tuple(app for app in config.APPS if app.name != "Spotify")
    llm = FakeLlm('{"answer": "Hi."}')
    registry = FakeDesktop(apps=installed).registry(llm)
    registry.handle("hello there")
    system = llm.calls[0][0]
    for command in registry.commands:
        assert f"- {command.name}: " in system
    for app in installed:
        assert f'"{app.name}"' in system
    assert "Spotify" not in system
    assert "JSON" in system and "one to three short sentences" in system


@pytest.mark.parametrize(
    ("raw", "check"),
    [
        ('{"command": "set_volume", "args": {"level": 30}}', lambda d: d.volume.set_calls == [30]),
        ('{"command": "open_app", "args": {"app": "Spotify"}}', lambda d: d.launched == ["Spotify"]),
        (
            '{"command": "open_app", "args": {"app": "visual studio code"}}',
            lambda d: d.launched == ["VS Code"],
        ),
        ('{"command": "volume_up", "args": {}}', lambda d: d.volume.set_calls == [60]),
        ('{"command": "volume_down"}', lambda d: d.volume.set_calls == [40]),
        ('{"command": "mute", "args": {}}', lambda d: d.volume.muted is True),
        ('{"command": "unmute", "args": {}}', lambda d: d.volume.muted is False),
        ('{"command": "screenshot", "args": {}}', lambda d: d.screenshots == 1),
    ],
)
def test_brain_runs_valid_whitelisted_commands(
    raw: str, check: Callable[[FakeDesktop], bool]
) -> None:
    desktop = FakeDesktop(FakeVolume(level=50, muted="unmute" in raw))
    desktop.registry(FakeLlm(raw)).handle("I can barely hear anything, could you sort it out")
    assert check(desktop)
    assert desktop.questions == []  # only lock_screen asks first


@pytest.mark.parametrize(
    ("answer", "reply", "locks"),
    [
        (True, "Locking the screen.", 1),
        (False, "Okay, I won't lock the screen.", 0),
    ],
)
def test_model_chosen_lock_asks_first(answer: bool, reply: str, locks: int) -> None:
    desktop = FakeDesktop()
    desktop.answers = [answer]
    llm = FakeLlm('{"command": "lock_screen", "args": {}}')
    assert desktop.registry(llm).handle("I'm stepping away from my desk for a bit.") == reply
    assert desktop.questions == ["Lock the screen? Say yes or no."]
    assert desktop.locks == locks


def test_model_chosen_lock_with_args_is_rejected_before_asking(desktop: FakeDesktop) -> None:
    llm = FakeLlm('{"command": "lock_screen", "args": {"now": true}}')
    assert desktop.registry(llm).handle("I'm off for lunch") == "Sorry, I can't do that."
    assert desktop.questions == []
    assert desktop.locks == 0


@pytest.mark.parametrize("text", ["Lock the screen.", "Hey Jarvis, lock my PC."])
def test_spoken_lock_phrase_does_not_ask(desktop: FakeDesktop, text: str) -> None:
    llm = FakeLlm()
    assert desktop.registry(llm).handle(text) == "Locking the screen."
    assert desktop.questions == []
    assert llm.calls == []
    assert desktop.locks == 1


@pytest.mark.parametrize(
    ("text", "reply"),
    [
        ("Crank it up, I love this song!", "Volume up to 60 percent."),
        ("Turn it up.", "Volume up to 60 percent."),
        ("Pump it up!", "Volume up to 60 percent."),
        ("Turn it up to eighty.", "Volume set to 80 percent."),
        ("Silence.", "Muted."),
        ("Silence please, I'm on a call.", "Muted."),
        ("Quiet!", "Muted."),
        ("Jarvis, quiet please.", "Muted."),
        ("quieter", "Volume down to 40 percent."),  # still volume down, not mute
        ("Don't crank it up.", "Okay, I won't."),
        ("Never silence the alarm.", "Okay, I won't."),
    ],
)
def test_new_volume_phrases(text: str, reply: str) -> None:
    desktop = FakeDesktop(FakeVolume(level=50, muted=False))
    llm = FakeLlm()
    assert desktop.registry(llm).handle(text) == reply
    assert llm.calls == []  # fixed phrases, no model involved


def test_brain_close_still_asks_for_confirmation(desktop: FakeDesktop) -> None:
    desktop.windows.open.add("Spotify")
    desktop.answers = [False]
    llm = FakeLlm('{"command": "close_app", "args": {"app": "Spotify"}}')
    reply = desktop.registry(llm).handle("I'm done listening to music for today")
    assert desktop.questions == ["Close Spotify? Say yes or no."]
    assert reply == "Okay, I'll leave Spotify open."
    assert desktop.windows.calls == []


@pytest.mark.parametrize(
    "raw",
    [
        # Invented or wrongly-typed commands.
        '{"command": "delete_files", "args": {"path": "C:/"}}',
        '{"command": "run_shell", "args": {"cmd": "shutdown /s"}}',
        '{"command": "Open_App", "args": {"app": "Notepad"}}',
        '{"command": 1}',
        '{"command": null}',
        '{"command": ["mute"]}',
        # Invalid args for real commands.
        '{"command": "set_volume", "args": {"level": 150}}',
        '{"command": "set_volume", "args": {"level": -1}}',
        '{"command": "set_volume", "args": {"level": "50"}}',
        '{"command": "set_volume", "args": {"level": 50.0}}',
        '{"command": "set_volume", "args": {"level": true}}',
        '{"command": "set_volume", "args": {}}',
        '{"command": "set_volume", "args": {"level": 50, "force": true}}',
        '{"command": "set_volume", "args": [50]}',
        '{"command": "set_volume", "args": "level 50"}',
        '{"command": "open_app", "args": {"app": "Photoshop"}}',
        '{"command": "open_app", "args": {"app": "notepad.exe"}}',
        '{"command": "open_app", "args": {"app": "cmd.exe /c del *"}}',
        '{"command": "open_app", "args": {"app": "Notepad and Chrome"}}',
        '{"command": "open_app", "args": {"app": ["Notepad"]}}',
        '{"command": "open_app", "args": {"name": "Notepad"}}',
        '{"command": "open_app", "args": {}}',
        '{"command": "close_app", "args": {"app": "explorer.exe"}}',
        '{"command": "lock_screen", "args": {"now": true}}',
        '{"command": "mute", "args": {"muted": true}}',
    ],
)
def test_brain_commands_that_fail_validation_never_run(desktop: FakeDesktop, raw: str) -> None:
    desktop.windows.open.update(app.name for app in config.APPS)
    reply = desktop.registry(FakeLlm(raw)).handle("do something for me")
    assert reply in ("Sorry, I can't do that.", "Sorry, I didn't understand that.")
    assert desktop.nothing_ran()


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "Sure! I'll open Notepad for you.",
        "open_app Notepad",
        '{"answer": "ok"',
        "[]",
        '"just a string"',
        "null",
        "42",
        '{"answer": ""}',
        '{"answer": "   "}',
        '{"answer": 42}',
        '{"answer": null}',
        '{"answer": "Done.", "command": "mute"}',
        '{"answer": "Hi.", "extra": 1}',
        '{"command": "mute", "args": {}, "answer": "Muting."}',
        '{"command": "mute", "args": {}, "reason": "user asked"}',
        '{"answer": "a", "answer": "b"}',
        '{"command": "mute", "command": "lock_screen"}',
        "{}",
    ],
)
def test_malformed_brain_replies_are_rejected(desktop: FakeDesktop, raw: str) -> None:
    reply = desktop.registry(FakeLlm(raw)).handle("do something for me")
    assert reply == "Sorry, I didn't understand that."
    assert desktop.nothing_ran()


@pytest.mark.parametrize(
    "text",
    [
        "I don't want the screen locked.",
        "Never make it louder at night.",
        "Do not take any pictures of my screen.",
    ],
)
def test_brain_commands_are_refused_when_the_user_negates(desktop: FakeDesktop, text: str) -> None:
    llm = FakeLlm('{"command": "lock_screen", "args": {}}')
    assert desktop.registry(llm).handle(text) == "Okay, I won't."
    assert desktop.nothing_ran()


def test_negated_speech_can_still_get_an_answer(desktop: FakeDesktop) -> None:
    llm = FakeLlm('{"answer": "Nobody knows for sure."}')
    assert desktop.registry(llm).handle("I don't know why cats purr") == "Nobody knows for sure."


def test_brain_offline_and_normal_commands_keep_working(desktop: FakeDesktop) -> None:
    llm = FakeLlm(BrainOffline("connection refused"), BrainOffline("connection refused"))
    registry = desktop.registry(llm)
    assert registry.handle("what's the weather like") == "My brain is offline."
    assert registry.handle("open notepad") == "Opening Notepad."
    assert registry.handle("tell me a joke") == "My brain is offline."
    assert desktop.launched == ["Notepad"]


def test_brain_timeout(desktop: FakeDesktop) -> None:
    llm = FakeLlm(BrainTimeout("20 s"))
    reply = desktop.registry(llm).handle("explain quantum physics")
    assert reply == "Sorry, my brain took too long to answer."


@pytest.mark.parametrize("missing", [(), ("Spotify",), ("Chrome",), ("Spotify", "Chrome")])
def test_prompt_examples_are_valid_commands_for_installed_apps(missing: tuple[str, ...]) -> None:
    from jarvis.brain import CommandRequest, parse_reply

    installed = tuple(app for app in config.APPS if app.name not in missing)
    desktop = FakeDesktop(apps=installed)
    registry = desktop.registry(FakeLlm())
    assert registry.brain is not None
    prompt = registry.brain.system_prompt
    for name in missing:
        assert name not in prompt
    by_name = {command.name: command for command in registry.commands}
    examples = prompt.split("Examples:\n", 1)[1]
    replies = [line for line in examples.splitlines() if line.startswith('{"command"')]
    assert len(replies) == 11 - len(missing)
    for raw in replies:
        request = parse_reply(raw, by_name, 3, 300)
        assert isinstance(request, CommandRequest), raw
        assert by_name[request.command].bind_args(request.args) is not None, raw  # would run


def test_without_a_brain_unmatched_speech_gets_the_old_reply(desktop: FakeDesktop) -> None:
    assert desktop.registry(None).handle("tell me a joke") == "Sorry, I don't know that command yet."


# --- web search and sites ---------------------------------------------------------

GOOGLE_URL = "https://www.google.com/search?q="
YOUTUBE_URL = "https://www.youtube.com/results?search_query="


@pytest.mark.parametrize(
    ("text", "engine", "query"),
    [
        # Google
        ("Search for cats.", "Google", "cats"),
        ("Hey Jarvis, search for the best pizza near me!", "Google", "the best pizza near me"),
        ("Google the weather in Paris.", "Google", "the weather in Paris"),
        ("Search Google for C++ tutorials?", "Google", "C++ tutorials"),
        ("Look up AT&T stock price.", "Google", "AT&T stock price"),
        ("look up what is 2+2?", "Google", "what is 2+2"),
        ("Search for C# vs Java, please.", "Google", "C# vs Java"),
        ("Search for YouTube downloader.", "Google", "YouTube downloader"),  # earliest phrase wins
        ("Search for cats on YouTube.", "YouTube", "cats"),  # Google query ending "on YouTube"
        ("Google lo-fi beats on YT!", "YouTube", "lo-fi beats"),
        ("Look up how to juggle on Y.T., please.", "YouTube", "how to juggle"),
        ("search google for cats on you tube", "YouTube", "cats"),
        ("Search for cats on YouTube music.", "Google", "cats on YouTube music"),  # doesn't end with it
        ("Search for YouTube premium price.", "Google", "YouTube premium price"),
        ("Search for नमस्ते दुनिया", "Google", "नमस्ते दुनिया"),
        ("Google 東京 天気", "Google", "東京 天気"),
        ("Search for café crème.", "Google", "café crème"),
        # YouTube
        ("Search YouTube for lo-fi beats...", "YouTube", "lo-fi beats"),
        ("YouTube Never Gonna Give You Up.", "YouTube", "Never Gonna Give You Up"),
        ("Play Bohemian Rhapsody on YouTube.", "YouTube", "Bohemian Rhapsody"),
        ("Hey Jarvis, play songs on repeat on YouTube!", "YouTube", "songs on repeat"),
        ("Find how to tie a tie on YouTube, please.", "YouTube", "how to tie a tie"),
        ("find cooking videos on youtube", "YouTube", "cooking videos"),
        ("Play “Despacito” on YouTube.", "YouTube", "Despacito"),
        ("play भजन on youtube", "YouTube", "भजन"),
    ],
)
def test_search_phrases_and_query_extraction(
    desktop: FakeDesktop, text: str, engine: str, query: str
) -> None:
    llm = FakeLlm()
    reply = desktop.registry(llm).handle(text)
    assert reply == f"Searching {engine} for {query}."
    base = GOOGLE_URL if engine == "Google" else YOUTUBE_URL
    (url,) = desktop.urls
    assert url.startswith(base)
    from urllib.parse import parse_qs, urlsplit

    assert list(parse_qs(urlsplit(url).query).values()) == [[query]]
    assert llm.calls == []  # fixed phrases, no model involved


@pytest.mark.parametrize(
    "text",
    [
        "search for",
        "Search for.",
        "Search for, please.",
        "Google!",
        "look up",
        "YouTube.",
        "search youtube for...",
        "Play on YouTube.",
        "find on youtube",
        "Search for on YouTube.",
        "google on yt",
    ],
)
def test_empty_search_asks_what_to_search_for(desktop: FakeDesktop, text: str) -> None:
    assert desktop.registry().handle(text) == "What should I search for?"
    assert desktop.nothing_ran()


def test_overlong_spoken_search_is_refused(desktop: FakeDesktop) -> None:
    text = "search for " + "very " * 50 + "long"
    assert desktop.registry().handle(text) == "Sorry, that search is too long."
    assert desktop.nothing_ran()


@pytest.mark.parametrize(
    "text",
    [
        "Don't search for cats.",
        "Do not google my name.",
        "Never look up my ex.",
        "Don't play Baby Shark on YouTube!",
        "Please don't search YouTube for that.",
        "dont youtube it",
    ],
)
def test_negated_searches_do_nothing(desktop: FakeDesktop, text: str) -> None:
    assert desktop.registry().handle(text) == "Okay, I won't."
    assert desktop.nothing_ran()


@pytest.mark.parametrize(
    "text",
    [
        "I was researching for hours",
        "I googled it already",
        "my favourite youtuber",
        "lookup tables are fast",
        "find my keys",
        "searching for meaning",
    ],
)
def test_search_near_misses(desktop: FakeDesktop, text: str) -> None:
    found = desktop.registry().match(text)
    assert found is None or found.command.name not in ("search_google", "search_youtube")


@pytest.mark.parametrize(
    ("text", "url", "reply"),
    [
        ("Open YouTube.", "https://www.youtube.com/", "Opening YouTube."),
        ("open gmail", "https://mail.google.com/", "Opening Gmail."),
        ("Open GitHub.", "https://github.com/", "Opening GitHub."),
        ("Open git hub please", "https://github.com/", "Opening GitHub."),
        ("Hey Jarvis, open the GitHub website.", "https://github.com/", "Opening GitHub."),
    ],
)
def test_open_whitelisted_site(desktop: FakeDesktop, text: str, url: str, reply: str) -> None:
    assert desktop.registry().handle(text) == reply
    assert desktop.urls == [url]
    assert desktop.launched == []


def test_apps_take_priority_and_unknown_sites_are_unknown(desktop: FakeDesktop) -> None:
    registry = desktop.registry()
    assert registry.handle("open google") == "Opening Chrome."  # app alias, not a search
    assert registry.handle("open reddit") == "Sorry, I don't know that app."
    assert registry.handle("open youtube music") == "Sorry, I don't know that app."
    assert registry.handle("close youtube") == "Sorry, I don't know that app."
    assert desktop.urls == []


def test_browser_failure(desktop: FakeDesktop) -> None:
    desktop.fail.add("browser")
    registry = desktop.registry()
    assert registry.handle("search for cats") == "Sorry, I couldn't search Google."
    assert registry.handle("open github") == "Sorry, I couldn't open GitHub."


# --- LLM-chosen searches -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "url", "reply"),
    [
        (
            '{"command": "search_youtube", "args": {"query": "funny cat videos"}}',
            YOUTUBE_URL + "funny+cat+videos",
            "Searching YouTube for funny cat videos.",
        ),
        (
            '{"command": "search_google", "args": {"query": "weather in Mumbai"}}',
            GOOGLE_URL + "weather+in+Mumbai",
            "Searching Google for weather in Mumbai.",
        ),
        (
            '{"command": "search_google", "args": {"query": "AT&T #1 ?"}}',
            GOOGLE_URL + "AT%26T+%231+%3F",
            "Searching Google for AT&T #1 ?.",
        ),
        (
            '{"command": "search_google", "args": {"query": "https://evil.example/x"}}',
            GOOGLE_URL + "https%3A%2F%2Fevil.example%2Fx",
            "Searching Google for https://evil.example/x.",
        ),
        (
            '{"command": "search_google", "args": {"query": "  spaced\\n  out  "}}',
            GOOGLE_URL + "spaced+out",
            "Searching Google for spaced out.",
        ),
        (
            '{"command": "search_google", "args": {"query": "नमस्ते"}}',
            GOOGLE_URL + "%E0%A4%A8%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%A4%E0%A5%87",
            "Searching Google for नमस्ते.",
        ),
        (
            '{"command": "open_app", "args": {"app": "YouTube"}}',
            "https://www.youtube.com/",
            "Opening YouTube.",
        ),
        (
            '{"command": "search_google", "args": {"query": "cat videos on YouTube"}}',
            YOUTUBE_URL + "cat+videos",
            "Searching YouTube for cat videos.",
        ),
    ],
)
def test_llm_chosen_search(desktop: FakeDesktop, raw: str, url: str, reply: str) -> None:
    assert desktop.registry(FakeLlm(raw)).handle("I'm bored, give me something to do") == reply
    assert desktop.urls == [url]
    assert desktop.questions == []  # searches run directly


@pytest.mark.parametrize(
    "raw",
    [
        '{"command": "search_google", "args": {"query": ""}}',
        '{"command": "search_google", "args": {"query": "   "}}',
        '{"command": "search_google", "args": {"query": "' + "x" * 201 + '"}}',
        '{"command": "search_google", "args": {"query": 42}}',
        '{"command": "search_google", "args": {"query": ["cats"]}}',
        '{"command": "search_google", "args": {"query": null}}',
        '{"command": "search_google", "args": {}}',
        '{"command": "search_google", "args": {"q": "cats"}}',
        '{"command": "search_google", "args": {"query": "cats", "engine": "bing"}}',
        '{"command": "search_google", "args": {"query": "cats\\u0000"}}',
        '{"command": "search_youtube", "args": {"query": "cats\\u001b[2J"}}',
        '{"command": "search_bing", "args": {"query": "cats"}}',
        '{"command": "open_url", "args": {"url": "https://evil.example"}}',
        '{"command": "open_app", "args": {"app": "https://evil.example"}}',
        '{"command": "open_app", "args": {"app": "Reddit"}}',
    ],
)
def test_invalid_llm_searches_never_open_anything(desktop: FakeDesktop, raw: str) -> None:
    reply = desktop.registry(FakeLlm(raw)).handle("find me something")
    # Invented ids are rejected while parsing; bad args by the command itself.
    assert reply in ("Sorry, I can't do that.", "Sorry, I didn't understand that.")
    assert desktop.nothing_ran()


def test_llm_search_is_refused_when_negated(desktop: FakeDesktop) -> None:
    llm = FakeLlm('{"command": "search_google", "args": {"query": "my ex"}}')
    assert desktop.registry(llm).handle("I really don't want to know about my ex") == "Okay, I won't."
    assert desktop.nothing_ran()


def test_llm_prompt_describes_search_and_sites(desktop: FakeDesktop) -> None:
    registry = desktop.registry(FakeLlm())
    assert registry.brain is not None
    prompt = registry.brain.system_prompt
    assert "- search_google: " in prompt and "- search_youtube: " in prompt
    assert f"at most {config.SEARCH_MAX_QUERY_CHARS} characters" in prompt
    for site in config.SITES:
        assert f'"{site.name}"' in prompt
    assert "news, scores, prices, weather, results), use search_google" in prompt
    assert "Never answer that you have no real-time data." in prompt


# --- media keys ----------------------------------------------------------------------

MEDIA_COMMAND_KEYS = {
    "media_play_pause": "play_pause",
    "media_next": "next",
    "media_previous": "previous",
    "media_stop": "stop",
}
MEDIA_PHRASES = [(name, phrase) for name in MEDIA_COMMAND_KEYS for phrase in PHRASES[name]]


@pytest.mark.parametrize(("name", "phrase"), MEDIA_PHRASES)
@pytest.mark.parametrize(
    "template",
    ["{}", "{}.", "Hey Jarvis, {}!", "{} please", "Jarvis, {} now, thank you."],
    ids=["bare", "period", "wake", "please", "polite"],
)
def test_every_media_phrase_presses_its_key_silently(name: str, phrase: str, template: str) -> None:
    desktop = FakeDesktop()
    llm = FakeLlm()
    reply = desktop.registry(llm).handle(template.format(phrase[0].upper() + phrase[1:]))
    assert reply == ""  # silent; app.py doesn't speak empty replies
    assert desktop.media_keys == [MEDIA_COMMAND_KEYS[name]]
    assert llm.calls == []


@pytest.mark.parametrize(
    ("text", "query"),
    [
        ("Play Bohemian Rhapsody on YouTube.", "Bohemian Rhapsody"),
        ("play the music on youtube", "the music"),
        ("Play pause on YouTube", "pause"),
        ("Hey Jarvis, play next on YouTube please", "next"),
        ("play skip this song on youtube", "skip this song"),
    ],
)
def test_play_x_on_youtube_is_still_a_youtube_search(desktop: FakeDesktop, text: str, query: str) -> None:
    assert desktop.registry().handle(text) == f"Searching YouTube for {query}."
    assert desktop.media_keys == []
    assert len(desktop.urls) == 1


@pytest.mark.parametrize(
    "text",
    [
        "playground",
        "Let's go to the playground.",
        "I'm skipping lunch today.",
        "skip lunch",
        "What's next on my calendar?",
        "next week is busy",
        "the previous owner left",
        "pause for a second",
        "resume my download",
        "stop playing games",
        "continue playing chess",
        "my playlist",
        "the last track meet",
        "I paused it",
        "replay that",
        "stop",
        "music",
    ],
)
def test_media_near_misses_press_nothing(desktop: FakeDesktop, text: str) -> None:
    found = desktop.registry().match(text)
    assert found is None or not found.command.name.startswith("media_")




@pytest.mark.parametrize(
    "text",
    [
        "Don't pause.",
        "Do not skip this song!",
        "Never stop the music.",
        "Don't stop playing.",
        "Please don't go back a song.",
        "dont play the music",
    ],
)
def test_negated_media_commands_do_nothing(desktop: FakeDesktop, text: str) -> None:
    assert desktop.registry().handle(text) == "Okay, I won't."
    assert desktop.nothing_ran()


def test_media_key_failure_is_spoken(desktop: FakeDesktop) -> None:
    desktop.fail.add("media")
    assert desktop.registry().handle("next song") == "Sorry, I couldn't control the music."


@pytest.mark.parametrize(("command", "key"), sorted(MEDIA_COMMAND_KEYS.items()))
def test_llm_chosen_media_commands(desktop: FakeDesktop, command: str, key: str) -> None:
    llm = FakeLlm(f'{{"command": "{command}", "args": {{}}}}')
    reply = desktop.registry(llm).handle("do the music thing")
    assert reply == ""
    assert desktop.media_keys == [key]
    assert desktop.questions == []


@pytest.mark.parametrize(
    "raw",
    [
        '{"command": "media_next", "args": {"times": 3}}',
        '{"command": "media_play_pause", "args": {"key": "VK_VOLUME_MUTE"}}',
        '{"command": "media_key", "args": {"name": "next"}}',
        '{"command": "press_key", "args": {"vk": 91}}',
    ],
)
def test_invalid_llm_media_commands_press_nothing(desktop: FakeDesktop, raw: str) -> None:
    reply = desktop.registry(FakeLlm(raw)).handle("do the music thing")
    assert reply in ("Sorry, I can't do that.", "Sorry, I didn't understand that.")
    assert desktop.nothing_ran()


def test_llm_media_command_refused_when_negated(desktop: FakeDesktop) -> None:
    llm = FakeLlm('{"command": "media_stop", "args": {}}')
    assert desktop.registry(llm).handle("I never want this song to end") == "Okay, I won't."
    assert desktop.nothing_ran()


# --- "play X" / "put on X" ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "query"),
    [
        ("Play Despacito.", "Despacito"),
        ("Hey Jarvis, play Shape of You, please.", "Shape of You"),
        ("play some jazz", "some jazz"),
        ("Play music.", "music"),
        ("Put on some Arijit Singh.", "some Arijit Singh"),
        ("put on lo-fi beats", "lo-fi beats"),
        ("Play the music video for Thriller.", "the music video for Thriller"),  # not "play the music"
        ("Play AC/DC!", "AC/DC"),
        ("play भजन", "भजन"),
        ("Play Bohemian Rhapsody on YouTube.", "Bohemian Rhapsody"),  # unchanged
        ("Play Despacito on YT", "Despacito"),
    ],
)
def test_play_or_put_on_x_searches_youtube(desktop: FakeDesktop, text: str, query: str) -> None:
    llm = FakeLlm()
    assert desktop.registry(llm).handle(text) == f"Searching YouTube for {query}."
    assert desktop.media_keys == []
    assert len(desktop.urls) == 1 and desktop.urls[0].startswith(YOUTUBE_URL)
    assert llm.calls == []


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("Play.", "play_pause"),
        ("play it", "play_pause"),
        ("Play it again, please.", "play_pause"),
        ("Hey Jarvis, play!", "play_pause"),
        ("Play the music.", "play_pause"),
        # "play" / "put on" + a media phrase does that, rather than searching for it.
        ("Play next.", "next"),
        ("play next song please", "next"),
        ("Play previous.", "previous"),
        ("play pause", "play_pause"),
        ("Put on pause.", "play_pause"),
        ("put on the next track", None),  # "the next track" isn't a media phrase: searches
    ],
)
def test_play_alone_or_with_a_media_phrase(desktop: FakeDesktop, text: str, key: str | None) -> None:
    desktop.registry().handle(text)
    if key is None:
        assert desktop.media_keys == [] and len(desktop.urls) == 1
    else:
        assert desktop.media_keys == [key]
        assert desktop.urls == []


@pytest.mark.parametrize("text", ["put on", "Put on.", "put on please"])
def test_put_on_alone_does_nothing(desktop: FakeDesktop, text: str) -> None:
    llm = FakeLlm('{"answer": "Put on what?"}')
    assert desktop.registry(llm).handle(text) == "Put on what?"  # nothing matched; the brain answers
    assert desktop.nothing_ran()


@pytest.mark.parametrize(
    "text",
    ["Don't play Despacito.", "Never put on Nickelback!", "Please don't play Baby Shark."],
)
def test_negated_play_x_does_nothing(desktop: FakeDesktop, text: str) -> None:
    assert desktop.registry().handle(text) == "Okay, I won't."
    assert desktop.nothing_ran()


@pytest.mark.parametrize("text", ["playground", "I love playing chess", "the display is broken", "putting on weight"])
def test_play_and_put_on_near_misses(desktop: FakeDesktop, text: str) -> None:
    found = desktop.registry().match(text)
    assert found is None or found.command.name != "search_youtube"


# --- next-track request phrases ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Change the song.",
        "change it",
        "Jarvis, change it, please.",
        "I don't like this song.",
        "Hey Jarvis, I don't like this song!",
        "I dont like this song",
        "I really don't like this song.",
        "Ugh, I can't stand this song.",
        "I can't stand this song, thanks.",
        "I don't like this song, skip it.",
    ],
)
def test_next_track_requests_are_not_mistaken_for_negation(desktop: FakeDesktop, text: str) -> None:
    llm = FakeLlm()
    found = desktop.registry(llm).match(text)
    assert found is not None and found.command.name == "media_next"
    assert not found.negated  # the "don't" / "can't" is part of the request
    assert desktop.registry(llm).handle(text) == ""
    assert desktop.media_keys == ["next"]
    assert llm.calls == []


@pytest.mark.parametrize(
    "text",
    [
        "Don't change the song.",
        "Never change it.",
        "Do not change the song!",
        "Never mind, I don't like this song.",  # "never mind" before cancels, like elsewhere
    ],
)
def test_negated_next_track_requests_do_nothing(desktop: FakeDesktop, text: str) -> None:
    assert desktop.registry().handle(text) == "Okay, I won't."
    assert desktop.nothing_ran()


def test_negated_request_with_extra_words_runs_nothing(desktop: FakeDesktop) -> None:
    llm = FakeLlm('{"command": "media_next", "args": {}}')  # even if the model picks next
    reply = desktop.registry(llm).handle("Do not change the song, I love it.")
    assert reply == "Okay, I won't."
    assert desktop.nothing_ran()


@pytest.mark.parametrize(
    "text",
    [
        "change it to blue",
        "I don't like this sofa",
        "I can't stand this weather",
        "change the song title in my notes",
        "change it later",
    ],
)
def test_next_track_near_misses(desktop: FakeDesktop, text: str) -> None:
    found = desktop.registry().match(text)
    assert found is None or found.command.name != "media_next"


def test_trigger_with_and_without_trailing_slot_can_coexist() -> None:
    def ok(_: str) -> str:
        return "ok"

    registry = CommandRegistry(
        [
            Command("bare", ("play",), ok, "", lambda _: None, standalone=True),
            Command("slot", ("play *",), ok, "", lambda _: None),
        ]
    )
    assert (found := registry.match("play")) is not None and found.command.name == "bare"
    assert (found := registry.match("play cats")) is not None and found.command.name == "slot"
    with pytest.raises(ValueError, match="registered twice"):
        CommandRegistry([Command("a", ("play *",), ok, "", lambda _: None)] * 2)


# --- "play X": top YouTube video via a (fake) lookup ------------------------------------

WATCH_URL = "https://www.youtube.com/watch?v="


class FakeLookup:
    def __init__(self, result: str | None | Exception = "kJQP7kiw5Fk") -> None:
        self.result = result
        self.queries: list[str] = []

    def __call__(self, query: str) -> str | None:
        self.queries.append(query)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def lookup(desktop: FakeDesktop) -> FakeLookup:
    fake = FakeLookup()
    desktop.find_top_video = fake
    return fake


@pytest.mark.parametrize(
    ("text", "query"),
    [
        ("Play Despacito.", "Despacito"),
        ("Put on some Arijit Singh.", "some Arijit Singh"),
        ("Play Bohemian Rhapsody on YouTube.", "Bohemian Rhapsody"),
        ("Hey Jarvis, play Shape of You on YT, please.", "Shape of You"),
        ("Play AC/DC!", "AC/DC"),
        ("play भजन", "भजन"),
    ],
)
def test_play_opens_the_top_video(
    desktop: FakeDesktop, lookup: FakeLookup, text: str, query: str
) -> None:
    assert desktop.registry().handle(text) == f"Playing {query} on YouTube."
    assert lookup.queries == [query]  # cleaned, "on YouTube" removed
    assert desktop.urls == [WATCH_URL + "kJQP7kiw5Fk"]


@pytest.mark.parametrize(
    "result",
    [
        None,  # empty results, a timeout or a failed lookup
        "",
        "kJQP7kiw5F",  # 10 characters
        "kJQP7kiw5Fkk",  # 12
        "kJQP7kiw5F&",
        "../../../ab",
        "x&list=evil",
        "kJQP7kiw5Fk\n",
        "kJQP7kiw5Fé",
        RuntimeError("yt-dlp broke"),
        OSError("network down"),
    ],
)
def test_play_falls_back_to_results_silently(
    desktop: FakeDesktop, result: str | None | Exception
) -> None:
    desktop.find_top_video = FakeLookup(result)
    reply = desktop.registry().handle("Play Despacito.")
    assert reply == "Searching YouTube for Despacito."  # the normal reply, nothing extra
    assert desktop.urls == [YOUTUBE_URL + "Despacito"]


def test_play_falls_back_after_a_real_timeout(desktop: FakeDesktop) -> None:
    from jarvis.youtube_lookup import YouTubeLookup

    def slow_extract(url: str, timeout: float) -> object:
        time.sleep(2.0)
        return {"entries": [{"id": "kJQP7kiw5Fk"}]}

    desktop.find_top_video = YouTubeLookup(0.2, slow_extract).top_video_id
    started = time.monotonic()
    assert desktop.registry().handle("play despacito") == "Searching YouTube for despacito."
    assert time.monotonic() - started < 1.0
    assert desktop.urls == [YOUTUBE_URL + "despacito"]


def test_play_without_a_lookup_opens_results(desktop: FakeDesktop) -> None:
    assert desktop.registry().handle("Play Despacito.") == "Searching YouTube for Despacito."
    assert desktop.urls == [YOUTUBE_URL + "Despacito"]


@pytest.mark.parametrize(
    "text",
    [
        "Search YouTube for Despacito.",
        "YouTube Despacito",
        "Find Despacito on YouTube.",
        "Search for Despacito on YouTube.",  # Google phrase redirected to YouTube results
    ],
)
def test_youtube_search_phrases_still_open_results(
    desktop: FakeDesktop, lookup: FakeLookup, text: str
) -> None:
    assert desktop.registry().handle(text) == "Searching YouTube for Despacito."
    assert desktop.urls == [YOUTUBE_URL + "Despacito"]
    assert lookup.queries == []  # no lookup for plain searches


@pytest.mark.parametrize(
    "text", ["play", "Play it.", "play next song", "Put on pause.", "Don't play Despacito.", "Play on YouTube."]
)
def test_no_lookup_for_media_phrases_negation_or_empty(
    desktop: FakeDesktop, lookup: FakeLookup, text: str
) -> None:
    desktop.registry().handle(text)
    assert lookup.queries == []
    assert not any(url.startswith(WATCH_URL) for url in desktop.urls)


def test_brain_follows_the_same_split(desktop: FakeDesktop, lookup: FakeLookup) -> None:
    llm = FakeLlm(
        '{"command": "play_youtube", "args": {"query": "Shape of You"}}',
        '{"command": "search_youtube", "args": {"query": "cat videos"}}',
    )
    registry = desktop.registry(llm)
    assert registry.handle("I'd love to hear that Ed Sheeran song") == "Playing Shape of You on YouTube."
    assert registry.handle("show me some cat videos") == "Searching YouTube for cat videos."
    assert desktop.urls == [WATCH_URL + "kJQP7kiw5Fk", YOUTUBE_URL + "cat+videos"]
    assert lookup.queries == ["Shape of You"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"command": "play_youtube", "args": {"query": ""}}',
        '{"command": "play_youtube", "args": {"query": "' + "x" * 201 + '"}}',
        '{"command": "play_youtube", "args": {"video_id": "kJQP7kiw5Fk"}}',  # ids only via lookup
        '{"command": "play_youtube", "args": {"query": "x", "id": "kJQP7kiw5Fk"}}',
        '{"command": "play_youtube", "args": {"url": "https://evil.example"}}',
    ],
)
def test_invalid_brain_play_requests(desktop: FakeDesktop, lookup: FakeLookup, raw: str) -> None:
    reply = desktop.registry(FakeLlm(raw)).handle("I fancy some tunes")
    assert reply in ("Sorry, I can't do that.", "Sorry, I didn't understand that.")
    assert lookup.queries == []
    assert desktop.nothing_ran()
