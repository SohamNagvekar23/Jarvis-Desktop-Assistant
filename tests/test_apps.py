import pytest

from jarvis import config
from jarvis.apps import App, AppCatalog
from jarvis.text import normalize

NOTEPAD = App("Notepad", ("notepad",), "notepad.exe", "exe", ("Notepad.exe",))
EXPLORER = App("File Explorer", ("file explorer", "my files"), "explorer.exe", "exe", ("explorer.exe",))


@pytest.mark.parametrize(
    ("words", "found"),
    [
        ("notepad", NOTEPAD),
        ("the notepad", NOTEPAD),
        ("up notepad", NOTEPAD),
        ("notepad app", NOTEPAD),
        ("notepad please", NOTEPAD),
        ("the notepad application for me please", NOTEPAD),
        ("all notepad windows", NOTEPAD),
        ("my files", EXPLORER),  # "my" is part of this alias, not filler
        ("the file explorer window", EXPLORER),
        ("", None),
        ("the", None),
        ("note", None),
        ("notepad and chrome", None),
        ("my notepad", None),
        ("files", None),
    ],
)
def test_find(words: str, found: App | None) -> None:
    assert AppCatalog([NOTEPAD, EXPLORER]).find(words) == found


def test_duplicate_alias_rejected() -> None:
    other = App("Other", ("notepad",), "other.exe", "exe", ("other.exe",))
    with pytest.raises(ValueError, match="used by two apps"):
        AppCatalog([NOTEPAD, other])


# --- the whitelist table in config.py --------------------------------------------


def test_whitelist_has_the_requested_apps_and_aliases() -> None:
    assert {app.name: app.aliases for app in config.APPS} == {
        "Notepad": ("notepad",),
        "Chrome": ("chrome", "google", "google chrome", "browser"),
        "Spotify": ("spotify",),
        "VS Code": ("vs code", "visual studio code", "code"),
        "File Explorer": ("file explorer", "explorer", "my files"),
        "Calculator": ("calculator", "calc"),
        "Settings": ("settings",),
        "Task Manager": ("task manager",),
    }


def test_aliases_are_unique_and_normalized() -> None:
    catalog = AppCatalog(config.APPS)  # raises on duplicates
    for app in catalog.apps:
        for alias in app.aliases:
            assert normalize(alias) == alias, f"{alias!r} would never match"


@pytest.mark.parametrize("app", config.APPS, ids=lambda app: app.name)
def test_launch_methods_are_well_formed(app: App) -> None:
    assert app.processes, "close needs to know which processes are the app's"
    assert all(name.lower().endswith(".exe") for name in app.processes)
    if app.launch_kind == "exe":
        assert app.launch.lower().endswith(".exe")
    elif app.launch_kind == "uri":
        assert app.launch.endswith(":")
    else:
        assert app.launch.lower().endswith(".lnk")


def test_explorer_only_closes_folder_windows_and_is_never_ended() -> None:
    explorer = next(app for app in config.APPS if app.name == "File Explorer")
    assert explorer.window_classes == ("CabinetWClass",)
    assert explorer.force_close is False
    assert all(app.force_close for app in config.APPS if app is not explorer)
