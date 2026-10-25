from jarvis import config
from jarvis.window_targets import (
    FRAME_WINDOW_CLASS,
    ProcessInfo,
    WindowInfo,
    select_targets,
)

ME = r"PC\me"
APPS = {app.name: app for app in config.APPS}


def window(hwnd: int, pid: int, class_name: str = "Main", **flags: object) -> WindowInfo:
    values: dict[str, object] = {"visible": True, "cloaked": False, "owned": False}
    values.update(flags)
    return WindowInfo(hwnd=hwnd, pid=pid, class_name=class_name, **values)  # type: ignore[arg-type]


def test_only_whitelisted_process_names_of_the_current_user() -> None:
    processes = [
        ProcessInfo(1, "Notepad.exe", ME),
        ProcessInfo(2, "notepad.exe", ME),  # case-insensitive
        ProcessInfo(3, "Notepad.exe", r"PC\someone_else"),
        ProcessInfo(4, "Notepad.exe", None),  # SYSTEM / unknown owner
        ProcessInfo(5, "Notepad++.exe", ME),
        ProcessInfo(6, "chrome.exe", ME),
    ]
    windows = [window(10 + pid, pid) for pid in range(1, 7)]
    targets = select_targets(APPS["Notepad"], processes, windows, ME)
    assert targets.pids == {1, 2}
    assert targets.windows == (11, 12)


def test_only_visible_uncloaked_main_windows() -> None:
    processes = [ProcessInfo(1, "Notepad.exe", ME)]
    windows = [
        window(10, 1),
        window(11, 1, visible=False),
        window(12, 1, cloaked=True),
        window(13, 1, owned=True),  # its "Save changes?" dialog
    ]
    assert select_targets(APPS["Notepad"], processes, windows, ME).windows == (10,)


def test_store_app_frames_are_selected_but_their_host_is_never_a_target() -> None:
    processes = [
        ProcessInfo(1, "CalculatorApp.exe", ME),
        ProcessInfo(2, "ApplicationFrameHost.exe", ME),
        ProcessInfo(3, "SystemSettings.exe", ME),
    ]
    windows = [
        window(20, 2, FRAME_WINDOW_CLASS, child_pids=frozenset({2, 1})),  # Calculator
        window(21, 2, FRAME_WINDOW_CLASS, child_pids=frozenset({2, 3})),  # Settings
        window(22, 2, FRAME_WINDOW_CLASS, child_pids=frozenset({2})),  # empty frame
    ]
    targets = select_targets(APPS["Calculator"], processes, windows, ME)
    assert targets.windows == (20,)
    assert targets.pids == {1}  # force close may end CalculatorApp.exe, never the host


def test_explorer_closes_folder_windows_only() -> None:
    processes = [ProcessInfo(1, "explorer.exe", ME)]
    windows = [
        window(30, 1, "CabinetWClass"),  # a folder window
        window(31, 1, "Shell_TrayWnd"),  # the taskbar
        window(32, 1, "Progman"),  # the desktop
        window(33, 1, "CabinetWClass"),
    ]
    assert select_targets(APPS["File Explorer"], processes, windows, ME).windows == (30, 33)


def test_nothing_running() -> None:
    targets = select_targets(APPS["Spotify"], [], [window(1, 1)], ME)
    assert targets.pids == frozenset()
    assert targets.windows == ()
