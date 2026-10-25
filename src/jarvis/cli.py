"""Command line entry point.

Kept deliberately light: flags, logging, the startup entry and the
single-instance check all happen before the assistant (and its heavy
libraries) is imported, so a second launch is detected at once and Jarvis
shows up quickly at login.
"""

import argparse
import ctypes
import io
import logging
import sys
from pathlib import Path

from jarvis import config, known_folders
from jarvis.logs import setup_logging
from jarvis.single_instance import SingleInstance
from jarvis.startup import StartupEntry, WindowsRunKey, startup_command

logger = logging.getLogger("jarvis.cli")


def log_file() -> Path:
    return known_folders.local_app_data() / config.LOG_SUBFOLDER / config.LOG_FILENAME


def startup_entry() -> StartupEntry:
    return StartupEntry(
        WindowsRunKey(config.STARTUP_RUN_KEY), config.STARTUP_VALUE_NAME, startup_command()
    )


def change_startup(install: bool) -> int:
    """--install-startup / --remove-startup. Returns the exit code."""
    try:
        entry = startup_entry()
        if install:
            result = entry.install()
            message = {
                "installed": "Jarvis will now start with Windows.",
                "updated": "Updated the startup entry to this location.",
                "unchanged": "Jarvis already starts with Windows.",
            }[result]
            print(f"{message}\n  {entry.command}")
            logger.info("Startup entry %s: %s", result, entry.command)
        elif entry.remove():
            print("Jarvis will no longer start with Windows.")
            logger.info("Startup entry removed")
        else:
            print("Jarvis wasn't set to start with Windows.")
    except OSError as error:
        print(f"Couldn't change the startup entry: {error}")
        logger.exception("Couldn't change the startup entry")
        return 1
    return 0


def show_fatal_error(message: str) -> None:
    """For errors outside the UI; pythonw.exe has no console to print to."""
    print(message)
    ctypes.windll.user32.MessageBoxW(None, message, "Jarvis", 0x10)  # MB_ICONERROR


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="jarvis", description="Windows voice assistant.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--no-ui",
        action="store_true",
        help='run in the console only, listening for "Hey Jarvis"',
    )
    modes.add_argument(
        "--push-to-talk",
        action="store_true",
        help='console only: press Enter to speak instead of saying "Hey Jarvis"',
    )
    modes.add_argument(
        "--startup",
        action="store_true",
        help="used by the Windows startup entry: open as the mini pill, load after a delay",
    )
    modes.add_argument("--install-startup", action="store_true", help="start Jarvis with Windows")
    modes.add_argument(
        "--remove-startup", action="store_true", help="stop starting Jarvis with Windows"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(log_file(), config.LOG_MAX_BYTES, config.LOG_BACKUP_COUNT)
    # The Windows console may use a legacy code page (cp1252); print non-English
    # transcripts and queries as escapes instead of crashing on them.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="backslashreplace")

    if args.install_startup or args.remove_startup:
        raise SystemExit(change_startup(install=args.install_startup))

    instance = SingleInstance(config.SINGLE_INSTANCE_MUTEX, config.SINGLE_INSTANCE_EVENT)
    if not instance.acquire():
        instance.signal_existing()
        logger.info("Already running; asked the running copy to come to the front")
        print("Jarvis is already running.")
        return
    logger.info("Jarvis starting (%s)", " ".join(sys.argv[1:]) or "UI")
    try:
        from jarvis import app  # only now: the assistant and its libraries

        if args.no_ui:
            app.run_console(push_to_talk=False)
        elif args.push_to_talk:
            app.run_console(push_to_talk=True)
        else:
            try:
                entry: StartupEntry | None = startup_entry()
            except OSError:
                entry = None  # no pythonw.exe: the settings toggle shows as unavailable
            app.run_with_ui(at_login=args.startup, startup=entry, instance=instance)
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as error:
        logger.exception("Jarvis stopped because of an error")
        show_fatal_error(
            f"Jarvis stopped because of an error: {error}\n\nDetails are in {log_file()}"
        )
    finally:
        instance.release()
        logger.info("Jarvis stopped")
    print("Goodbye.")
