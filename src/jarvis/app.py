"""Running Jarvis: the desktop UI or the console modes (see cli.py for the flags).

Heavy libraries (speech models, audio, COM) are imported inside the functions
that need them, so the UI appears quickly, especially at login.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from jarvis import config, known_folders, launcher
from jarvis.apps import App
from jarvis.assistant import WAKE_PROMPT, Assistant, UiConfirmer, WakeWordResponder
from jarvis.brain import BrainUnavailable
from jarvis.commands import CommandRegistry, build_registry
from jarvis.confirmation import is_clear_yes
from jarvis.ui_bridge import ACTIVE, ERROR, IDLE_LEVEL, NULL_PRESENTER, WARN, Presenter

if TYPE_CHECKING:
    from jarvis.ollama_client import OllamaClient
    from jarvis.single_instance import SingleInstance
    from jarvis.startup import StartupEntry
    from jarvis.stt import Transcriber
    from jarvis.tts import Speaker

logger = logging.getLogger(__name__)


def installed_apps() -> dict[App, str]:
    """Whitelisted apps found on this PC, with what to launch for each."""
    targets: dict[App, str] = {}
    for app in config.APPS:
        target = launcher.find_launch_target(app)
        if target is None:
            print(f"Skipping {app.name}: not installed.")
        else:
            targets[app] = target
    return targets


def start_brain(presenter: Presenter = NULL_PRESENTER) -> OllamaClient:
    """Create the Ollama client and load the model in the background."""
    from jarvis.ollama_client import OllamaClient
    client = OllamaClient(
        config.OLLAMA_URL,
        config.BRAIN_MODEL,
        config.BRAIN_TIMEOUT_SECONDS,
        config.BRAIN_KEEP_ALIVE,
        config.BRAIN_MAX_TOKENS,
        config.BRAIN_TEMPERATURE,
    )

    def warm_up() -> None:
        presenter.module("brain", IDLE_LEVEL, "LOADING")
        try:
            client.warm_up(config.BRAIN_WARMUP_TIMEOUT_SECONDS)
        except BrainUnavailable as error:
            logger.warning("My brain is offline (%s). Normal commands still work.", error)
            presenter.module("brain", WARN, "OFFLINE")
        else:
            presenter.module("brain", ACTIVE, "ONLINE")

    threading.Thread(target=warm_up, name="brain-warm-up", daemon=True).start()
    return client


def build_command_registry(
    confirm: Callable[[str], bool], presenter: Presenter = NULL_PRESENTER
) -> CommandRegistry:
    from jarvis.app_windows import WindowsAppWindows
    from jarvis.lock import lock_screen
    from jarvis.media_keys import press_media_key
    from jarvis.screenshot import ScreenCapture
    from jarvis.volume import MasterVolume
    from jarvis.youtube_lookup import YouTubeLookup

    def youtube_status(ok: bool) -> None:
        presenter.module("youtube", ACTIVE if ok else WARN, "READY" if ok else "FAILED")

    targets = installed_apps()
    return build_registry(
        llm=start_brain(presenter),
        apps=targets,
        launch_app=lambda app: launcher.launch(targets[app]),
        app_windows=WindowsAppWindows(config.CLOSE_GRACE_SECONDS, config.FORCE_CLOSE_WAIT_SECONDS),
        confirm=confirm,
        volume=MasterVolume(),
        take_screenshot=ScreenCapture(
            known_folders.pictures() / config.SCREENSHOT_SUBFOLDER, config.SCREENSHOT_PREFIX
        ).capture,
        lock_screen=lock_screen,
        press_media_key=press_media_key,
        sites=config.SITES,
        open_url=launcher.open_url,
        find_top_video=YouTubeLookup(
            config.YOUTUBE_LOOKUP_TIMEOUT_SECONDS, on_status=youtube_status
        ).top_video_id,
    )


def load_transcriber() -> Transcriber:
    from jarvis.stt import Transcriber

    print("Loading speech model...")
    return Transcriber(
        config.WHISPER_MODEL,
        config.WHISPER_DEVICE,
        config.WHISPER_COMPUTE_TYPE,
        config.WHISPER_LANGUAGE,
        config.WHISPER_BEAM_SIZE,
    )


def beep() -> None:
    from jarvis.sounds import beep as play_beep

    play_beep(config.BEEP_FREQUENCY_HZ, config.BEEP_DURATION_MS)


def build_listening():  # type: ignore[no-untyped-def]
    """The wake-word listener, the shared VAD, the microphone and the recorder
    for yes/no answers."""
    from jarvis.listener import Listener, ListenSettings, RecordSettings, UtteranceRecorder
    from jarvis.microphone import Microphone
    from jarvis.wakeword import OpenWakeWordDetector, SileroVad

    print("Loading wake word model...")
    vad = SileroVad()
    listener = Listener(
        OpenWakeWordDetector(config.WAKE_WORD_MODEL),
        vad,
        ListenSettings(
            sample_rate=config.SAMPLE_RATE,
            vad_threshold=config.VAD_THRESHOLD,
            end_silence_seconds=config.END_SILENCE_SECONDS,
            max_record_seconds=config.MAX_RECORD_SECONDS,
            no_speech_timeout_seconds=config.NO_SPEECH_TIMEOUT_SECONDS,
            wake_threshold=config.WAKE_THRESHOLD,
        ),
    )
    mic = Microphone(
        config.SAMPLE_RATE,
        config.CHUNK_SAMPLES,
        math.ceil(config.MAX_QUEUED_SECONDS * config.SAMPLE_RATE / config.CHUNK_SAMPLES),
        config.MIC_STALL_SECONDS,
    )
    answer_recorder = UtteranceRecorder(
        vad,  # shared model; reset at the start of every recording
        RecordSettings(
            sample_rate=config.SAMPLE_RATE,
            vad_threshold=config.VAD_THRESHOLD,
            end_silence_seconds=config.CONFIRM_END_SILENCE_SECONDS,
            max_record_seconds=config.CONFIRM_MAX_SECONDS,
            no_speech_timeout_seconds=config.CONFIRM_NO_SPEECH_TIMEOUT_SECONDS,
        ),
    )
    return listener, mic, answer_recorder


# --- console modes (--no-ui, --push-to-talk) ---


def run_push_to_talk(transcriber: Transcriber, speaker: Speaker) -> None:
    from jarvis.audio import Recorder

    def confirm(question: str) -> bool:
        speaker.say(question)
        return is_clear_yes(input("Type yes or no: "), typed=True)

    assistant = Assistant(build_command_registry(confirm), transcriber, speaker)
    recorder = Recorder(config.SAMPLE_RATE, config.RECORD_SECONDS)
    speaker.say("Jarvis is ready.")
    while True:
        entry = input("\nPress Enter and speak (q to quit): ")
        if entry.strip().lower() in config.QUIT_WORDS:
            return
        print(f"Listening for {config.RECORD_SECONDS:g} seconds...")
        assistant.respond(recorder.record())


def run_wake_word(transcriber: Transcriber, speaker: Speaker) -> None:
    from jarvis.listener import listen_forever, record_utterance

    listener, mic, answer_recorder = build_listening()

    def confirm(question: str) -> bool:
        """Ask out loud, then record the answer straight away (no wake word)."""
        speaker.say(question)
        beep()
        mic.discard_pending()  # drop the question and beep picked up by the mic
        audio = record_utterance(mic, answer_recorder)
        if audio is None:
            print("No answer heard.")
            return False
        answer = transcriber.transcribe(audio)
        print(f"You answered: {answer!r}")
        return is_clear_yes(answer)

    assistant = Assistant(build_command_registry(confirm), transcriber, speaker)
    speaker.say("Jarvis is ready.")
    print(WAKE_PROMPT)
    listen_forever(lambda: mic, listener, WakeWordResponder(assistant, beep), config.MIC_RETRY_SECONDS)


# --- desktop UI ---


def run_with_ui(
    *,
    at_login: bool = False,
    startup: StartupEntry | None = None,
    instance: SingleInstance | None = None,
) -> None:
    from jarvis.ui_app import run_ui
    from jarvis.ui_bridge import Controller, EventBus

    bus = EventBus()
    controller = Controller(bus)
    waiting = "WAITING" if at_login else "LOADING"
    for name, text in (
        ("microphone", "STARTING"),
        ("stt", waiting),
        ("brain", waiting),
        ("voice", waiting),
        ("youtube", "NOT LOADED"),
    ):
        bus.module(name, IDLE_LEVEL, text)
    delay = config.STARTUP_DELAY_SECONDS if at_login else 0.0
    run_ui(
        bus,
        controller,
        lambda: run_assistant_for_ui(bus, controller, delay),
        start_minimized=at_login,
        startup=startup,
        instance=instance,
    )


def run_assistant_for_ui(bus, controller, delay_seconds: float = 0.0) -> None:  # type: ignore[no-untyped-def]
    """The assistant thread behind the UI: loads the models, then runs the worker.

    delay_seconds: at login, wait this long first so Windows finishes starting.
    """
    if delay_seconds > 0:
        logger.info("Started at login; loading in %g s", delay_seconds)
        if controller.wait_stopped(delay_seconds):
            return  # closed before we even started
    for name in ("stt", "brain", "voice"):
        bus.module(name, IDLE_LEVEL, "LOADING")
    try:
        import comtypes  # SAPI voice, volume and app control all use COM on this thread

        comtypes.CoInitialize()
        from jarvis.assistant_worker import AssistantWorker
        from jarvis.listener import HoldRecorder

        from jarvis.tts import Speaker

        transcriber = load_transcriber()
        bus.module("stt", ACTIVE, "READY")
        speaker = Speaker(config.TTS_RATE)
        bus.module("voice", ACTIVE, "READY")
        listener, mic, answer_recorder = build_listening()
    except Exception as error:
        logger.exception("Jarvis couldn't start")
        bus.module("stt", ERROR, "FAILED")
        from jarvis.cli import log_file

        bus.error(f"Jarvis couldn't start: {error}. Details are in {log_file()}")
        return

    confirmer: list[UiConfirmer] = []  # set below; the registry needs confirm first
    registry = build_command_registry(lambda question: confirmer[0](question), bus)
    assistant = Assistant(registry, transcriber, speaker, bus)
    confirmer.append(
        UiConfirmer(
            assistant,
            transcriber,
            controller,
            bus,
            mic,
            mic,
            answer_recorder,
            beep,
            config.CONFIRM_TYPED_TIMEOUT_SECONDS,
        )
    )
    worker = AssistantWorker(
        assistant=assistant,
        controller=controller,
        presenter=bus,
        mic=mic,
        listener=listener,
        responder=WakeWordResponder(assistant, beep, bus),
        hold_recorder=HoldRecorder(config.SAMPLE_RATE, config.HOLD_MAX_SECONDS),
        min_hold_samples=round(config.SAMPLE_RATE * config.HOLD_MIN_SECONDS),
        retry_seconds=config.MIC_RETRY_SECONDS,
    )
    assistant.say("Jarvis is ready.")
    worker.run()


def run_console(push_to_talk: bool) -> None:
    """--no-ui and --push-to-talk: exactly the console behavior from before the UI."""
    from jarvis.tts import Speaker

    transcriber = load_transcriber()
    speaker = Speaker(config.TTS_RATE)
    if push_to_talk:
        run_push_to_talk(transcriber, speaker)
    else:
        run_wake_word(transcriber, speaker)
