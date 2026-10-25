"""All tunable settings for Jarvis live here."""

from jarvis.apps import App
from jarvis.web import Site

# Audio capture (Whisper and openWakeWord expect 16 kHz mono).
SAMPLE_RATE: int = 16_000
# Fixed recording length in --push-to-talk mode.
RECORD_SECONDS: float = 4.0

# Wake word (openWakeWord prebuilt model, ONNX runtime).
WAKE_WORD_MODEL: str = "hey_jarvis"
WAKE_THRESHOLD: float = 0.5  # 0..1; raise if it fires on its own, lower if it misses you
CHUNK_SAMPLES: int = 1280  # 80 ms at 16 kHz, the frame size openWakeWord is built for

# Recording the command after the wake word (Silero VAD bundled with openWakeWord).
VAD_THRESHOLD: float = 0.5  # 0..1 speech probability per chunk
END_SILENCE_SECONDS: float = 0.8  # stop this long after speech ends
MAX_RECORD_SECONDS: float = 8.0
NO_SPEECH_TIMEOUT_SECONDS: float = 4.0  # give up if no speech starts within this

# Microphone stream.
MAX_QUEUED_SECONDS: float = 5.0  # audio kept while busy; older chunks are dropped
MIC_STALL_SECONDS: float = 2.0  # no audio for this long means the mic stream died
MIC_RETRY_SECONDS: float = 2.0  # wait before reopening the mic after an error

# Beep played when the wake word is heard.
BEEP_FREQUENCY_HZ: int = 880
BEEP_DURATION_MS: int = 120

# Speech-to-text (faster-whisper). int8 on CPU keeps memory and latency low.
WHISPER_MODEL: str = "base"
WHISPER_DEVICE: str = "cpu"
WHISPER_COMPUTE_TYPE: str = "int8"
WHISPER_LANGUAGE: str = "en"
WHISPER_BEAM_SIZE: int = 1

# Text-to-speech (pyttsx3), words per minute.
TTS_RATE: int = 180

# Whitelisted applications for "open <app>" / "close <app>". The only apps Jarvis
# will start, and the only processes it will ever close. Apps that aren't
# installed are skipped at startup. No install paths are hard-coded: Windows
# resolves exe names (PATH / App Paths), URIs and Start menu shortcuts.
APPS: tuple[App, ...] = (
    App("Notepad", ("notepad",), "notepad.exe", "exe", ("Notepad.exe",)),
    App(
        "Chrome",
        ("chrome", "google", "google chrome", "browser"),
        "chrome.exe",
        "exe",
        ("chrome.exe",),
    ),
    App("Spotify", ("spotify",), "spotify:", "uri", ("Spotify.exe",)),
    App(
        "VS Code",
        ("vs code", "visual studio code", "code"),
        "Visual Studio Code.lnk",
        "shortcut",
        ("Code.exe",),
    ),
    # Only folder windows are closed; explorer.exe also runs the taskbar and
    # desktop, so it is never ended.
    App(
        "File Explorer",
        ("file explorer", "explorer", "my files"),
        "explorer.exe",
        "exe",
        ("explorer.exe",),
        window_classes=("CabinetWClass",),
        force_close=False,
    ),
    App(
        "Calculator",
        ("calculator", "calc"),
        "calc.exe",
        "exe",
        ("CalculatorApp.exe", "Calculator.exe"),
    ),
    App("Settings", ("settings",), "ms-settings:", "uri", ("SystemSettings.exe",)),
    App("Task Manager", ("task manager",), "taskmgr.exe", "exe", ("Taskmgr.exe",)),
)

# Whitelisted websites for "open <site>". Only https URLs.
SITES: tuple[Site, ...] = (
    Site("YouTube", ("youtube",), "https://www.youtube.com/"),
    Site("Gmail", ("gmail",), "https://mail.google.com/"),
    Site("GitHub", ("github", "git hub"), "https://github.com/"),
)

# Search URL templates; {query} is replaced by the URL-encoded query and nothing
# else about the URL can change.
SEARCH_URL_TEMPLATES: dict[str, str] = {
    "google": "https://www.google.com/search?q={query}",
    "youtube": "https://www.youtube.com/results?search_query={query}",
}
SEARCH_MAX_QUERY_CHARS: int = 200

# "play X": open the top YouTube result directly, found with yt-dlp. If the lookup
# fails or takes longer than this, the search results page opens instead.
YOUTUBE_WATCH_URL_TEMPLATE: str = "https://www.youtube.com/watch?v={id}"
YOUTUBE_LOOKUP_TIMEOUT_SECONDS: float = 5.0

# "close <app>": how long to wait for windows to close after asking them to
# (like clicking X), and after force closing.
CLOSE_GRACE_SECONDS: float = 5.0
FORCE_CLOSE_WAIT_SECONDS: float = 3.0

# LLM "brain" (local Ollama server), used only when no command phrase matches.
OLLAMA_URL: str = "http://localhost:11434"
BRAIN_MODEL: str = "llama3.2:3b"
BRAIN_TIMEOUT_SECONDS: float = 20.0  # per question
BRAIN_WARMUP_TIMEOUT_SECONDS: float = 120.0  # loading the model at startup
BRAIN_KEEP_ALIVE: str = "30m"  # keep the model loaded this long after each request
BRAIN_MAX_TOKENS: int = 150  # caps reply length and latency
BRAIN_TEMPERATURE: float = 0.2
BRAIN_MAX_ANSWER_SENTENCES: int = 3
BRAIN_MAX_ANSWER_CHARS: int = 300

# Desktop UI (pywebview on Edge WebView2). Window positions are remembered in
# <the user's roaming AppData>\<UI_STATE_SUBFOLDER>\<UI_STATE_FILENAME>.
UI_MAIN_SIZE: tuple[int, int] = (1440, 820)
UI_MAIN_MIN_SIZE: tuple[int, int] = (760, 560)
UI_MINI_SIZE: tuple[int, int] = (420, 96)
UI_MINI_MARGIN: tuple[int, int] = (24, 72)  # from the screen's right and bottom edges
UI_METER_INTERVAL_SECONDS: float = 2.0
UI_STATE_SUBFOLDER: str = "Jarvis"
UI_STATE_FILENAME: str = "ui.json"
HOLD_MAX_SECONDS: float = 15.0  # hold-to-talk records at most this long
HOLD_MIN_SECONDS: float = 0.3  # shorter holds are treated as taps and ignored
CONFIRM_TYPED_TIMEOUT_SECONDS: float = 10.0  # how long the UI waits for a typed yes/no

# "Start with Windows": a per-user value under HKCU\<STARTUP_RUN_KEY>, no admin.
STARTUP_RUN_KEY: str = r"Software\Microsoft\Windows\CurrentVersion\Run"
STARTUP_VALUE_NAME: str = "Jarvis"
STARTUP_DELAY_SECONDS: float = 15.0  # with --startup, wait this long before loading models

# Only one Jarvis per Windows session ("Local\" names are per logged-in user).
SINGLE_INSTANCE_MUTEX: str = r"Local\Jarvis.SingleInstance"
SINGLE_INSTANCE_EVENT: str = r"Local\Jarvis.Activate"

# Log file: <the user's local AppData>\<LOG_SUBFOLDER>\<LOG_FILENAME>, rotated.
LOG_SUBFOLDER: str = "Jarvis"
LOG_FILENAME: str = "jarvis.log"
LOG_MAX_BYTES: int = 1_000_000
LOG_BACKUP_COUNT: int = 3  # jarvis.log.1 .. .3: at most about 4 MB in total

# Recording the spoken yes/no answer to a confirmation question.
CONFIRM_END_SILENCE_SECONDS: float = 0.6
CONFIRM_MAX_SECONDS: float = 3.0
CONFIRM_NO_SPEECH_TIMEOUT_SECONDS: float = 4.0

# Volume change for "volume up" / "volume down", in percentage points.
VOLUME_STEP_PERCENT: int = 10

# Screenshots go to <the user's Pictures folder>\<SCREENSHOT_SUBFOLDER>.
SCREENSHOT_SUBFOLDER: str = "Jarvis"
SCREENSHOT_PREFIX: str = "screenshot"

QUIT_WORDS: frozenset[str] = frozenset({"q", "quit", "exit"})
