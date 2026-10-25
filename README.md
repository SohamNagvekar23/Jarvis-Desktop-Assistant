# Jarvis

A Windows voice assistant in Python.

Say **"Hey Jarvis"**, wait for the beep, then say a command. Jarvis records
until you stop talking, transcribes it (faster-whisper `base`, on CPU), runs
the matching whitelisted command, and replies out loud (pyttsx3).

## Setup

Requires Windows and Python 3.11+.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On first run, faster-whisper downloads the `base` model (about 145 MB) from
Hugging Face and caches it. openWakeWord downloads its models (about 10 MB:
`hey_jarvis`, its feature extractors and Silero VAD) from its GitHub releases
into its package folder.

## Run

```powershell
python -m jarvis                  # desktop UI, always listening for "Hey Jarvis"
python -m jarvis --no-ui          # console only, always listening for "Hey Jarvis"
python -m jarvis --push-to-talk   # console fallback: press Enter, speak for 4 seconds
```

In the console modes, press Ctrl+C to quit (or type `q` in push-to-talk mode).

### Start with Windows

```powershell
python -m jarvis --install-startup   # start Jarvis at login
python -m jarvis --remove-startup    # stop starting Jarvis at login
```

You can also use the switch in the UI's settings panel.

- This adds a per-user value named `Jarvis` under
  `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`, so no admin rights are
  needed. It runs this venv's `pythonw.exe -m jarvis --startup` by absolute
  path, so no console window appears.
- If you move the project or recreate the venv, run `--install-startup` again
  (or switch the setting on again) to point the entry at the new location. The
  settings panel tells you when the entry is out of date.
- At login (`--startup`), Jarvis opens as the mini pill and waits
  `STARTUP_DELAY_SECONDS` (15 s) before loading its models, so it doesn't slow
  down logging in.
- Only one Jarvis runs at a time. Launching it again brings the running one's
  window to the front instead.
- Logs go to `%LOCALAPPDATA%\Jarvis\jarvis.log`, rotated at 1 MB with 3 old
  files kept. If Jarvis can't start, the UI shows the error and where the log
  is; if even the UI can't open, a message box does.

### Desktop UI

A frameless window (pywebview on Edge WebView2; the WebView2 runtime ships
with Windows 11). Everything is local, in `src/jarvis/ui/`: HTML, CSS, JS and
the Space Grotesk and JetBrains Mono fonts (woff2, with their SIL Open Font
License texts). The pages make no network requests.

- **Left:** CPU, Jarvis's memory, GPU memory (NVIDIA only, read through the
  driver's `nvml.dll`; shows N/A otherwise) and volume, updated every 2 s.
  Status dots for microphone, speech-to-text, brain, voice and YouTube
  lookup. The wake word and its threshold.
- **Center:** the orb shows what Jarvis is doing: STANDBY, WAKE DETECTED,
  LISTENING, PROCESSING (amber) or RESPONDING. It is animated with CSS only.
- **Right:** today's log. Each entry shows what you said, Jarvis's reply and a
  tag: CMD (a command phrase), BRAIN (the LLM) or WEB (searches and sites).
- **Bottom:** mute, a command box, HOLD TO TALK, quick actions and settings.
  - Typed commands and quick actions go through exactly the same pipeline as
    speech: whitelist, negation and confirmations.
  - Confirmation questions appear above the box with YES / NO buttons. You can
    also type or say the answer.
  - Mute closes the microphone, so wake words aren't heard. Typing and hold to
    talk still work.
- **Minimize** switches to a small always-on-top pill near the bottom right.
  Click its orb to bring the full window back. Both windows remember where you
  dragged them (`%APPDATA%\Jarvis\ui.json`). Close quits Jarvis.
- The UI runs on the main thread and the assistant on a background thread.
  They're connected by a thread-safe event queue that only sends changes
  (`ui_bridge.py`).

### Brain (local LLM)

Anything that matches no command phrase goes to a local LLM through
[Ollama](https://ollama.com) (`OLLAMA_URL`, default `http://localhost:11434`),
using `BRAIN_MODEL` (default `llama3.2:3b`; get it with `ollama pull llama3.2:3b`).

- The model gets your words plus a system prompt listing the whitelisted
  commands, and must reply with JSON: `{"command": "<id>", "args": {...}}` or
  `{"answer": "..."}`. Answers are spoken, cut to three sentences.
- The reply is validated strictly. Only whitelisted command ids with valid args
  run, through the same code as the spoken phrases, so "close" still asks for
  confirmation. If the model picks lock_screen (for example for "I'm stepping
  away"), Jarvis asks "Lock the screen? Say yes or no" first; the spoken phrase
  "lock the screen" locks straight away. Other model-chosen actions run
  directly. Anything else (invented commands, wrong args, non-JSON, extra
  keys) is refused and nothing runs. If your words contain "don't", "do not" or
  "never", the model can answer but can't run a command.
- The model is loaded at startup in the background and kept loaded for
  `BRAIN_KEEP_ALIVE` (30 minutes) after each question. A question takes about
  2.5 s on this PC. If Ollama isn't running, Jarvis says "My brain is offline."
  and normal commands keep working.
- `llama3.2:3b` sometimes picks a nearby command for something Jarvis can't do,
  for example opening File Explorer for "delete my files". It can only choose
  from the whitelist, so this is wrong but not harmful.

### How listening works

- One continuous 16 kHz mono microphone stream, read in 80 ms chunks. Each
  chunk is scored by openWakeWord's `hey_jarvis` model on ONNX Runtime (TFLite
  isn't available on Windows). That uses about 3 to 10% of one CPU core.
- When the score reaches `WAKE_THRESHOLD`, Jarvis beeps and records. Silero
  VAD decides chunk by chunk whether you're speaking. Recording stops after
  `END_SILENCE_SECONDS` (0.8 s) of silence once you've started, at
  `MAX_RECORD_SECONDS` (8 s), or after `NO_SPEECH_TIMEOUT_SECONDS` (4 s) if
  you never start.
- While a command is handled, wake words are ignored. Audio that queued up
  meanwhile (the beep, Jarvis's own reply) is discarded, and the wake-word
  model is reset so it can't fire on stale sound.
- Errors are logged and Jarvis keeps listening. If the microphone fails or is
  unplugged, it reopens the stream every `MIC_RETRY_SECONDS`.
- Pause briefly after "Hey Jarvis": anything you say during the beep is
  discarded, so "Hey Jarvis open Chrome" said in one breath may lose the
  command.

All thresholds and timings are in `src/jarvis/config.py`. If the wake word
fires by itself, raise `WAKE_THRESHOLD`; if it misses you, lower it.

## Commands

| Command         | Say, for example                                                         |
|-----------------|--------------------------------------------------------------------------|
| Open an app     | "open <app>", "launch <app>", "start <app>"                             |
| Close an app    | "close <app>", "quit <app>", "exit <app>" (asks first)                  |
| Open a website  | "open youtube", "open gmail", "open github"                             |
| Google search   | "search for X", "google X", "search google for X", "look up X"          |
| Play on YouTube | "play X", "put on X", "play X on youtube" (opens the top video)          |
| YouTube search  | "search youtube for X", "youtube X", "find X on youtube" (results page) |
| Volume up       | "volume up", "turn it up", "crank it up", "pump it up", "louder"        |
| Volume down     | "volume down", "turn down the volume", "lower the volume", "quieter"    |
| Set volume      | "set volume to 40", "set the volume to forty-five", "volume at 40%"     |
| Mute            | "mute", "mute the sound", "silence", "silence please", "quiet"          |
| Unmute          | "unmute", "unmute the sound", "turn off mute", "mute off"               |
| Screenshot      | "take a screenshot", "screenshot", "capture the screen", "print screen" |
| Lock the screen | "lock the screen", "lock my computer", "lock my pc"                     |
| Play / pause    | "pause", "pause the music", "resume", "play", "play the music", "continue playing" |
| Next track      | "next", "next song", "skip", "change the song", "change it", "I don't like this song", "can't stand this song" |
| Previous track  | "previous", "previous song", "go back a song", "last track"            |
| Stop            | "stop the music", "stop playing"                                        |

The full phrase list for each command is in `src/jarvis/commands.py`.

### Media keys

- Media commands press the Windows media keys (`media_keys.py`, `SendInput`),
  so they control whatever player Windows routes them to: Spotify, YouTube in
  the browser, and most players. They reply silently, so Jarvis doesn't talk
  over the music; only a failure is spoken.
- Play and pause are the same key, a toggle: "pause" while paused resumes.
  "Previous" in most players restarts the current song first.
- These phrases only count on their own, optionally followed by "please",
  "now" or similar, so "what's next on my calendar" and "skip lunch" don't
  press keys.
- "Play X", "put on X" and "play X on YouTube" open the top YouTube video for
  X directly ("Playing Despacito on YouTube."). Jarvis finds it with yt-dlp:
  one flat search result, nothing downloaded, imported only on first use (about
  +35 MB). The id must be exactly 11 of `[A-Za-z0-9_-]` and opens as
  `https://www.youtube.com/watch?v=<id>`. If the lookup fails, takes over
  `YOUTUBE_LOOKUP_TIMEOUT_SECONDS` (5 s) or finds nothing, the results page
  opens instead, with the normal "Searching YouTube for X." reply. A lookup
  takes about 1.5 to 2 s. "Search YouTube for X" always opens the results page.
- yt-dlp is deliberately unpinned. If "play X" starts opening results pages
  every time, YouTube has probably changed: run
  `python -m pip install -U yt-dlp`.
- "Play" on its own, "play it" or "play the music" is play/pause. "Play" plus a media
  phrase does that action: "play next song" skips, and "put on pause" pauses.
- "I don't like this song" and "can't stand this song" are requests to skip.
  Their "don't" doesn't count as negation, even before another phrase ("I
  don't like this song, skip it"). "Don't change the song" is still refused.
- The brain can choose them too ("this song is so boring, change it" skips).

### Web search and sites

- Searches open in your default browser. URLs are built only from the fixed
  https templates in `SEARCH_URL_TEMPLATES` (`config.py`), with the query
  URL-encoded, so speech can't produce any other site, path or scheme.
  Websites come from the `SITES` whitelist (YouTube, Gmail, GitHub).
- The query keeps your exact words, including case, symbols ("C++", "AT&T")
  and non-English text. Punctuation Whisper adds at the ends ("cats.") and a
  trailing "please" are removed. Jarvis replies "Searching YouTube for X."
- An empty query gets "What should I search for?". Queries are capped at
  `SEARCH_MAX_QUERY_CHARS` (200).
- A Google search whose query ends with "on YouTube" or "on YT" searches
  YouTube instead: "search for cats on YouTube" searches YouTube for "cats".
- The brain can choose searches too, for example "show me cute dog videos"
  searches YouTube. For anything needing current information (news, scores,
  prices, weather, results) it searches Google instead of answering from memory.
- The Windows voice may not pronounce non-English queries; the search itself
  is unaffected.

### Apps

Apps come from one whitelist table, `APPS` in `src/jarvis/config.py`.

| App           | Say "open ..." / "close ..."                       | Launched via                    |
|---------------|----------------------------------------------------|---------------------------------|
| Notepad       | notepad                                            | `notepad.exe`                   |
| Chrome        | chrome, google, google chrome, browser             | `chrome.exe`                    |
| Spotify       | spotify                                            | `spotify:` URI                  |
| VS Code       | vs code, visual studio code, code                  | Start menu shortcut             |
| File Explorer | file explorer, explorer, my files                  | `explorer.exe`                  |
| Calculator    | calculator, calc                                   | `calc.exe`                      |
| Settings      | settings                                           | `ms-settings:` URI              |
| Task Manager  | task manager                                       | `taskmgr.exe`                   |

- At startup, apps that aren't installed are skipped (Jarvis prints which).
  Any other name gets "Sorry, I don't know that app." and nothing runs.
- "Close Spotify" asks "Close Spotify? Say yes or no", beeps, and records your
  answer without the wake word. Only a clear yes ("yes", "yeah", "sure", "go
  ahead"...) closes the app; anything else, including silence, leaves it open.
  In `--push-to-talk` mode you type the answer instead, and `y` also counts.
- Closing is graceful: Jarvis asks the app's windows to close, as if you
  clicked X. If the app stays open (for example, it's asking to save), Jarvis
  asks again before force closing. Force closing only ends that app's own
  processes, running as you.
- File Explorer: only folder windows are closed. `explorer.exe` also runs the
  taskbar and desktop, so it is never ended.
- Task Manager running as administrator can't be closed by a non-admin
  Jarvis; it says so.

- Phrases match anywhere in what you say, ignoring case and punctuation, so
  "Hey Jarvis, open Chrome please" works. If more than one phrase fits, the
  one that starts earliest wins: "close the lock screen settings" is a close
  command, not a lock. Between phrases starting at the same word, the longest
  wins, so "mute off" unmutes instead of muting.
- "Don't", "do not" or "never" before a command cancels it: "don't close
  Spotify" gets "Okay, I won't." and nothing runs.
- Volume levels can be digits ("40", "40%") or words ("forty", "forty-five",
  "a hundred"). Anything outside 0 to 100 is refused out loud.
  "Turn the volume up to fifty" sets the volume to 50.
- Volume up/down step by 10% (`VOLUME_STEP_PERCENT`). Changing the volume also
  unmutes, like the keyboard volume keys.
- Screenshots cover all monitors and are saved as PNG files in
  `Pictures\Jarvis` (wherever Windows keeps your Pictures folder, including
  OneDrive). An existing file is never overwritten: a clash gets `_1`, `_2`, ...
- After "mute", the spoken reply is silent because the speakers are muted. It
  is still printed in the console.

To add a command, register a `Command` with its trigger phrases in
`build_registry()` in `src/jarvis/commands.py`. Settings (recording length,
model, voice rate, app whitelist, volume step, screenshot folder, close and
confirmation timings, sites and search templates) are in `src/jarvis/config.py`. Windows-specific code
lives in `launcher.py`, `app_windows.py`, `volume.py`, `screenshot.py`,
`known_folders.py` and `lock.py`. Which processes and windows "close" may touch
is decided by the pure function in `window_targets.py`. The command
handlers receive these as arguments, so tests run against fakes. In the same
way, the listening logic (`listener.py`) gets its microphone
(`microphone.py`), wake-word model and VAD (`wakeword.py`) passed in, and is
tested with fake audio and a fake detector.

## Tests

```powershell
python -m pytest
```
