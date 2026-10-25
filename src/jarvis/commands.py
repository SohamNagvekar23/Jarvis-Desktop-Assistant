"""Whitelist of voice commands, trigger-phrase matching and the LLM fallback.

Every action Jarvis can take must be registered in build_registry(). OS access
is passed in (see app.py), so tests can substitute fakes.
"""

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jarvis import config
from jarvis.apps import App, AppCatalog, CloseResult
from jarvis.brain import Answer, Brain, BrainOffline, BrainTimeout, CommandRequest, LlmClient
from jarvis.spoken_numbers import parse_number
from jarvis.text import is_negated, normalize
from jarvis.web import (
    Site,
    SiteCatalog,
    clean_query,
    is_valid_query,
    search_url,
    split_on_youtube,
    watch_url,
)

logger = logging.getLogger(__name__)

# Validates JSON args from the LLM; returns a ready-to-run action, or None if invalid.
ArgBinder = Callable[[Mapping[str, object]], Callable[[], str] | None]


class Volume(Protocol):
    def get_level(self) -> int: ...
    def set_level(self, percent: int) -> None: ...
    def set_muted(self, muted: bool) -> None: ...


class AppWindows(Protocol):
    def is_open(self, app: App) -> bool: ...
    def close(self, app: App) -> CloseResult:
        """Ask the app's windows to close (like clicking X) and wait briefly."""
        ...

    def force_close(self, app: App) -> CloseResult:
        """End the app's own processes."""
        ...


@dataclass(frozen=True)
class Command:
    name: str
    # Phrases; "*" marks a slot in the middle ("play * on youtube"). A trailing
    # "*" ("play *") means words must follow, and not just filler or a
    # standalone phrase like "next". Without "*", the rest of the sentence
    # follows the phrase.
    triggers: tuple[str, ...]
    # Receives the words spoken after the trigger phrase (or in its "*" slot).
    handler: Callable[[str], str]
    # For the LLM: what the command does and its args, shown in the system prompt.
    description: str
    bind_args: ArgBinder
    # False: the handler gets normalized words. True: it gets the original
    # transcript text, with case, punctuation and non-English letters intact
    # (search queries like "C++" or "AT&T").
    raw_rest: bool = False
    # True: a trigger only counts if nothing but filler follows it, so short
    # phrases like "next" or "play" don't fire inside "what's next on my
    # calendar" or "play Despacito" (which then go to other phrases or the LLM).
    standalone: bool = False
    # Log tag for the UI: "cmd", "web", or a function of the words after the
    # trigger ("open youtube" is web, "open notepad" is cmd).
    tag: str | Callable[[str], str] = "cmd"


@dataclass(frozen=True)
class Reply:
    text: str  # to speak; "" means reply silently (media keys)
    tag: str  # "cmd" (a phrase command), "brain" (the LLM) or "web"
    brain_online: bool | None = None  # set when the LLM was asked


@dataclass(frozen=True)
class Match:
    command: Command
    rest: str  # what the handler receives, see Command.raw_rest
    negated: bool  # "don't", "do not" or "never" came before the trigger


# Words that may follow a standalone trigger: "next song please", "pause it now".
_STANDALONE_FILLER = frozenset({"please", "now", "jarvis", "thanks", "thank", "you", "for", "me", "it", "again"})


def _only_filler(words: str) -> bool:
    return all(word in _STANDALONE_FILLER for word in words.split())


def _has_non_ascii_words(raw: str | None) -> bool:
    """normalize() drops non-English letters, so check the original text too."""
    return raw is not None and any(ch.isalnum() and not ch.isascii() for ch in raw)


def _without_trailing_filler(words: str) -> str:
    tokens = words.split()
    while tokens and tokens[-1] in _STANDALONE_FILLER:
        tokens.pop()
    return " ".join(tokens)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Find a normalized phrase in original text, the way normalize() splits words."""
    words = phrase.split()
    body = r"[^A-Za-z0-9]+".join(re.escape(word) for word in words)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE | re.ASCII)


@dataclass(frozen=True)
class _Trigger:
    prefix: str  # padded normalized phrase before the slot, e.g. " play "
    suffix: str  # padded normalized phrase after the slot, e.g. " on youtube ", or ""
    needs_rest: bool  # trailing "*": real words must follow the prefix
    command: Command
    prefix_pattern: re.Pattern[str]
    suffix_pattern: re.Pattern[str] | None

    @classmethod
    def parse(cls, trigger: str, command: Command) -> "_Trigger":
        before, star, after = trigger.partition("*")
        prefix, suffix = normalize(before), normalize(after)
        if not prefix:
            raise ValueError(f"trigger {trigger!r} must start with words")
        return cls(
            f" {prefix} ",
            f" {suffix} " if suffix else "",
            bool(star) and not suffix,
            command,
            _phrase_pattern(prefix),
            _phrase_pattern(suffix) if suffix else None,
        )

    def find(self, padded: str) -> tuple[int, int, int] | None:
        """(start, rest start, rest end) in padded normalized text, or None."""
        start = padded.find(self.prefix)
        if start == -1:
            return None
        rest_start = start + len(self.prefix) - 1
        if not self.suffix:
            return start, rest_start, len(padded)
        rest_end = padded.rfind(self.suffix)  # the last one, so the query may contain it
        return (start, rest_start, rest_end) if rest_end >= rest_start else None

    def raw_rest(self, text: str) -> str | None:
        """The original text in the slot / after the phrase; None if not found."""
        prefix = self.prefix_pattern.search(text)
        if prefix is None:
            return None
        if self.suffix_pattern is None:
            return text[prefix.end() :]
        suffixes = list(self.suffix_pattern.finditer(text, prefix.end()))
        return text[prefix.end() : suffixes[-1].start()] if suffixes else None


class CommandRegistry:
    def __init__(
        self,
        commands: Iterable[Command],
        llm: LlmClient | None = None,
        llm_examples: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.commands: tuple[Command, ...] = tuple(commands)
        self._by_name = {command.name: command for command in self.commands}
        triggers: dict[tuple[str, str, bool], _Trigger] = {}
        for command in self.commands:
            for phrase in command.triggers:
                trigger = _Trigger.parse(phrase, command)
                key = (trigger.prefix, trigger.suffix, trigger.needs_rest)
                if key in triggers:
                    raise ValueError(f"trigger {phrase!r} is registered twice")
                triggers[key] = trigger
        self._triggers = tuple(triggers.values())
        # Phrases that contain a negation but are requests ("i don t like this
        # song"); their "don't" must not negate a later phrase ("..., skip it").
        self._negating_phrases = tuple(
            t.prefix for t in self._triggers if is_negated(t.prefix.strip())
        )
        # Phrases like "next song": "play next song" means that, not a search.
        self._standalone_phrases = frozenset(
            t.prefix.strip() for t in self._triggers if t.command.standalone and not t.suffix
        )
        self.brain = (
            None
            if llm is None
            else Brain(
                llm,
                {command.name: command.description for command in self.commands},
                config.BRAIN_MAX_ANSWER_SENTENCES,
                config.BRAIN_MAX_ANSWER_CHARS,
                llm_examples,
            )
        )

    def match(self, text: str) -> Match | None:
        """Find the command whose trigger appears as whole words in text.

        The trigger that starts earliest wins ("close the lock screen settings"
        is a close command). Between triggers starting at the same word, the
        longest wins ("mute off" beats "mute").
        """
        padded = f" {normalize(text)} "
        best: tuple[tuple[int, int], _Trigger, tuple[int, int, int]] | None = None
        for trigger in self._triggers:
            span = trigger.find(padded)
            if span is None:
                continue
            words = padded[span[1] : span[2]]
            nothing_said = _only_filler(words) and not _has_non_ascii_words(trigger.raw_rest(text))
            if trigger.command.standalone and not nothing_said:
                continue
            if trigger.needs_rest and (
                nothing_said or _without_trailing_filler(words) in self._standalone_phrases
            ):
                continue
            rank = (span[0], -(len(trigger.prefix) + len(trigger.suffix)))
            if best is None or rank < best[0]:
                best = (rank, trigger, span)
        if best is None:
            return None
        _, trigger, (start, rest_start, rest_end) = best
        rest = padded[rest_start:rest_end].strip()
        if trigger.command.raw_rest:
            # Trigger words never involve normalize()'s spelling merges, so the
            # first match in the original text is the same occurrence.
            raw = trigger.raw_rest(text)
            rest = raw if raw is not None else rest
        return Match(trigger.command, rest=rest, negated=self._is_negated(padded[: start + 1]))

    def _is_negated(self, padded_words: str) -> bool:
        for phrase in self._negating_phrases:
            padded_words = padded_words.replace(phrase, " ")
        return is_negated(padded_words.strip())

    def handle(self, text: str) -> str:
        """Run the command matching text and return the reply to speak."""
        return self.respond(text).text

    def respond(self, text: str) -> Reply:
        """Like handle(), plus where the reply came from (for the UI log)."""
        if not normalize(text):
            return Reply("Sorry, I didn't catch that.", "cmd")
        found = self.match(text)
        if found is None:
            return self._ask_brain(text)
        if found.negated:
            return Reply("Okay, I won't.", "cmd")
        command = found.command
        tag = command.tag(found.rest) if callable(command.tag) else command.tag
        return Reply(command.handler(found.rest), tag)

    def _ask_brain(self, text: str) -> Reply:
        """Only for speech that matched no trigger phrase."""
        if self.brain is None:
            return Reply("Sorry, I don't know that command yet.", "cmd")
        try:
            reply = self.brain.ask(text)
        except BrainOffline as error:
            logger.warning("Brain offline: %s", error)
            return Reply("My brain is offline.", "brain", brain_online=False)
        except BrainTimeout as error:
            logger.warning("Brain timeout: %s", error)
            return Reply("Sorry, my brain took too long to answer.", "brain", brain_online=True)
        if isinstance(reply, Answer):
            return Reply(reply.text, "brain", brain_online=True)
        if isinstance(reply, CommandRequest):
            if self._is_negated(f" {normalize(text)} "):  # "I don't want the screen locked"
                return Reply("Okay, I won't.", "brain", brain_online=True)
            command = self._by_name.get(reply.command)
            run = command.bind_args(reply.args) if command else None
            if command is None or run is None:
                logger.warning("Rejected brain command %r with args %r", reply.command, reply.args)
                return Reply("Sorry, I can't do that.", "brain", brain_online=True)
            tag = "web" if command.tag == "web" else "brain"
            return Reply(run(), tag, brain_online=True)
        logger.warning("Rejected brain reply: %s", reply.reason)
        return Reply("Sorry, I didn't understand that.", "brain", brain_online=True)


def _no_args(run: Callable[[], str]) -> ArgBinder:
    return lambda args: run if not args else None


def build_registry(
    *,
    apps: Iterable[App],
    launch_app: Callable[[App], None],
    app_windows: AppWindows,
    confirm: Callable[[str], bool],
    volume: Volume,
    take_screenshot: Callable[[], Path],
    lock_screen: Callable[[], None],
    press_media_key: Callable[[str], None] | None = None,
    sites: Iterable[Site] = (),
    open_url: Callable[[str], None] | None = None,
    find_top_video: Callable[[str], str | None] | None = None,
    llm: LlmClient | None = None,
) -> CommandRegistry:
    """Build the whitelist. apps are the installed whitelisted apps; sites the
    whitelisted websites; open_url opens an https URL in the default browser;
    confirm(question) asks the user and returns True only on a clear yes;
    press_media_key presses a media key by name (play_pause, next, previous, stop);
    find_top_video returns the top YouTube result's video id for a query, or None;
    llm, if given, answers speech that matches no trigger phrase."""
    catalog = AppCatalog(apps)
    site_catalog = SiteCatalog(sites)
    app_names = ", ".join(f'"{app.name}"' for app in catalog.apps)
    site_names = ", ".join(f'"{site.name}"' for site in site_catalog.sites)
    max_query = config.SEARCH_MAX_QUERY_CHARS

    # --- actions shared by spoken phrases and LLM requests ---

    def open_app(app: App) -> str:
        try:
            launch_app(app)
        except OSError:
            return f"Sorry, I couldn't open {app.name}."
        return f"Opening {app.name}."

    def close_app(app: App) -> str:
        try:
            if not app_windows.is_open(app):
                return f"{app.name} isn't open."
            if not confirm(f"Close {app.name}? Say yes or no."):
                return f"Okay, I'll leave {app.name} open."
            result = app_windows.close(app)
            if result is CloseResult.STILL_OPEN:
                if not app.force_close:
                    return f"Sorry, {app.name} didn't close."
                question = (
                    f"{app.name} is still open. It may be asking to save your work. "
                    "Force it to close? Say yes or no."
                )
                if not confirm(question):
                    return f"Okay, {app.name} is still open."
                result = app_windows.force_close(app)
                if result is CloseResult.CLOSED:
                    return f"Force closed {app.name}."
        except OSError:
            return f"Sorry, I couldn't close {app.name}."
        if result is CloseResult.CLOSED:
            return f"Closed {app.name}."
        if result is CloseResult.NOT_OPEN:
            return f"{app.name} isn't open."
        if result is CloseResult.DENIED:
            return f"Sorry, I'm not allowed to close {app.name}. It may be running as administrator."
        return f"Sorry, I couldn't close {app.name}."

    def apply_volume(level: int) -> str:
        try:
            volume.set_level(level)
            volume.set_muted(False)  # like the volume keys, changing volume unmutes
        except OSError:
            return "Sorry, I couldn't change the volume."
        return f"Volume set to {level} percent."

    def step_volume(direction: int) -> str:
        try:
            current = volume.get_level()
            level = min(100, max(0, current + direction * config.VOLUME_STEP_PERCENT))
            volume.set_level(level)
            volume.set_muted(False)
        except OSError:
            return "Sorry, I couldn't change the volume."
        if level == current:
            return f"Volume is already at {level} percent."
        return f"Volume {'up' if direction > 0 else 'down'} to {level} percent."

    def set_muted(muted: bool) -> str:
        try:
            volume.set_muted(muted)
        except OSError:
            return "Sorry, I couldn't change the volume."
        return "Muted." if muted else "Unmuted."

    def screenshot() -> str:
        try:
            path = take_screenshot()
        except OSError:
            return "Sorry, I couldn't take a screenshot."
        return f"Screenshot saved as {path.name}."

    def lock() -> str:
        try:
            lock_screen()
        except OSError:
            return "Sorry, I couldn't lock the screen."
        return "Locking the screen."

    def browse(url: str, success: str, failure: str) -> str:
        if open_url is None:
            return failure
        try:
            open_url(url)
        except (OSError, ValueError):
            return failure
        return success

    def open_site(site: Site) -> str:
        return browse(site.url, f"Opening {site.name}.", f"Sorry, I couldn't open {site.name}.")

    def search(engine: str, label: str) -> Callable[[str], str]:
        """Runs a search for an already-cleaned, validated query."""

        def run(query: str) -> str:
            url = search_url(config.SEARCH_URL_TEMPLATES[engine], query)
            return browse(url, f"Searching {label} for {query}.", f"Sorry, I couldn't search {label}.")

        return run

    youtube_only = search("youtube", "YouTube")
    google_only = search("google", "Google")

    def youtube_query(action: Callable[[str], str]) -> Callable[[str], str]:
        def run(query: str) -> str:
            # "play Despacito on YT" (via "play *") is about "Despacito".
            on_youtube = split_on_youtube(query)
            if on_youtube == "":
                return "What should I search for?"
            return action(query if on_youtube is None else on_youtube)

        return run

    search_youtube = youtube_query(youtube_only)

    def play_top_video(query: str) -> str:
        """Open the top YouTube result directly; on any failure, the results page."""
        video_id = None
        if find_top_video is not None:
            try:
                video_id = find_top_video(query)
            except Exception:  # a broken lookup must never break "play X"
                logger.exception("YouTube lookup failed")
        url = watch_url(config.YOUTUBE_WATCH_URL_TEMPLATE, video_id) if video_id else None
        if url is None:
            if video_id:
                logger.warning("Ignoring invalid YouTube video id %r", video_id)
            return youtube_only(query)  # "Searching YouTube for X." and nothing extra
        return browse(url, f"Playing {query} on YouTube.", "Sorry, I couldn't open YouTube.")

    play_youtube = youtube_query(play_top_video)

    def search_google(query: str) -> str:
        # "search for cats on YouTube" means a YouTube search for "cats".
        on_youtube = split_on_youtube(query)
        if on_youtube is None:
            return google_only(query)
        if not on_youtube:
            return "What should I search for?"
        return search_youtube(on_youtube)

    def media(key: str) -> Callable[[], str]:
        def run() -> str:
            if press_media_key is None:
                return "Sorry, I couldn't control the music."
            try:
                press_media_key(key)
            except OSError:
                return "Sorry, I couldn't control the music."
            return ""  # silent: speaking over the music is annoying

        return run

    # --- spoken phrases: parse the words after the trigger ---

    def open_handler(rest: str) -> str:
        if not rest:
            return "Open which app?"
        app = catalog.find(rest)
        if app is not None:
            return open_app(app)
        site = site_catalog.find(rest)
        if site is not None:
            return open_site(site)
        return "Sorry, I don't know that app."

    def search_handler(run: Callable[[str], str]) -> Callable[[str], str]:
        def handler(raw_rest: str) -> str:
            query = clean_query(raw_rest)
            if not query:
                return "What should I search for?"
            if not is_valid_query(query, max_query):
                return "Sorry, that search is too long."
            return run(query)

        return handler

    def app_handler(action: Callable[[App], str], verb: str) -> Callable[[str], str]:
        def handler(rest: str) -> str:
            if not rest:
                return f"{verb} which app?"
            app = catalog.find(rest)
            if app is None:
                return "Sorry, I don't know that app."
            return action(app)

        return handler

    def set_volume_handler(rest: str) -> str:
        words = rest.split()
        if words[:1] in (["to"], ["at"]):
            words = words[1:]
        level = parse_number(" ".join(words))
        if level is None:
            return "Sorry, I need a volume level from 0 to 100."
        if not 0 <= level <= 100:
            return f"Sorry, {level} is out of range. Volume goes from 0 to 100."
        return apply_volume(level)

    def step_handler(direction: int) -> Callable[[str], str]:
        def handler(rest: str) -> str:
            if rest.split()[:1] == ["to"]:  # "turn the volume up to fifty"
                return set_volume_handler(rest)
            return step_volume(direction)

        return handler

    # --- LLM requests: strict JSON args ---

    def confirmed_lock() -> str:
        # The model may have guessed (e.g. "I'm stepping away"), so it asks first;
        # the spoken phrase "lock the screen" locks straight away.
        if not confirm("Lock the screen? Say yes or no."):
            return "Okay, I won't lock the screen."
        return lock()

    def bind_app(action: Callable[[App], str]) -> ArgBinder:
        def bind(args: Mapping[str, object]) -> Callable[[], str] | None:
            name = args.get("app")
            if set(args) != {"app"} or not isinstance(name, str):
                return None
            app = catalog.find(normalize(name))
            return None if app is None else (lambda: action(app))

        return bind

    def bind_open(args: Mapping[str, object]) -> Callable[[], str] | None:
        name = args.get("app")
        if set(args) != {"app"} or not isinstance(name, str):
            return None
        app = catalog.find(normalize(name))
        if app is not None:
            return lambda: open_app(app)
        site = site_catalog.find(normalize(name))
        return None if site is None else (lambda: open_site(site))

    def bind_query(run: Callable[[str], str]) -> ArgBinder:
        def bind(args: Mapping[str, object]) -> Callable[[], str] | None:
            query = args.get("query")
            if set(args) != {"query"} or not isinstance(query, str):
                return None
            query = " ".join(query.split())
            if not is_valid_query(query, max_query):
                return None  # empty, too long, or control characters
            return lambda: run(query)

        return bind

    def bind_level(args: Mapping[str, object]) -> Callable[[], str] | None:
        level = args.get("level")
        if set(args) != {"level"} or type(level) is not int or not 0 <= level <= 100:
            return None  # rejects true/false, 50.0, "50", out of range and extra args
        return lambda: apply_volume(level)

    # Worked examples for the LLM prompt; app examples only for installed apps.
    installed = {app.name for app in catalog.apps}
    llm_examples = [
        ("I want to watch funny cat videos", '{"command": "search_youtube", "args": {"query": "funny cat videos"}}'),
        ("what's the weather in Mumbai today", '{"command": "search_google", "args": {"query": "weather in Mumbai today"}}'),
        ("did Real Madrid win last night", '{"command": "search_google", "args": {"query": "Real Madrid result last night"}}'),
        ("I'd love to hear Shape of You", '{"command": "play_youtube", "args": {"query": "Shape of You"}}'),
        ("this track is terrible", '{"command": "media_next", "args": {}}'),
        ("wait, the phone is ringing, hold it", '{"command": "media_play_pause", "args": {}}'),
        ("I want to hear the one before this again", '{"command": "media_previous", "args": {}}'),
        ("it's too loud, make it twenty percent", '{"command": "set_volume", "args": {"level": 20}}'),
        ("I can't hear anything", '{"command": "volume_up", "args": {}}'),
    ]
    if "Spotify" in installed:
        llm_examples.append(("put some music on", '{"command": "open_app", "args": {"app": "Spotify"}}'))
    if "Chrome" in installed:
        llm_examples.append(
            ("I'm done browsing for today", '{"command": "close_app", "args": {"app": "Chrome"}}')
        )

    return CommandRegistry(
        (
            Command(
                "open_app",
                ("open", "launch", "start"),
                open_handler,
                f'Open an app or website. args: {{"app": one of {app_names}, {site_names}}}',
                bind_open,
                tag=lambda rest: "web" if not catalog.find(rest) and site_catalog.find(rest) else "cmd",
            ),
            Command(
                "search_google",
                ("search for", "google", "search google for", "look up"),
                search_handler(search_google),
                "Search Google in the browser. Use it for current information you can't "
                "know (news, weather, prices, scores). "
                f'args: {{"query": the search words, at most {max_query} characters}}',
                bind_query(search_google),
                raw_rest=True,
                tag="web",
            ),
            Command(
                "search_youtube",
                ("search youtube for", "youtube", "find * on youtube"),
                search_handler(search_youtube),
                "Show YouTube search results in the browser, to browse or choose from. "
                f'args: {{"query": the search words, at most {max_query} characters}}',
                bind_query(search_youtube),
                raw_rest=True,
                tag="web",
            ),
            Command(
                "play_youtube",
                ("play * on youtube", "play *", "put on *"),
                search_handler(play_youtube),
                "Play a specific song or video: opens the top YouTube result directly. "
                f'args: {{"query": the song or video, at most {max_query} characters}}',
                bind_query(play_youtube),
                raw_rest=True,
                tag="web",
            ),
            Command(
                "close_app",
                ("close", "quit", "exit"),
                app_handler(close_app, "Close"),
                f'Close an app (Jarvis asks the user to confirm). args: {{"app": one of {app_names}}}',
                bind_app(close_app),
            ),
            Command(
                "volume_up",
                (
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
                step_handler(+1),
                f"Raise the volume by {config.VOLUME_STEP_PERCENT} percent. args: {{}}",
                _no_args(lambda: step_volume(+1)),
            ),
            Command(
                "volume_down",
                (
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
                step_handler(-1),
                f"Lower the volume by {config.VOLUME_STEP_PERCENT} percent. args: {{}}",
                _no_args(lambda: step_volume(-1)),
            ),
            Command(
                "set_volume",
                (
                    "set volume",
                    "set the volume",
                    "change volume",
                    "change the volume",
                    "volume to",
                    "volume at",
                ),
                set_volume_handler,
                'Set the volume. args: {"level": whole number from 0 to 100}',
                bind_level,
            ),
            Command(
                "mute",
                (
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
                lambda _: set_muted(True),
                "Mute the sound. args: {}",
                _no_args(lambda: set_muted(True)),
            ),
            Command(
                "unmute",
                (
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
                lambda _: set_muted(False),
                "Unmute the sound. args: {}",
                _no_args(lambda: set_muted(False)),
            ),
            Command(
                "screenshot",
                (
                    "screenshot",
                    "take a screenshot",
                    "take screenshot",
                    "capture the screen",
                    "capture screen",
                    "print screen",
                ),
                lambda _: screenshot(),
                "Take a screenshot of all screens. args: {}",
                _no_args(screenshot),
            ),
            Command(
                "lock_screen",
                (
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
                lambda _: lock(),
                "Lock the PC. args: {}",
                _no_args(confirmed_lock),
            ),
            Command(
                "media_play_pause",
                ("pause", "pause the music", "resume", "play", "play the music", "continue playing"),
                lambda _: media("play_pause")(),
                "Play or pause the music or video that is playing (one toggle key). args: {}",
                _no_args(media("play_pause")),
                standalone=True,
            ),
            Command(
                "media_next",
                (
                    "next",
                    "next song",
                    "next track",
                    "skip",
                    "skip this song",
                    "change the song",
                    "change it",
                    # Requests, not negated commands: the "don't" / "can't" is part
                    # of the phrase, and negation only counts before a phrase.
                    "i don't like this song",
                    "don't like this song",
                    "can't stand this song",
                ),
                lambda _: media("next")(),
                "Skip to the next song or track. args: {}",
                _no_args(media("next")),
                standalone=True,
            ),
            Command(
                "media_previous",
                ("previous", "previous song", "go back a song", "last track"),
                lambda _: media("previous")(),
                "Go to the previous song or track. args: {}",
                _no_args(media("previous")),
                standalone=True,
            ),
            Command(
                "media_stop",
                ("stop the music", "stop playing"),
                lambda _: media("stop")(),
                "Stop the music or video. args: {}",
                _no_args(media("stop")),
                standalone=True,
            ),
        ),
        llm,
        llm_examples,
    )
