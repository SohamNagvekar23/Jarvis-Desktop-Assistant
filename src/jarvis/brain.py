"""LLM fallback for speech that matches no command phrase.

The model only *suggests*: its JSON reply is parsed strictly here, and the
command registry re-validates any command and its args against the whitelist
before running it. Nothing the model invents is ever executed.
"""

import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class BrainUnavailable(Exception):
    """The LLM could not be asked."""


class BrainOffline(BrainUnavailable):
    """The LLM server isn't running or rejected the request."""


class BrainTimeout(BrainUnavailable):
    """The LLM didn't answer in time."""


class LlmClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw reply; raise BrainOffline or BrainTimeout."""
        ...


@dataclass(frozen=True)
class Answer:
    text: str  # ready to speak


@dataclass(frozen=True)
class CommandRequest:
    command: str  # a whitelisted command id; args not validated yet
    args: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Rejected:
    reason: str


BrainReply = Answer | CommandRequest | Rejected

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN = re.compile(r"[*_#`~>|]+")


# Generic examples of answers; command examples come from the registry, which
# knows which apps are installed.
_ANSWER_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("how far away is the moon", '{"answer": "About 384,000 kilometres on average."}'),
    ("empty the recycle bin", '{"answer": "Sorry, I can\'t do that."}'),
    ("restart the PC", '{"answer": "Sorry, I can\'t restart the PC."}'),
)


def build_system_prompt(
    commands: Mapping[str, str], command_examples: Sequence[tuple[str, str]] = ()
) -> str:
    """commands maps each command id to a description of what it does and its args;
    command_examples are (user words, JSON reply) pairs."""
    listing = "\n".join(f"- {name}: {description}" for name, description in commands.items())
    examples = "\n".join(
        f"User: {user}\n{reply}" for user, reply in (*command_examples, *_ANSWER_EXAMPLES)
    )
    live_information = (
        "- For anything that needs current or live information (news, scores, prices, "
        "weather, results), use search_google with a short query. Never answer that you "
        "have no real-time data.\n"
        if "search_google" in commands
        else ""
    )
    return (
        "You are Jarvis, a voice assistant on the user's Windows PC. The user's words "
        "were transcribed from speech and may contain small errors.\n"
        "Reply with exactly one JSON object and nothing else, in one of two forms:\n"
        '{"command": "<id>", "args": {...}} to run one of the commands below, or\n'
        '{"answer": "<reply>"} for anything else.\n\n'
        f"Commands (use these ids and args exactly):\n{listing}\n\n"
        "Rules:\n"
        "- If the user asks for one of these actions, even indirectly, reply with the "
        "command, not an answer. Only a command changes anything on the PC.\n"
        "- An answer never changes anything. Never say in an answer that you did, are "
        "doing or will do something on the PC, or what its state is now.\n"
        "- Never invent command ids, args or app names. If what the user wants is not "
        "exactly one of these commands, do not pick a similar one: answer that you can't.\n"
        "- If the user says not to do something, do not use a command.\n"
        f"{live_information}"
        "- You cannot do anything else on the PC (files, web, messages, shutting down, "
        "other apps). If asked, answer briefly that you can't.\n"
        "- Answers are spoken aloud: one to three short sentences, plain text, "
        "no lists, markdown or emoji.\n\n"
        f"Examples:\n{examples}"
    )


def parse_reply(raw: str, allowed_commands: Collection[str], max_sentences: int, max_chars: int) -> BrainReply:
    """Strictly parse the model's reply. Anything off-format is Rejected."""
    try:
        data = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (ValueError, TypeError) as error:
        return Rejected(f"not valid JSON: {error}")
    if not isinstance(data, dict):
        return Rejected("not a JSON object")

    keys = set(data)
    if keys == {"answer"}:
        if not isinstance(data["answer"], str):
            return Rejected("answer is not a string")
        text = shorten(data["answer"], max_sentences, max_chars)
        return Answer(text) if text else Rejected("empty answer")

    if keys in ({"command"}, {"command", "args"}):
        command = data["command"]
        args = data.get("args", {})
        if not isinstance(command, str) or command not in allowed_commands:
            return Rejected(f"unknown command {command!r}")
        if not isinstance(args, dict):
            return Rejected("args is not a JSON object")
        return CommandRequest(command, args)

    return Rejected(f"unexpected keys {sorted(keys)}")


def shorten(text: str, max_sentences: int, max_chars: int) -> str:
    """Plain text of at most max_sentences sentences and about max_chars characters."""
    text = " ".join(_MARKDOWN.sub(" ", text).split())
    text = " ".join(_SENTENCE_END.split(text)[:max_sentences])
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return text


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate keys in {keys}")
    return dict(pairs)


class Brain:
    def __init__(
        self,
        client: LlmClient,
        commands: Mapping[str, str],
        max_sentences: int,
        max_chars: int,
        command_examples: Sequence[tuple[str, str]] = (),
    ) -> None:
        self._client = client
        self._allowed = frozenset(commands)
        # Built once and identical every time, so Ollama can reuse its prompt cache.
        self.system_prompt = build_system_prompt(commands, command_examples)
        self._max_sentences = max_sentences
        self._max_chars = max_chars

    def ask(self, text: str) -> BrainReply:
        raw = self._client.complete(self.system_prompt, text)
        return parse_reply(raw, self._allowed, self._max_sentences, self._max_chars)
