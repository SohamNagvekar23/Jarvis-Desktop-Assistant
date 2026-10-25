import pytest

from jarvis.brain import (
    Answer,
    Brain,
    BrainReply,
    CommandRequest,
    Rejected,
    build_system_prompt,
    parse_reply,
    shorten,
)

ALLOWED = {"mute", "set_volume"}


def parse(raw: str) -> BrainReply:
    return parse_reply(raw, ALLOWED, max_sentences=3, max_chars=300)


def test_answer() -> None:
    assert parse('{"answer": "  It is   sunny.  "}') == Answer("It is sunny.")


def test_command_with_and_without_args() -> None:
    assert parse('{"command": "set_volume", "args": {"level": 5}}') == CommandRequest(
        "set_volume", {"level": 5}
    )
    assert parse('{"command": "mute"}') == CommandRequest("mute", {})


def test_command_args_are_checked_by_the_command_not_here() -> None:
    # Each command validates its own args (see test_commands); the brain only
    # guarantees the id is whitelisted and args is an object.
    assert parse('{"command": "set_volume", "args": {"level": 999}}') == CommandRequest(
        "set_volume", {"level": 999}
    )


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "",
        "[1, 2]",
        '{"command": "shutdown"}',
        '{"command": "mute", "args": null}',
        '{"answer": "x", "answer": "y"}',
        '{"answer": "x", "command": "mute"}',
        '{"thought": "hmm"}',
    ],
)
def test_rejected(raw: str) -> None:
    assert isinstance(parse(raw), Rejected)


@pytest.mark.parametrize(
    ("text", "short"),
    [
        ("One. Two. Three. Four.", "One. Two. Three."),
        ("**Bold** and `code` and # heading", "Bold and code and heading"),
        ("Line one.\n\nLine two.", "Line one. Line two."),
        ("No full stop at all", "No full stop at all"),
    ],
)
def test_shorten(text: str, short: str) -> None:
    assert shorten(text, 3, 300) == short


def test_shorten_caps_run_on_sentences() -> None:
    result = shorten("word " * 200, 3, 50)
    assert len(result) <= 53
    assert result.endswith("...")


def test_prompt_lists_commands_and_the_two_reply_forms() -> None:
    prompt = build_system_prompt({"mute": "Mute the sound. args: {}", "lock_screen": "Lock. args: {}"})
    assert "- mute: Mute the sound. args: {}" in prompt
    assert "- lock_screen: Lock. args: {}" in prompt
    assert '{"command": "<id>", "args": {...}}' in prompt
    assert '{"answer": "<reply>"}' in prompt


def test_live_information_rule_only_when_search_google_exists() -> None:
    rule = "use search_google with a short query. Never answer that you have no real-time data."
    with_search = build_system_prompt({"search_google": "Search Google. args: {}"})
    without_search = build_system_prompt({"mute": "Mute. args: {}"})
    assert rule in with_search
    assert "search_google" not in without_search  # never point the model at a missing command


class RecordingLlm:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


def test_brain_sends_the_same_system_prompt_every_time() -> None:
    llm = RecordingLlm('{"answer": "Hi."}')
    brain = Brain(llm, {"mute": "Mute. args: {}"}, 3, 300)
    assert brain.ask("hello") == Answer("Hi.")
    assert brain.ask("hi again") == Answer("Hi.")
    assert llm.calls[0][0] == llm.calls[1][0] == brain.system_prompt
    assert [user for _, user in llm.calls] == ["hello", "hi again"]
