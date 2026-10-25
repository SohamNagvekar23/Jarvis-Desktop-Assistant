import pytest

from jarvis.text import is_negated, normalize


@pytest.mark.parametrize(
    ("text", "normalized"),
    [
        ("Open Notepad.", "open notepad"),
        ("open note pad", "open notepad"),
        ("Un-mute!", "unmute"),
        ("Take a screen shot", "take a screenshot"),
        ("Open VSCode", "open vs code"),
        ("Open V.S. Code", "open vs code"),
        ("set volume to 40%", "set volume to 40"),
        ("set volume to 40 per cent", "set volume to 40 percent"),
        ("set volume to forty-five", "set volume to forty five"),
        ("set volume to -10", "set volume to minus 10"),
        ("Don't", "don t"),
        ("I dont like it", "i don t like it"),  # same as "don't"
        ("I cant stand it", "i can t stand it"),  # same as "can't"
        ("  ...  ", ""),
    ],
)
def test_normalize(text: str, normalized: str) -> None:
    assert normalize(text) == normalized


@pytest.mark.parametrize(
    ("prefix", "negated"),
    [
        ("don t", True),
        ("please don t", True),
        ("dont", True),
        ("do not", True),
        ("never", True),
        ("never mind", True),
        ("", False),
        ("do", False),
        ("not", False),  # only the forms in the safety rule
        ("donut", False),
        ("nevermore", False),
    ],
)
def test_is_negated(prefix: str, negated: bool) -> None:
    assert is_negated(prefix) is negated
