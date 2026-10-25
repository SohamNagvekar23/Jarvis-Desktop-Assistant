import pytest

from jarvis.confirmation import is_clear_yes


@pytest.mark.parametrize(
    "answer",
    [
        "yes",
        "Yes.",
        "YES!",
        "Yes, please.",
        "Yeah.",
        "Yep",
        "Yup.",
        "Sure.",
        "Okay.",
        "OK",
        "Yes, close it.",
        "Close it.",
        "Do it.",
        "Go ahead.",
        "Yes, Jarvis.",
        "Yes, thank you.",
        "Yes yes",
        "confirm",
    ],
)
@pytest.mark.parametrize("typed", [False, True], ids=["spoken", "typed"])
def test_clear_yes(answer: str, typed: bool) -> None:
    assert is_clear_yes(answer, typed=typed)


@pytest.mark.parametrize("answer", ["y", "Y", " y ", "y."])
def test_y_counts_only_when_typed(answer: str) -> None:
    assert is_clear_yes(answer, typed=True)
    assert not is_clear_yes(answer)  # spoken (the default)


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "   ",
        "no",
        "No.",
        "Nope.",
        "No, don't.",
        "Don't close it.",
        "Yes, no.",
        "No, yes.",
        "Yes, don't.",
        "Maybe.",
        "I guess.",
        "Wait.",
        "Hold on.",
        "Thank you.",  # Whisper's usual hallucination on silence
        "you",
        "Yes and open Chrome.",
        "yesterday",
        "eyes",
        "not sure",
    ],
)
@pytest.mark.parametrize("typed", [False, True], ids=["spoken", "typed"])
def test_anything_else_is_not_yes(answer: str, typed: bool) -> None:
    assert not is_clear_yes(answer, typed=typed)


@pytest.mark.parametrize("answer", ["n", "yy", "y n", "why"])
def test_other_short_typed_answers_are_not_yes(answer: str) -> None:
    assert not is_clear_yes(answer, typed=True)
