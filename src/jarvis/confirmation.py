"""Decide whether a spoken or typed answer is a clear "yes"."""

from jarvis.text import normalize

# The whole answer must be one of these (after dropping politeness words).
# Anything else, including silence, Whisper's "Thank you." hallucination,
# "yes no" or "I guess", is treated as no.
_YES: frozenset[str] = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "ok",
        "okay",
        "confirm",
        "do it",
        "go ahead",
        "close it",
        "yes do it",
        "yes go ahead",
        "yes close it",
        "yes yes",
    }
)
# Accepted only when typed: a lone letter in a transcript is too likely to be noise.
_TYPED_YES: frozenset[str] = frozenset({"y"})
_POLITE: frozenset[str] = frozenset({"please", "jarvis", "thanks", "thank you"})


def is_clear_yes(answer: str, *, typed: bool = False) -> bool:
    words = normalize(answer)
    for polite in _POLITE:
        words = f" {words} ".replace(f" {polite} ", " ").strip()
    words = " ".join(words.split())
    return words in _YES or (typed and words in _TYPED_YES)
