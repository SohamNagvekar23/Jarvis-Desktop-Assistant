"""Normalizing transcribed speech for matching."""

import re

_NEGATIVE_DIGITS = re.compile(r"(?<![a-z0-9])-(?=[0-9])")
_NON_WORD = re.compile(r"[^a-z0-9]+")
# Spellings Whisper produces interchangeably, mapped to the one triggers use.
_SPELLINGS: tuple[tuple[str, str], ...] = (
    (" un mute ", " unmute "),
    (" note pad ", " notepad "),
    (" screen shot ", " screenshot "),
    (" per cent ", " percent "),
    (" vscode ", " vs code "),
    (" v s code ", " vs code "),
    (" dont ", " don t "),
    (" cant ", " can t "),
)
# Normalized forms of "don't", "dont", "do not", "never".
_NEGATIONS: tuple[str, ...] = (" don t ", " dont ", " do not ", " never ")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace and unify spellings."""
    text = _NEGATIVE_DIGITS.sub(" minus ", text.lower())
    padded = f" {_NON_WORD.sub(' ', text).strip()} "
    for variant, canonical in _SPELLINGS:
        while variant in padded:
            padded = padded.replace(variant, canonical)
    return padded.strip()


def is_negated(normalized_prefix: str) -> bool:
    """True if the normalized words before a command contain a negation."""
    padded = f" {normalized_prefix} "
    return any(negation in padded for negation in _NEGATIONS)
