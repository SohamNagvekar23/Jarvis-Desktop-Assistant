"""Parse spoken numbers ("forty-five", "a hundred", "40%") from normalized text."""

_UNITS: dict[str, int] = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS: dict[str, int] = {
    word: value
    for word, value in zip(
        "twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)
    )
}
_FILLER: frozenset[str] = frozenset({"percent", "please", "now"})
_NEGATIVE: frozenset[str] = frozenset({"minus", "negative"})


def parse_number(text: str) -> int | None:
    """Return the integer spoken in text, or None if it is not exactly one number.

    Expects normalize()d text: lowercase words, "%" and hyphens already removed,
    and a leading "-" on digits rewritten as "minus".
    """
    tokens = [token for token in text.split() if token not in _FILLER]
    sign = 1
    if tokens and tokens[0] in _NEGATIVE:
        sign, tokens = -1, tokens[1:]
    if len(tokens) == 1 and tokens[0].isdigit():
        return sign * int(tokens[0])
    value = _parse_words(tokens)
    return None if value is None else sign * value


def _parse_words(tokens: list[str]) -> int | None:
    """Parse "[a|one..nine] hundred [and] <0-99>" or "<0-99>"."""
    if "hundred" not in tokens:
        return _parse_below_hundred(tokens)
    split = tokens.index("hundred")
    multiplier_words, rest = tokens[:split], tokens[split + 1 :]
    if multiplier_words in ([], ["a"]):
        multiplier = 1
    elif len(multiplier_words) == 1 and 1 <= _UNITS.get(multiplier_words[0], 0) <= 9:
        multiplier = _UNITS[multiplier_words[0]]
    else:
        return None
    if rest[:1] == ["and"]:
        rest = rest[1:]
        if not rest:
            return None
    if not rest:
        return multiplier * 100
    remainder = _parse_below_hundred(rest)
    if not remainder:  # None, or "hundred zero"
        return None
    return multiplier * 100 + remainder


def _parse_below_hundred(tokens: list[str]) -> int | None:
    if len(tokens) == 1:
        word = tokens[0]
        return _UNITS.get(word, _TENS.get(word))
    if len(tokens) == 2 and tokens[0] in _TENS and 1 <= _UNITS.get(tokens[1], 0) <= 9:
        return _TENS[tokens[0]] + _UNITS[tokens[1]]
    return None
