import pytest

from jarvis.spoken_numbers import parse_number


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("0", 0),
        ("7", 7),
        ("40", 40),
        ("100", 100),
        ("40 percent", 40),
        ("40 percent please", 40),
        ("zero", 0),
        ("one", 1),
        ("nine", 9),
        ("ten", 10),
        ("fifteen", 15),
        ("nineteen", 19),
        ("twenty", 20),
        ("forty", 40),
        ("forty five", 45),
        ("ninety nine", 99),
        ("hundred", 100),
        ("a hundred", 100),
        ("one hundred", 100),
        ("a hundred percent", 100),
        ("one hundred and one", 101),
        ("a hundred and fifty", 150),
        ("one hundred twenty five", 125),
        ("two hundred", 200),
        ("nine hundred ninety nine", 999),
        ("1000", 1000),
        ("minus 10", -10),
        ("minus five", -5),
        ("negative forty", -40),
    ],
)
def test_parses(text: str, value: int) -> None:
    assert parse_number(text) == value


@pytest.mark.parametrize(
    "text",
    [
        "",
        "percent",
        "loud",
        "max",
        "a",
        "minus",
        "forty forty",
        "five forty",
        "twenty ten",
        "forty zero",
        "40 5",
        "40 five",
        "hundred zero",
        "a hundred and",
        "and fifty",
        "hundred hundred",
        "ten hundred",
        "zero hundred",
        "a thousand",
        "forty two hundred",
    ],
)
def test_rejects(text: str) -> None:
    assert parse_number(text) is None
