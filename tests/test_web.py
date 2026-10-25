from urllib.parse import parse_qs, urlsplit

import pytest

from jarvis import config
from jarvis.text import normalize
from jarvis.web import (
    Site,
    SiteCatalog,
    clean_query,
    is_valid_query,
    search_url,
    split_on_youtube,
    watch_url,
)

GOOGLE = config.SEARCH_URL_TEMPLATES["google"]
YOUTUBE = config.SEARCH_URL_TEMPLATES["youtube"]


# --- URL encoding ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "encoded"),
    [
        ("cats", "cats"),
        ("funny cats", "funny+cats"),
        ("AT&T", "AT%26T"),
        ("what is this?", "what+is+this%3F"),
        ("C# tutorial", "C%23+tutorial"),
        ("C++", "C%2B%2B"),
        ("a=b&c=d", "a%3Db%26c%3Dd"),
        ("50% off", "50%25+off"),
        ("path/to/file", "path%2Fto%2Ffile"),
        ("café crème", "caf%C3%A9+cr%C3%A8me"),
        ("नमस्ते", "%E0%A4%A8%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%A4%E0%A5%87"),
        ("東京 天気", "%E6%9D%B1%E4%BA%AC+%E5%A4%A9%E6%B0%97"),
        ("{query}", "%7Bquery%7D"),
    ],
)
def test_query_is_url_encoded(query: str, encoded: str) -> None:
    assert search_url(GOOGLE, query) == f"https://www.google.com/search?q={encoded}"
    assert search_url(YOUTUBE, query) == f"https://www.youtube.com/results?search_query={encoded}"


@pytest.mark.parametrize(
    "query",
    [
        "AT&T stock & bonds",
        "is it? #really",
        "a&q=evil&b",  # trying to add a second q parameter
        "x#fragment",
        "https://evil.example/phish",
        "javascript:alert(1)",
        "//evil.example",
        "@evil.example",
        "../../etc/passwd",
        "line one\r\nHost: evil.example",
        "नमस्ते दुनिया 東京 café 🎵",
        "%2F%3F already encoded",
        "",
    ],
)
@pytest.mark.parametrize("template", [GOOGLE, YOUTUBE], ids=["google", "youtube"])
def test_query_cannot_change_the_destination(template: str, query: str) -> None:
    url = search_url(template, query)
    built, fixed = urlsplit(url), urlsplit(template)
    assert built.scheme == "https"
    assert (built.netloc, built.path) == (fixed.netloc, fixed.path)
    assert built.fragment == ""
    params = parse_qs(built.query, keep_blank_values=True)
    (name,) = params  # exactly one parameter
    assert params[name] == [query]  # decodes back to exactly what was said


@pytest.mark.parametrize(
    "template",
    [
        "http://www.google.com/search?q={query}",
        "javascript:{query}",
        "https://www.google.com/search?q=",
        "https://www.google.com/{query}?q={query}",
    ],
)
def test_bad_templates_are_refused(template: str) -> None:
    with pytest.raises(ValueError):
        search_url(template, "cats")


def test_configured_templates_are_fixed_https_urls() -> None:
    assert set(config.SEARCH_URL_TEMPLATES) == {"google", "youtube"}
    for template in config.SEARCH_URL_TEMPLATES.values():
        assert template.startswith("https://")
        assert template.endswith("={query}")
        assert template.count("{query}") == 1


# --- query cleaning ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "query"),
    [
        (" cats.", "cats"),
        (" cats!", "cats"),
        (" cats?", "cats"),
        (" cats...", "cats"),
        (" cats\u2026", "cats"),
        (", cats,", "cats"),
        (' "cats".', "cats"),
        (" \u201ccats\u201d.", "cats"),
        (" cats, please.", "cats"),
        (" cats please", "cats"),
        (" cats, thank you!", "cats"),
        (" please.", ""),
        ("  funny\n cats  ", "funny cats"),
        (" C++.", "C++"),
        (" C#?", "C#"),
        (" AT&T.", "AT&T"),
        (" what is 2+2?", "what is 2+2"),
        (" -5 degrees", "-5 degrees"),
        (" Bohemian Rhapsody ", "Bohemian Rhapsody"),  # case kept
        (" नमस्ते दुनिया।", "नमस्ते दुनिया।"),  # not English punctuation; left alone
        (" 東京 天気", "東京 天気"),
        ("", ""),
        (" ...", ""),
    ],
)
def test_clean_query(raw: str, query: str) -> None:
    assert clean_query(raw) == query


@pytest.mark.parametrize(
    ("query", "rest"),
    [
        ("cats on YouTube", "cats"),
        ("cats on youtube", "cats"),
        ("cats on YT", "cats"),
        ("cats on Y.T", "cats"),  # "Y.T." after clean_query strips the last dot
        ("cats on you tube", "cats"),
        ("funny cats, on YouTube", "funny cats"),
        ("on YouTube", ""),
        ("on yt", ""),
        ("cats on YouTube music", None),
        ("cats on youtubers", None),
        ("cats in YouTube", None),
        ("catson YouTube", None),
        ("YouTube premium", None),
        ("cats", None),
    ],
)
def test_split_on_youtube(query: str, rest: str | None) -> None:
    assert split_on_youtube(query) == rest


WATCH = config.YOUTUBE_WATCH_URL_TEMPLATE


@pytest.mark.parametrize("video_id", ["kJQP7kiw5Fk", "0CACYW9SQMo", "a_b-c_d-e_f", "___________", "-----------"])
def test_watch_url_for_valid_ids(video_id: str) -> None:
    assert watch_url(WATCH, video_id) == f"https://www.youtube.com/watch?v={video_id}"


@pytest.mark.parametrize(
    "video_id",
    [
        "",
        "kJQP7kiw5F",  # 10
        "kJQP7kiw5Fkk",  # 12
        "kJQP7kiw5F&",
        "kJQP7kiw5F#",
        "kJQP7kiw5F?",
        "kJQP7kiw5F/",
        "kJQP7kiw5F=",
        "kJQP7kiw5F ",
        "kJQP7kiw5F.",
        "kJQP7kiw5F\n",
        "kJQP7kiw5Fk\n",  # fullmatch: no trailing newline sneaks through
        "kJQP7kiw5Fé",
        "../../../ab",
        "x&list=evil",
        "kJQP7kiw5Fk&autoplay=1",
    ],
)
def test_watch_url_rejects_invalid_ids(video_id: str) -> None:
    assert watch_url(WATCH, video_id) is None


def test_watch_template_is_a_fixed_https_url() -> None:
    assert WATCH == "https://www.youtube.com/watch?v={id}"
    with pytest.raises(ValueError):
        watch_url("http://www.youtube.com/watch?v={id}", "kJQP7kiw5Fk")


@pytest.mark.parametrize(
    ("query", "valid"),
    [
        ("cats", True),
        ("x" * 200, True),
        ("x" * 201, False),
        ("", False),
        ("cats\x00", False),
        ("cats\x1b[2J", False),
        ("नमस्ते", True),
    ],
)
def test_is_valid_query(query: str, valid: bool) -> None:
    assert is_valid_query(query, 200) is valid


# --- sites ----------------------------------------------------------------------


def test_configured_sites() -> None:
    assert {site.name: (site.aliases, site.url) for site in config.SITES} == {
        "YouTube": (("youtube",), "https://www.youtube.com/"),
        "Gmail": (("gmail",), "https://mail.google.com/"),
        "GitHub": (("github", "git hub"), "https://github.com/"),
    }
    catalog = SiteCatalog(config.SITES)  # validates https and unique aliases
    for site in catalog.sites:
        for alias in site.aliases:
            assert normalize(alias) == alias


def test_site_and_app_aliases_never_overlap() -> None:
    app_aliases = {alias for app in config.APPS for alias in app.aliases}
    site_aliases = {alias for site in config.SITES for alias in site.aliases}
    assert not app_aliases & site_aliases


@pytest.mark.parametrize(
    ("words", "name"),
    [
        ("youtube", "YouTube"),
        ("the github website", "GitHub"),
        ("git hub", "GitHub"),
        ("gmail please", "Gmail"),
        ("my gmail", None),
        ("youtube music", None),
    ],
)
def test_site_lookup(words: str, name: str | None) -> None:
    site = SiteCatalog(config.SITES).find(words)
    assert (site.name if site else None) == name


@pytest.mark.parametrize("url", ["http://example.com/", "javascript:alert(1)", "file:///C:/"])
def test_non_https_sites_are_refused(url: str) -> None:
    with pytest.raises(ValueError, match="https"):
        SiteCatalog([Site("Bad", ("bad",), url)])
