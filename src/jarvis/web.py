"""Whitelisted websites and search URLs (no OS code).

URLs come only from fixed https templates and site entries in config.py; speech
only ever fills in the URL-encoded query, so it can't change the scheme, host
or path.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import quote_plus, urlsplit

from jarvis.apps import strip_filler

QUERY_PLACEHOLDER = "{query}"

# Characters Whisper (or people) put around a query: "Search for cats." ->
# "cats". Not "+", "-", "#" or "&", which can be part of a query ("C++").
_EDGE_CHARACTERS = " \t\r\n.,!?;:…\"'“”‘’"
_TRAILING_POLITE = ("please", "thanks", "thank you")


@dataclass(frozen=True)
class Site:
    name: str  # used in spoken replies
    aliases: tuple[str, ...]  # "open <alias>"
    url: str  # fixed https URL


class SiteCatalog:
    def __init__(self, sites: Iterable[Site]) -> None:
        self.sites: tuple[Site, ...] = tuple(sites)
        self._by_alias: dict[str, Site] = {}
        for site in self.sites:
            if urlsplit(site.url).scheme != "https":
                raise ValueError(f"{site.name}: only https URLs are allowed")
            for alias in site.aliases:
                if alias in self._by_alias:
                    raise ValueError(f"alias {alias!r} is used by two sites")
                self._by_alias[alias] = site

    def find(self, words: str) -> Site | None:
        """Find the site named by normalized words ("the github website")."""
        return self._by_alias.get(strip_filler(words))


def clean_query(text: str) -> str:
    """Trim what surrounds a spoken query, keeping its own characters and case."""
    query = " ".join(text.split()).strip(_EDGE_CHARACTERS)
    lowered = query.lower()
    for polite in _TRAILING_POLITE:
        if lowered == polite:
            return ""
        if lowered.endswith(" " + polite):
            return query[: -len(polite)].strip(_EDGE_CHARACTERS)
    return query


_ON_YOUTUBE = re.compile(r"(?:^|(?<=\s))on\s+(?:youtube|you\s+tube|yt|y\.\s*t)\s*$", re.IGNORECASE)


def split_on_youtube(query: str) -> str | None:
    """For "cats on YouTube" / "cats on YT", the query without that ending
    ("cats", possibly ""); None if the query doesn't end that way."""
    found = _ON_YOUTUBE.search(query)
    return None if found is None else query[: found.start()].strip(_EDGE_CHARACTERS)


_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")


def watch_url(template: str, video_id: str) -> str | None:
    """The https watch URL for a YouTube video id, or None if the id isn't exactly
    11 characters of [A-Za-z0-9_-] (so it can't add anything to the URL)."""
    if not _VIDEO_ID.fullmatch(video_id):
        return None
    if not template.startswith("https://") or template.count("{id}") != 1:
        raise ValueError(f"bad watch template {template!r}")
    return template.replace("{id}", video_id)


def is_valid_query(query: str, max_chars: int) -> bool:
    return 0 < len(query) <= max_chars and query.isprintable()


def search_url(template: str, query: str) -> str:
    """Fill a fixed https template with the URL-encoded query.

    quote_plus with safe="" encodes every reserved character, including
    & ? # / = + and %, and non-ASCII text as UTF-8.
    """
    if not template.startswith("https://") or template.count(QUERY_PLACEHOLDER) != 1:
        raise ValueError(f"bad search template {template!r}")
    url = template.replace(QUERY_PLACEHOLDER, quote_plus(query, safe=""))
    # Defense in depth: the query must not have changed where the URL points.
    built, fixed = urlsplit(url), urlsplit(template)
    if (built.scheme, built.netloc, built.path) != (fixed.scheme, fixed.netloc, fixed.path):
        raise ValueError("query changed the URL's destination")
    return url
