"""MLB team names and nicknames, used by the relevance model to tell 'this comment is about
our roster team' from 'this comment is about some other club'.

Shared city names (New York, Chicago, Los Angeles) are left out: they cannot separate Mets
from Yankees or Cubs from White Sox. 'sox' alone is Boston's only once 'white sox' is
removed, see ``_prep_text``.
"""

from __future__ import annotations

import re

TEAM_TERMS: dict[str, tuple[str, ...]] = {
    "Arizona Diamondbacks": ("diamondbacks", "dbacks", "d-backs", "arizona"),
    "Atlanta Braves": ("braves", "atlanta"),
    "Baltimore Orioles": ("orioles", "baltimore"),
    "Boston Red Sox": ("red sox", "redsox", "bosox", "sox", "boston", "fenway"),
    "Chicago Cubs": ("cubs", "cubbies", "wrigley"),
    "Chicago White Sox": ("whitesox", "chisox"),
    "Cincinnati Reds": ("reds", "cincinnati"),
    "Cleveland Guardians": ("guardians", "cleveland"),
    "Colorado Rockies": ("rockies", "colorado", "coors"),
    "Detroit Tigers": ("tigers", "detroit"),
    "Houston Astros": ("astros", "houston"),
    "Kansas City Royals": ("royals", "kansas city"),
    "Los Angeles Angels": ("angels", "halos"),
    "Los Angeles Dodgers": ("dodgers",),
    "Miami Marlins": ("marlins", "miami"),
    "Milwaukee Brewers": ("brewers", "milwaukee"),
    "Minnesota Twins": ("twins", "minnesota"),
    "New York Mets": ("mets",),
    "New York Yankees": ("yankees", "yanks"),
    "Oakland Athletics": ("athletics", "oakland", "sacramento"),
    "Philadelphia Phillies": ("phillies", "phils", "philadelphia"),
    "Pittsburgh Pirates": ("pirates", "pittsburgh"),
    "San Diego Padres": ("padres", "san diego"),
    "San Francisco Giants": ("giants", "san francisco"),
    "Seattle Mariners": ("mariners", "seattle"),
    "St. Louis Cardinals": ("cardinals", "cards", "st louis", "st. louis"),
    "Tampa Bay Rays": ("rays", "tampa"),
    "Texas Rangers": ("rangers", "texas"),
    "Toronto Blue Jays": ("blue jays", "bluejays", "jays", "toronto"),
    "Washington Nationals": ("nationals", "nats", "washington"),
}


def _compile(terms: tuple[str, ...]) -> re.Pattern:
    return re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(t) for t in terms) + r")(?![a-z])")


_PATTERNS = {team: _compile(terms) for team, terms in TEAM_TERMS.items()}


def _prep_text(text: str) -> str:
    return text.lower().replace("white sox", "whitesox")


def teams_in(text: str) -> set[str]:
    """Canonical names of the teams mentioned in ``text``."""
    t = _prep_text(text)
    return {team for team, pat in _PATTERNS.items() if pat.search(t)}


def canonical_team(value: str) -> str:
    """Map a team full name (exact or a suffix like 'Athletics') to its ``TEAM_TERMS`` key.

    Unknown values come back unchanged, so an abbreviation never matches a mention.
    """
    value = value.strip()
    if value in TEAM_TERMS:
        return value
    for team in TEAM_TERMS:
        if value and team.endswith(" " + value):
            return team
    return value
