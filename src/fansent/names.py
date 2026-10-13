"""Player-name matching for free text.

A mention is a full-name match, or a last-name match when that last name is
unique across all tracked players (so 'Suzuki' works but a shared last name
would only match on full name).

Matching is accent-insensitive (text and names are both folded), and a bare
name that is also a common first name is ignored when it is immediately followed
by a capitalized word that is not part of the player's own name ('Anthony Volpe'
is not Roman Anthony). An ALL-CAPS following word is shouting, not a surname
('NICO WTF'), so it blocks only when it is a tracked surname ('ANTHONY VOLPE').
``name_pattern`` exposes both behaviours for alias layers.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache

from fansent.first_names import COMMON_FIRST_NAMES

NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})

# Horizontal whitespace only: a paragraph break ends the name.
_NEXT_WORD = re.compile(r"[^\S\n]+([^\W\d_][\w'’-]*)")


@lru_cache(maxsize=4096)
def _fold_char(c: str) -> str:
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFKD", c) if not unicodedata.combining(ch)
    )
    # Length-preserving: expanding folds (ligatures, ß) keep the original char so
    # match offsets in the folded text index the original text.
    return stripped if len(stripped) == 1 else c


def fold(s: str) -> str:
    """Strip diacritics from precomposed characters, one-for-one."""
    if s.isascii():
        return s
    return "".join(map(_fold_char, s))


def _tokens(name: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[\w'’-]+", fold(name))}


def _is_blocked(text: str, end: int, own: frozenset[str], surnames: frozenset[str]) -> bool:
    m = _NEXT_WORD.match(text, end)
    if not m:
        return False
    word = m.group(1)
    if not word[0].isupper() or word.lower() in own:
        return False
    # Case alone cannot separate 'NICO WTF' from 'ANTHONY VOLPE'; a roster lookup can.
    if len(word) >= 2 and word.isupper():
        return word.lower() in surnames
    return True


class NamePattern:
    """Quacks like a compiled pattern (``search``) for ``find_mentions``.

    ``search`` takes raw text and folds it itself, so callers can combine it with
    other patterns over the same unfolded text. Returned matches index the original
    text (folding is length-preserving).
    """

    def __init__(
        self, regex: re.Pattern, own: frozenset[str], guard: bool, surnames: frozenset[str]
    ) -> None:
        self.regex = regex
        self.own = own
        self.guard = guard
        self.surnames = surnames

    def search(self, text: str) -> re.Match | None:
        folded = fold(text)
        for m in self.regex.finditer(folded):
            if not self.guard or not _is_blocked(folded, m.end(), self.own, self.surnames):
                return m
        return None


def name_pattern(
    name: str,
    *,
    full_name: str,
    guard: bool = False,
    case_sensitive: bool = False,
    surnames: frozenset[str] = frozenset(),
) -> NamePattern:
    """Whole-word, accent-insensitive pattern for ``name`` (a full name, surname, or alias).

    ``guard=True`` drops hits followed by a capitalized word that is not one of
    ``full_name``'s own tokens (or a Jr/Sr/II/III/IV suffix): use it for bare first
    names and for surnames that double as first names. A following ALL-CAPS word (2+
    letters) blocks only if its lowercase fold is in ``surnames``. Known limitation: a name at the
    end of a sentence followed by a capitalized sentence start ('Anthony Then homered')
    is dropped too.
    """
    body = r"\s+".join(re.escape(w) for w in fold(name).split())
    regex = re.compile(rf"\b{body}\b", 0 if case_sensitive else re.IGNORECASE)
    return NamePattern(regex, frozenset(_tokens(full_name) | NAME_SUFFIXES), guard, surnames)


class _FullOrLast:
    def __init__(self, full: NamePattern, last: NamePattern | None) -> None:
        self.parts = [full] if last is None else [full, last]

    def search(self, text: str) -> re.Match | None:
        for part in self.parts:
            if m := part.search(text):
                return m
        return None


def build_matchers(players: list[dict]) -> list[tuple[dict, _FullOrLast]]:
    last_counts = Counter(fold(p["last_name"]).lower() for p in players)
    first_names = set(COMMON_FIRST_NAMES)
    for p in players:
        first_names |= _tokens(p["full_name"]) - _tokens(p["last_name"])
    surnames = frozenset(fold(p["last_name"]).lower() for p in players)
    matchers = []
    for p in players:
        full = name_pattern(p["full_name"], full_name=p["full_name"])
        last = p["last_name"]
        last_pat = None
        if last_counts[fold(last).lower()] == 1 and len(last) >= 4:
            last_pat = name_pattern(
                last,
                full_name=p["full_name"],
                guard=fold(last).lower() in first_names,
                surnames=surnames,
            )
        matchers.append((p, _FullOrLast(full, last_pat)))
    return matchers


def find_mentions(text: str, matchers: list) -> list[dict]:
    return [p for p, pat in matchers if pat.search(text)]
