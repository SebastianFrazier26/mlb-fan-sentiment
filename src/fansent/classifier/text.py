"""Text prep: accent fold, target/other player masking, emoji-preserving tokenizer."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from fansent.names import NAME_SUFFIXES, fold

TGT = "__TGT__"
OTH = "__OTH__"

# Skin-tone modifiers (1F3FB-1F3FF) are excluded so they do not become tokens of their own.
_EMOJI = "[\U0001f000-\U0001f3fa\U0001f400-\U0001faff☀-➿⭐⭕]"
_TOKEN = re.compile(rf"__(?:TGT|OTH)__|[^\W_]+(?:'[^\W_]+)*|[!?]|{_EMOJI}")


def tokenize(masked: str) -> list[str]:
    """Lowercased word/emoji/!/? tokens; placeholders keep their case."""
    text = masked.replace("’", "'")
    return [t if t in (TGT, OTH) else t.lower() for t in _TOKEN.findall(text)]


def _phrase(name: str) -> str:
    return r"\s+".join(re.escape(w) for w in fold(name).split())


def _alternation(names: Iterable[str]) -> re.Pattern | None:
    names = sorted({n for n in names if n.strip()}, key=lambda n: (-len(n), n))
    if not names:
        return None
    body = "|".join(_phrase(n) for n in names)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


def _key(s: str) -> str:
    return " ".join(fold(s).lower().split())


def _surname(full_name: str) -> str:
    parts = [p for p in full_name.split() if p.lower().strip(".") not in NAME_SUFFIXES]
    return parts[-1] if parts else ""


class Masker:
    """Replaces the target's name/aliases with __TGT__ and other rostered players with __OTH__.

    The target is masked first and with a looser rule than the roster (any surname, even a shared
    one) because the row tells us who it is about. The roster pass only credits full names and
    surnames that are unique and >=4 chars, mirroring ``names.build_matchers``.
    """

    def __init__(self, players: Iterable[dict] = ()) -> None:
        players = list(players)
        self._last_by_id = {str(p["player_id"]): p.get("last_name", "") for p in players}
        counts = Counter(_key(p.get("last_name", "")) for p in players)
        self._owner: dict[str, str] = {}
        for p in players:
            pid = str(p["player_id"])
            self._owner[_key(p["full_name"])] = pid
            last = p.get("last_name", "")
            if len(last) >= 4 and counts[_key(last)] == 1:
                self._owner.setdefault(_key(last), pid)
        self._others = _alternation(self._owner)
        self._target_cache: dict[tuple, re.Pattern | None] = {}

    def _target_pattern(self, name: str, player_id: str, aliases: tuple[str, ...]):
        key = (name, player_id, aliases)
        if key not in self._target_cache:
            names = {name, *aliases}
            for last in (_surname(name), self._last_by_id.get(player_id, "")):
                if len(last) >= 3:
                    names.add(last)
            self._target_cache[key] = _alternation(names)
        return self._target_cache[key]

    def mask(
        self, text: str, *, name: str, player_id: str = "", aliases: Iterable[str] = ()
    ) -> str:
        out = fold(text)
        tgt = self._target_pattern(name, str(player_id), tuple(aliases))
        if tgt:
            out = tgt.sub(TGT, out)
        if self._others:
            out = self._others.sub(lambda m: OTH, out)
        return out
