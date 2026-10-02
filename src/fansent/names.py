"""Player-name matching for free text.

A mention is a full-name match, or a last-name match when that last name is
unique across all tracked players (so 'Suzuki' works but a shared last name
would only match on full name).
"""

from __future__ import annotations

import re
from collections import Counter


def build_matchers(players: list[dict]) -> list[tuple[dict, re.Pattern]]:
    last_counts = Counter(p["last_name"].lower() for p in players)
    matchers = []
    for p in players:
        parts = [re.escape(p["full_name"])]
        last = p["last_name"]
        if last_counts[last.lower()] == 1 and len(last) >= 4:
            parts.append(re.escape(last))
        pattern = re.compile(rf"\b(?:{'|'.join(parts)})\b", re.IGNORECASE)
        matchers.append((p, pattern))
    return matchers


def find_mentions(text: str, matchers: list[tuple[dict, re.Pattern]]) -> list[dict]:
    return [p for p, pat in matchers if pat.search(text)]
