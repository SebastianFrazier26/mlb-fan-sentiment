"""VADER with a baseball lexicon overlay.

Every overlay entry (word, phrase or emoji) is rewritten in the text to a unique placeholder
word whose valence is injected into the VADER lexicon. One mechanism covers all three kinds,
and VADER's own negation, booster and punctuation rules still apply around the placeholder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DEFAULT_LEXICON = Path(__file__).with_name("baseball_lexicon.json")


def load_entries(path: Path = DEFAULT_LEXICON) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        return {k.lower(): float(v) for k, v in json.load(f)["entries"].items()}


class LexiconVader:
    def __init__(self, entries: dict[str, float] | None = None) -> None:
        self.entries = (
            load_entries() if entries is None else {k.lower(): v for k, v in entries.items()}
        )
        self.analyzer = SentimentIntensityAnalyzer()
        keys = sorted(self.entries, key=lambda k: (-len(k), k))
        self._token = {k: f"zzlex{i}" for i, k in enumerate(keys)}
        for k, tok in self._token.items():
            self.analyzer.lexicon[tok] = self.entries[k]
        parts = [rf"(?<!\w){re.escape(k)}(?!\w)" if k[0].isalnum() else re.escape(k) for k in keys]
        self._pattern = re.compile("|".join(parts), re.IGNORECASE) if parts else None

    def rewrite(self, text: str) -> str:
        if not self._pattern:
            return text
        text = text.replace("’", "'")
        return self._pattern.sub(lambda m: f" {self._token[m.group(0).lower()]} ", text)

    def compound(self, text: str) -> float:
        return self.analyzer.polarity_scores(self.rewrite(text))["compound"]
