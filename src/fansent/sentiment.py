"""Score collected Reddit texts into sentiment.csv (compound column = score in [-1, 1])."""

from __future__ import annotations

from pathlib import Path

from .classifier.data import Example
from .classifier.evaluate import load_players
from .classifier.scorers import build_scorer
from .util import PLAYERS_CSV, SENTIMENT_CSV, TEXTS_CSV, read_csv, write_csv

FIELDS = ["text_id", "source", "subreddit", "date", "player_id", "player_name", "compound"]


def run(scorer: str = "vader", model: Path | None = None) -> None:
    texts = read_csv(TEXTS_CSV)
    s = build_scorer(scorer, model, load_players(PLAYERS_CSV))
    examples = [
        Example(text=t["text"], player=t["player_name"], player_id=str(t["player_id"]))
        for t in texts
    ]
    scores = s.score(examples)
    rows = [
        {
            "text_id": t["text_id"],
            "source": t["source"],
            "subreddit": t["subreddit"],
            "date": t["date"],
            "player_id": t["player_id"],
            "player_name": t["player_name"],
            "compound": round(float(c), 4),
        }
        for t, c in zip(texts, scores, strict=True)
    ]
    n = write_csv(SENTIMENT_CSV, rows, FIELDS)
    print(f"Scored {n} texts ({scorer}) -> {SENTIMENT_CSV}")
