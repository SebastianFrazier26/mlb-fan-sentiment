"""Score collected Reddit texts with VADER, writing sentiment.csv."""

from __future__ import annotations

from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .util import DEFAULT_DATA_DIR, SENTIMENT_FILE, TEXTS_FILE, read_csv, write_csv

FIELDS = ["text_id", "source", "subreddit", "date", "player_id", "player_name", "compound"]


def run(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    texts = read_csv(data_dir / TEXTS_FILE)
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for t in texts:
        score = analyzer.polarity_scores(t["text"])
        rows.append(
            {
                "text_id": t["text_id"],
                "source": t["source"],
                "subreddit": t["subreddit"],
                "date": t["date"],
                "player_id": t["player_id"],
                "player_name": t["player_name"],
                "compound": round(score["compound"], 4),
            }
        )
    out = data_dir / SENTIMENT_FILE
    n = write_csv(out, rows, FIELDS)
    print(f"Scored {n} texts -> {out}")
