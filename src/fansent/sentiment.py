"""Score collected Reddit texts with VADER, writing sentiment.csv."""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .util import SENTIMENT_CSV, TEXTS_CSV, read_csv, write_csv

FIELDS = ["text_id", "source", "subreddit", "date", "player_id", "player_name", "compound"]


def run() -> None:
    texts = read_csv(TEXTS_CSV)
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
    n = write_csv(SENTIMENT_CSV, rows, FIELDS)
    print(f"Scored {n} texts -> {SENTIMENT_CSV}")
