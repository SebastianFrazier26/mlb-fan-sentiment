"""Score collected Reddit texts into sentiment.csv (compound column = score in [-1, 1])."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from .classifier.data import Example
from .classifier.evaluate import load_players
from .classifier.scorers import build_scorer
from .util import (
    DEFAULT_DATA_DIR,
    MODEL_FILE,
    MODELS_DIR,
    PARENT_TEXTS_FILE,
    PLAYERS_FILE,
    REDDIT_ATTRIBUTION_FILE,
    SENTIMENT_FILE,
    TEXTS_FILE,
    read_csv,
    write_csv,
)

# Matches the parent_text length of the labeled training samples; the model must see the same
# span at scoring time. Truncation precedes masking, as it did when the samples were built.
PARENT_MAX_CHARS = 300
REPLY_INHERITED = "reply_inherited"

FIELDS = ["text_id", "source", "subreddit", "date", "player_id", "player_name", "compound"]


def _read_optional(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parent_texts_for(texts: list[dict], data_dir: Path) -> tuple[list[str], int]:
    """Parent text per texts.csv row ('' for explicit rows and for replies whose parent is
    unavailable) plus the number of reply rows lacking parent text."""
    reply_parent = {
        (a["text_id"], str(a["player_id"])): a.get("parent_id") or ""
        for a in _read_optional(data_dir / REDDIT_ATTRIBUTION_FILE)
        if a.get("attribution") == REPLY_INHERITED
    }
    lookup = {
        r["parent_id"]: (r.get("text") or "").strip()
        for r in _read_optional(data_dir / PARENT_TEXTS_FILE)
    }
    out, missing = [], 0
    for t in texts:
        key = (t["text_id"], str(t["player_id"]))
        if key not in reply_parent:
            out.append("")
            continue
        parent = lookup.get(reply_parent[key], "")[:PARENT_MAX_CHARS].strip()
        missing += not parent
        out.append(parent)
    return out, missing


def run(
    scorer: str = "vader", model: Path | None = None, data_dir: Path = DEFAULT_DATA_DIR
) -> None:
    texts = read_csv(data_dir / TEXTS_FILE)
    if scorer == "clf" and model is None:
        model = data_dir / MODELS_DIR / MODEL_FILE
    s = build_scorer(scorer, model, load_players(data_dir / PLAYERS_FILE))
    parents = [""] * len(texts)
    if scorer == "clf" and s.model.config.features.parent:
        parents, missing = parent_texts_for(texts, data_dir)
        if missing:
            print(
                f"warning: {missing} reply_inherited rows have no parent text "
                f"(need {PARENT_TEXTS_FILE} with their parent_id); scored as non-replies, "
                "which this --parent model was not trained on.",
                file=sys.stderr,
            )
    examples = [
        Example(
            text=t["text"], player=t["player_name"], player_id=str(t["player_id"]), parent_text=p
        )
        for t, p in zip(texts, parents, strict=True)
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
    out = data_dir / SENTIMENT_FILE
    n = write_csv(out, rows, FIELDS)
    print(f"Scored {n} texts ({scorer}) -> {out}")
