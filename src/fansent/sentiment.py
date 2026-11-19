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
    RELEVANCE_FILE,
    RELEVANCE_MODEL_FILE,
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
RELEVANCE_FIELDS = ["text_id", "player_id", "p_relevant", "kept"]


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


def relevance_filter(
    texts: list[dict], model_path: Path, players: list[dict], data_dir: Path
) -> list[dict]:
    """Drop rows the relevance model calls 'na'; write the audit sidecar; return kept rows."""
    from .classifier.relevance import RelevanceClassifier

    model = RelevanceClassifier.load(model_path, players)
    parents, _ = parent_texts_for(texts, data_dir)
    examples = [
        Example(
            text=t["text"],
            player=t["player_name"],
            player_id=str(t["player_id"]),
            parent_text=p,
            subreddit=t.get("subreddit") or "",
            date=t.get("date") or "",
        )
        for t, p in zip(texts, parents, strict=True)
    ]
    p_na = model.p_na(examples)
    drop = p_na >= model.config.threshold
    write_csv(
        data_dir / RELEVANCE_FILE,
        [
            {
                "text_id": t["text_id"],
                "player_id": t["player_id"],
                "p_relevant": round(1.0 - float(p), 4),
                "kept": int(not d),
            }
            for t, p, d in zip(texts, p_na, drop, strict=True)
        ],
        RELEVANCE_FIELDS,
    )
    kept = {s: [0, 0] for s in sorted({t["subreddit"] for t in texts})}
    for t, d in zip(texts, drop, strict=True):
        kept[t["subreddit"]][int(d)] += 1
    print(f"Relevance filter ({model_path.name}, drop P(na) >= {model.config.threshold:.3f}):")
    for sub, (k, d) in kept.items():
        print(f"  {sub or '(none)':<12} kept {k:>7}  dropped {d:>7}  ({d / max(1, k + d):.1%})")
    return [t for t, d in zip(texts, drop, strict=True) if not d]


def _attach_fasttext(model, data_dir: Path) -> None:
    """The model file only caches vectors for tokens seen in training; without the .bin every
    other token silently gets no embedding, unlike at clf-eval time."""
    from .classifier.embeddings import DEFAULT_BIN, EMBEDDINGS_DIR, FastTextVectors

    bin_path = data_dir / EMBEDDINGS_DIR / DEFAULT_BIN
    if not bin_path.exists():
        print(f"warning: {bin_path} not found; unseen tokens get no embedding", file=sys.stderr)
        return
    model.vectorizer.vectors.provider = FastTextVectors(bin_path)
    print(f"Using {bin_path.name} for tokens not cached in the model")


def run(
    scorer: str = "vader",
    model: Path | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    relevance_model: Path | None = None,
    use_relevance: bool = True,
) -> None:
    texts = read_csv(data_dir / TEXTS_FILE)
    players = load_players(data_dir / PLAYERS_FILE)
    if relevance_model is None:
        relevance_model = data_dir / MODELS_DIR / RELEVANCE_MODEL_FILE
        use_relevance = use_relevance and relevance_model.exists()
    if use_relevance:
        texts = relevance_filter(texts, relevance_model, players, data_dir)
    if scorer == "clf" and model is None:
        model = data_dir / MODELS_DIR / MODEL_FILE
    s = build_scorer(scorer, model, players)
    if scorer == "clf" and s.model.config.features.emb:
        _attach_fasttext(s.model, data_dir)
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
