"""Shared helpers: paths, dates, CSV I/O."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import time
from pathlib import Path

import requests

ENV_VAR = "FANSENT_DATA_DIR"
# Relative to the CWD, unlike fansent-collect (anchored to its checkout); kept for compatibility.
DEFAULT_DATA_DIR = Path("data")

PLAYERS_FILE = "players.csv"
PLAYER_TEAMS_FILE = "player_teams.csv"
TEAMS_FILE = "teams.csv"
MLB_RAW_DIR = Path("raw") / "mlb"
STATS_FILE = "stats.csv"
STATS_COUNTS_FILE = "stats_counts.csv"
OAA_FILE = "oaa_monthly.csv"
TEXTS_FILE = "texts.csv"
REDDIT_ATTRIBUTION_FILE = "reddit_attribution.csv"
PARENT_TEXTS_FILE = "parent_texts.csv"
SENTIMENT_FILE = "sentiment.csv"
RELEVANCE_FILE = "relevance.csv"
NEWS_TONE_FILE = "news_tone.csv"
MODELS_DIR = "models"
MODEL_FILE = "clf.npz"
RELEVANCE_MODEL_FILE = "relevance.npz"


def resolve_data_dir(cli_value: str | None = None) -> Path:
    """--data-dir > $FANSENT_DATA_DIR > ./data. Empty strings count as unset."""
    for value in (cli_value, os.environ.get(ENV_VAR)):
        if value:
            return Path(value).expanduser()
    return DEFAULT_DATA_DIR


def week_start(d: dt.date) -> dt.date:
    """Monday of the ISO week containing d."""
    return d - dt.timedelta(days=d.weekday())


def parse_ip(ip: str) -> float:
    """MLB innings-pitched strings use .1/.2 for thirds ('5.2' == 5 2/3)."""
    if not ip:
        return 0.0
    whole, _, frac = str(ip).partition(".")
    return int(whole or 0) + int(frac or 0) / 3.0


def ip_to_outs(ip: str) -> int:
    """Exact outs from an innings-pitched string ('5.2' -> 17); avoids float thirds."""
    whole, _, frac = str(ip or "0").partition(".")
    return int(whole or 0) * 3 + int(frac or 0)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str], append: bool = False) -> int:
    """Write (or append) dict rows; returns number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    mode = "a" if append and exists else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w" or not exists:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found — run the earlier pipeline step first.")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cached_get_json(
    url: str, params: dict | None, cache_path: Path | None, pause: float = 0.0
) -> dict:
    """GET JSON, reading/writing cache_path when given. Only network hits are paced."""
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
    if pause:
        time.sleep(pause)
    return data
