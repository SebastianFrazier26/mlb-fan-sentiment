"""Shared helpers: paths, dates, CSV I/O."""

from __future__ import annotations

import csv
import datetime as dt
import os
from pathlib import Path

ENV_VAR = "FANSENT_DATA_DIR"
# Relative to the CWD, unlike fansent-collect (anchored to its checkout); kept for compatibility.
DEFAULT_DATA_DIR = Path("data")

PLAYERS_FILE = "players.csv"
STATS_FILE = "stats.csv"
STATS_COUNTS_FILE = "stats_counts.csv"
OAA_FILE = "oaa_monthly.csv"
TEXTS_FILE = "texts.csv"
SENTIMENT_FILE = "sentiment.csv"
NEWS_TONE_FILE = "news_tone.csv"


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
