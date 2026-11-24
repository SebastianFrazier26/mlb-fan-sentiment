"""MLB regular-season schedule -> schedule.csv (first-pitch times for fan-day alignment).

One Stats API request per season, cached under raw/mlb/schedule_<season>.json. Only games that
were played (codedGameState F = final) are kept; a postponed game shows up again on its
makeup date with its own gamePk.
"""

from __future__ import annotations

from pathlib import Path

from .util import MLB_RAW_DIR, cached_get_json, write_csv

API = "https://statsapi.mlb.com/api/v1/schedule"
SCHEDULE_FILE = "schedule.csv"
FIELDS = [
    "game_pk",
    "official_date",
    "game_utc",
    "home_id",
    "away_id",
    "double_header",
    "game_number",
]


def parse(data: dict) -> list[dict]:
    rows = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("codedGameState") != "F":
                continue
            rows.append(
                {
                    "game_pk": g["gamePk"],
                    "official_date": g["officialDate"],
                    "game_utc": g["gameDate"],
                    "home_id": g["teams"]["home"]["team"]["id"],
                    "away_id": g["teams"]["away"]["team"]["id"],
                    "double_header": g.get("doubleHeader", "N"),
                    "game_number": g.get("gameNumber", 1),
                }
            )
    rows.sort(key=lambda r: (r["game_utc"], r["game_pk"]))
    return rows


def run(season: int, data_dir: Path) -> int:
    cache = data_dir / MLB_RAW_DIR / f"schedule_{season}.json"
    data = cached_get_json(API, {"sportId": 1, "season": season, "gameType": "R"}, cache)
    rows = parse(data)
    n = write_csv(data_dir / SCHEDULE_FILE, rows, FIELDS)
    print(f"Wrote {n} final games to {data_dir / SCHEDULE_FILE}")
    return n
