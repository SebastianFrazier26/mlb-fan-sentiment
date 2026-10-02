"""Pull per-game logs from the MLB Stats API and aggregate to weekly performance.

Performance metric: OPS for hitters, ERA for pitchers. stats.csv carries a
`higher_is_better` flag so analysis can orient both the same way.
"""

from __future__ import annotations

import datetime as dt
import time
from collections import defaultdict

import requests

from .util import PLAYERS_CSV, STATS_CSV, parse_ip, read_csv, week_start, write_csv

API = "https://statsapi.mlb.com/api/v1"

FIELDS = [
    "player_id",
    "full_name",
    "team",
    "season",
    "week_start",
    "games",
    "metric",
    "value",
    "higher_is_better",
    "sample",  # PA for hitters, IP for pitchers — analysis filters tiny samples
]


def fetch_game_log(player_id: int, season: int, group: str) -> list[dict]:
    resp = requests.get(
        f"{API}/people/{player_id}/stats",
        params={"stats": "gameLog", "season": season, "group": group},
        timeout=30,
    )
    resp.raise_for_status()
    for block in resp.json().get("stats", []):
        return block.get("splits", [])
    return []


def weekly_hitting(splits: list[dict]) -> list[dict]:
    weeks: dict[dt.date, defaultdict] = {}
    for s in splits:
        d = dt.date.fromisoformat(s["date"])
        w = weeks.setdefault(week_start(d), defaultdict(float))
        st = s["stat"]
        w["games"] += 1
        for k in (
            "atBats",
            "hits",
            "baseOnBalls",
            "hitByPitch",
            "sacFlies",
            "totalBases",
            "plateAppearances",
        ):
            w[k] += int(st.get(k, 0) or 0)
    out = []
    for wk, w in sorted(weeks.items()):
        ab, h = w["atBats"], w["hits"]
        bb, hbp, sf, tb = w["baseOnBalls"], w["hitByPitch"], w["sacFlies"], w["totalBases"]
        obp_den = ab + bb + hbp + sf
        if ab == 0 or obp_den == 0:
            continue
        ops = (h + bb + hbp) / obp_den + tb / ab
        out.append(
            {
                "week_start": wk.isoformat(),
                "games": int(w["games"]),
                "metric": "OPS",
                "value": round(ops, 4),
                "higher_is_better": True,
                "sample": int(w["plateAppearances"]),
            }
        )
    return out


def weekly_pitching(splits: list[dict]) -> list[dict]:
    weeks: dict[dt.date, defaultdict] = {}
    for s in splits:
        d = dt.date.fromisoformat(s["date"])
        w = weeks.setdefault(week_start(d), defaultdict(float))
        st = s["stat"]
        w["games"] += 1
        w["ip"] += parse_ip(st.get("inningsPitched", "0"))
        w["er"] += int(st.get("earnedRuns", 0) or 0)
    out = []
    for wk, w in sorted(weeks.items()):
        if w["ip"] <= 0:
            continue
        era = 9.0 * w["er"] / w["ip"]
        out.append(
            {
                "week_start": wk.isoformat(),
                "games": int(w["games"]),
                "metric": "ERA",
                "value": round(era, 4),
                "higher_is_better": False,
                "sample": round(w["ip"], 1),
            }
        )
    return out


def run(season: int, limit: int | None = None, pause: float = 0.3) -> None:
    players = read_csv(PLAYERS_CSV)
    if limit:
        players = players[:limit]
    rows = []
    for i, p in enumerate(players, 1):
        pitcher = p["is_pitcher"] == "True"
        group = "pitching" if pitcher else "hitting"
        splits = fetch_game_log(int(p["player_id"]), season, group)
        weekly = weekly_pitching(splits) if pitcher else weekly_hitting(splits)
        for w in weekly:
            rows.append(
                {
                    "player_id": p["player_id"],
                    "full_name": p["full_name"],
                    "team": p["team"],
                    "season": season,
                    **w,
                }
            )
        print(f"[{i}/{len(players)}] {p['full_name']}: {len(weekly)} weeks")
        time.sleep(pause)
    n = write_csv(STATS_CSV, rows, FIELDS)
    print(f"Wrote {n} player-weeks to {STATS_CSV}")
