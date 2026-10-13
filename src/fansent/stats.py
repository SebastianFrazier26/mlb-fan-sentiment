"""Pull per-game logs from the MLB Stats API and aggregate to weekly (Mon-Sun) performance.

Metrics: OPS, AVG (hitters); ERA, WHIP (pitchers); FLD_PCT, ERRORS, CS_PCT (fielding).
Output is tidy long (one row per player-week-metric) with the week's value and the
season-to-date value through the end of that week. Raw weekly and season-to-date counts
go to a wide companion file. Rates are always recomputed from summed counts.
"""

from __future__ import annotations

import datetime as dt
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

from .util import (
    DEFAULT_DATA_DIR,
    PLAYERS_FILE,
    STATS_COUNTS_FILE,
    STATS_FILE,
    ip_to_outs,
    read_csv,
    week_start,
    write_csv,
)

API = "https://statsapi.mlb.com/api/v1"

FIELDS = [
    "player_id",
    "full_name",
    "team",
    "season",
    "week_start",
    "group",
    "metric",
    "value",
    "std_value",
    "weight",
    "std_weight",
    "higher_is_better",
]

COUNT_FIELDS = [
    "games",
    "pa",
    "ab",
    "h",
    "bb",
    "hbp",
    "sf",
    "tb",
    "outs",
    "er",
    "chances",
    "putouts",
    "assists",
    "errors",
    "sb",
    "cs",
]
COUNTS_FIELDS = (
    ["player_id", "full_name", "team", "season", "week_start", "group"]
    + COUNT_FIELDS
    + [f"std_{c}" for c in COUNT_FIELDS]
)

HITTING_KEYS = {
    "atBats": "ab",
    "hits": "h",
    "baseOnBalls": "bb",
    "hitByPitch": "hbp",
    "sacFlies": "sf",
    "totalBases": "tb",
    "plateAppearances": "pa",
}
FIELDING_KEYS = {
    "chances": "chances",
    "putOuts": "putouts",
    "assists": "assists",
    "errors": "errors",
    "stolenBases": "sb",
    "caughtStealing": "cs",
}

Counts = dict[str, int]


def _ratio(num: float, den: float) -> float | None:
    return num / den if den > 0 else None


def _ops(c: Counts) -> float | None:
    obp = _ratio(c["h"] + c["bb"] + c["hbp"], c["ab"] + c["bb"] + c["hbp"] + c["sf"])
    slg = _ratio(c["tb"], c["ab"])
    return None if obp is None or slg is None else obp + slg


@dataclass(frozen=True)
class Metric:
    name: str
    group: str
    higher_is_better: bool
    value: Callable[[Counts], float | None]
    weight: Callable[[Counts], float]


METRICS = [
    Metric("OPS", "hitting", True, _ops, lambda c: c["pa"]),
    Metric("AVG", "hitting", True, lambda c: _ratio(c["h"], c["ab"]), lambda c: c["ab"]),
    Metric(
        "ERA", "pitching", False, lambda c: _ratio(27 * c["er"], c["outs"]), lambda c: c["outs"] / 3
    ),
    Metric(
        "WHIP",
        "pitching",
        False,
        lambda c: _ratio(3 * (c["h"] + c["bb"]), c["outs"]),
        lambda c: c["outs"] / 3,
    ),
    Metric(
        "FLD_PCT",
        "fielding",
        True,
        lambda c: _ratio(c["putouts"] + c["assists"], c["chances"]),
        lambda c: c["chances"],
    ),
    Metric("ERRORS", "fielding", False, lambda c: float(c["errors"]), lambda c: c["chances"]),
    # Only catchers see SB/CS; everyone else has zero attempts and no row.
    Metric(
        "CS_PCT",
        "fielding",
        True,
        lambda c: _ratio(c["cs"], c["sb"] + c["cs"]),
        lambda c: c["sb"] + c["cs"],
    ),
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


def _int(st: dict, key: str) -> int:
    return int(st.get(key, 0) or 0)


def weekly_counts(splits: list[dict], group: str) -> dict[dt.date, Counts]:
    """Sum raw counting stats per Monday-keyed week. Non-regular-season splits are dropped.

    Fielding logs have one split per game *per position played*, so counts are summed
    across positions (a player's total defensive workload that week) and `games` counts
    distinct games, not splits.
    """
    weeks: dict[dt.date, Counts] = {}
    game_ids: dict[dt.date, set] = defaultdict(set)
    for s in splits:
        if s.get("gameType", "R") != "R":
            continue
        d = dt.date.fromisoformat(s["date"])
        wk = week_start(d)
        w = weeks.setdefault(wk, dict.fromkeys(COUNT_FIELDS, 0))
        st = s["stat"]
        game_ids[wk].add((s.get("game") or {}).get("gamePk") or (s["date"], len(game_ids[wk])))
        if group == "hitting":
            for k, name in HITTING_KEYS.items():
                w[name] += _int(st, k)
        elif group == "pitching":
            w["outs"] += ip_to_outs(st.get("inningsPitched", "0"))
            w["er"] += _int(st, "earnedRuns")
            w["h"] += _int(st, "hits")
            w["bb"] += _int(st, "baseOnBalls")
        elif group == "fielding":
            for k, name in FIELDING_KEYS.items():
                w[name] += _int(st, k)
        else:
            raise ValueError(f"unknown group {group!r}")
    for wk, w in weeks.items():
        w["games"] = len(game_ids[wk])
    return weeks


def cumulative(weeks: dict[dt.date, Counts]) -> dict[dt.date, Counts]:
    """Season-to-date counts through the END of each week: only weeks <= t contribute."""
    running = dict.fromkeys(COUNT_FIELDS, 0)
    out: dict[dt.date, Counts] = {}
    for wk in sorted(weeks):
        for k in COUNT_FIELDS:
            running[k] += weeks[wk][k]
        out[wk] = dict(running)
    return out


def weekly_metrics(splits: list[dict], group: str) -> tuple[list[dict], list[dict]]:
    """Returns (long metric rows, wide count rows) for one player and group.

    A metric row exists only for weeks where that metric's weekly denominator is positive.
    """
    weeks = weekly_counts(splits, group)
    std = cumulative(weeks)
    metrics = [m for m in METRICS if m.group == group]
    long_rows, count_rows = [], []
    for wk in sorted(weeks):
        c, sc = weeks[wk], std[wk]
        count_rows.append(
            {
                "week_start": wk.isoformat(),
                "group": group,
                **c,
                **{f"std_{k}": v for k, v in sc.items()},
            }
        )
        for m in metrics:
            value, std_value = m.value(c), m.value(sc)
            if value is None or m.weight(c) <= 0:
                continue
            long_rows.append(
                {
                    "week_start": wk.isoformat(),
                    "group": group,
                    "metric": m.name,
                    "value": round(value, 4),
                    "std_value": None if std_value is None else round(std_value, 4),
                    "weight": round(m.weight(c), 4),
                    "std_weight": round(m.weight(sc), 4),
                    "higher_is_better": m.higher_is_better,
                }
            )
    return long_rows, count_rows


def run(
    season: int,
    limit: int | None = None,
    pause: float = 0.3,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    players = read_csv(data_dir / PLAYERS_FILE)
    if limit:
        players = players[:limit]
    rows, count_rows = [], []
    for i, p in enumerate(players, 1):
        pitcher = p["is_pitcher"] == "True"
        ident = {
            "player_id": p["player_id"],
            "full_name": p["full_name"],
            "team": p["team"],
            "season": season,
        }
        n_weeks = 0
        for group in ("pitching" if pitcher else "hitting", "fielding"):
            splits = fetch_game_log(int(p["player_id"]), season, group)
            long_rows, wide_rows = weekly_metrics(splits, group)
            rows += [{**ident, **r} for r in long_rows]
            count_rows += [{**ident, **r} for r in wide_rows]
            n_weeks += len(wide_rows)
            time.sleep(pause)
        print(f"[{i}/{len(players)}] {p['full_name']}: {n_weeks} group-weeks")
    stats_csv, counts_csv = data_dir / STATS_FILE, data_dir / STATS_COUNTS_FILE
    n = write_csv(stats_csv, rows, FIELDS)
    write_csv(counts_csv, count_rows, COUNTS_FIELDS)
    print(f"Wrote {n} player-week-metric rows to {stats_csv} ({counts_csv} for counts)")
