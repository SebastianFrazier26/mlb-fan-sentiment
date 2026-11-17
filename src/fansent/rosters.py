"""League-wide dated 40-man rosters from the MLB Stats API (free, no key).

Outputs (contract shared with the private fansent-collect repo):
  players.csv      one row per unique player_id, team = team at the last week seen
  player_teams.csv (player_id, team_id, week_start): one row per player-week on a 40-man
  teams.csv        team_id, team (abbrev), name
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from .util import (
    DEFAULT_DATA_DIR,
    MLB_RAW_DIR,
    PLAYER_TEAMS_FILE,
    PLAYERS_FILE,
    TEAMS_FILE,
    cached_get_json,
    write_csv,
)

API = "https://statsapi.mlb.com/api/v1"

# Monday of the first and last regular-season weeks.
SEASON_WEEKS = {2026: (dt.date(2026, 3, 23), dt.date(2026, 9, 21))}

# Thursday snapshot: the middle of the Mon-Sun week, so a trade is attributed to the new team
# for the week it falls in about as often as to the old one (a Monday snapshot would lag every
# mid-week move by a full week).
SNAPSHOT_OFFSET_DAYS = 3

SUFFIXES = {"jr.", "jr", "sr.", "sr", "ii", "iii", "iv"}

PLAYER_FIELDS = [
    "player_id",
    "full_name",
    "first_name",
    "last_name",
    "suffix",
    "team",
    "team_id",
    "subreddit",
    "position",
    "is_pitcher",
]
PLAYER_TEAM_FIELDS = ["player_id", "team_id", "week_start"]
TEAM_FIELDS = ["team_id", "team", "name"]

Fetch = Callable[[str, dict | None, Path | None], dict]


def season_mondays(first: dt.date, last: dt.date) -> list[dt.date]:
    return [first + dt.timedelta(weeks=i) for i in range((last - first).days // 7 + 1)]


def split_name(person: dict) -> tuple[str, str, str]:
    """(first, last, suffix) from API person fields, never from the display name.

    `lastName` already excludes the suffix ("Acuña" / nameSuffix "Jr."); `useLastName` includes
    it, so it is only a fallback, with a trailing suffix token stripped.
    """
    suffix = (person.get("nameSuffix") or "").strip()
    last = (person.get("lastName") or "").strip()
    if not last:
        parts = (person.get("useLastName") or person.get("fullName") or "").split()
        if parts and parts[-1].lower() in SUFFIXES:
            suffix = suffix or parts.pop()
        last = parts[-1] if parts else ""
    return (person.get("firstName") or "").strip(), last, suffix


def fetch_teams(season: int, data_dir: Path, fetch: Fetch) -> list[dict]:
    cache = data_dir / MLB_RAW_DIR / "rosters" / f"teams_{season}.json"
    data = fetch(f"{API}/teams", {"sportId": 1, "season": season}, cache)
    return sorted(
        ({"team_id": t["id"], "team": t["abbreviation"], "name": t["name"]} for t in data["teams"]),
        key=lambda t: t["team_id"],
    )


def fetch_weekly_rosters(
    teams: list[dict],
    mondays: list[dt.date],
    data_dir: Path,
    fetch: Fetch,
    today: dt.date | None = None,
) -> list[dict]:
    """One dict per (team, week, roster entry): player_id, team_id, week_start, position, name."""
    today = today or dt.datetime.now(dt.UTC).date()
    out = []
    for monday in mondays:
        snap = monday + dt.timedelta(days=SNAPSHOT_OFFSET_DAYS)
        for t in teams:
            # Never cache a snapshot that is still in the future/present: the roster can change.
            cache = (
                data_dir
                / MLB_RAW_DIR
                / "rosters"
                / str(monday.year)
                / f"{t['team_id']}_{snap}.json"
                if snap < today
                else None
            )
            data = fetch(
                f"{API}/teams/{t['team_id']}/roster",
                {"rosterType": "40Man", "date": snap.isoformat()},
                cache,
            )
            for entry in data.get("roster", []):
                out.append(
                    {
                        "player_id": entry["person"]["id"],
                        "team_id": t["team_id"],
                        "week_start": monday.isoformat(),
                        "position": entry.get("position", {}).get("abbreviation", ""),
                        "full_name": entry["person"].get("fullName", ""),
                    }
                )
    return out


def fetch_people(ids: list[int], data_dir: Path, fetch: Fetch, batch: int = 100) -> dict[int, dict]:
    """player_id -> API person record, per-player cache so reruns only fetch new ids."""
    cache_dir = data_dir / MLB_RAW_DIR / "people"
    people: dict[int, dict] = {}
    missing = []
    for pid in ids:
        f = cache_dir / f"{pid}.json"
        if f.exists():
            people[pid] = json.loads(f.read_text(encoding="utf-8"))
        else:
            missing.append(pid)
    for i in range(0, len(missing), batch):
        chunk = missing[i : i + batch]
        data = fetch(f"{API}/people", {"personIds": ",".join(map(str, chunk))}, None)
        for p in data.get("people", []):
            people[p["id"]] = p
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"{p['id']}.json").write_text(json.dumps(p), encoding="utf-8")
    return people


def build_outputs(
    entries: list[dict],
    people: dict[int, dict],
    teams: list[dict],
    team_subreddit: dict[int, str],
) -> tuple[list[dict], list[dict]]:
    """(players rows, player_teams rows). Player's team/position = the last week he appears."""
    abbrev = {t["team_id"]: t["team"] for t in teams}
    seen = {(e["player_id"], e["team_id"], e["week_start"]) for e in entries}
    player_teams = [
        {"player_id": p, "team_id": t, "week_start": w} for p, t, w in sorted(seen, key=_pt_key)
    ]
    last: dict[int, dict] = {}
    for e in sorted(entries, key=lambda e: e["week_start"]):
        last[e["player_id"]] = e
    players = []
    for pid, e in sorted(last.items()):
        person = people.get(pid, {"fullName": e["full_name"]})
        first, last_name, suffix = split_name(person)
        full = person.get("fullName") or e["full_name"]
        players.append(
            {
                "player_id": pid,
                "full_name": full,
                "first_name": first,
                "last_name": last_name or _fallback_last(full),
                "suffix": suffix,
                "team": abbrev.get(e["team_id"], ""),
                "team_id": e["team_id"],
                "subreddit": team_subreddit.get(e["team_id"], ""),
                "position": e["position"],
                "is_pitcher": e["position"] == "P",
            }
        )
    return players, player_teams


def _pt_key(t: tuple) -> tuple:
    return (t[2], t[1], t[0])


def _fallback_last(full: str) -> str:
    parts = re.split(r"\s+", full.strip())
    if parts and parts[-1].lower() in SUFFIXES:
        parts.pop()
    return parts[-1] if parts else ""


def run(
    season: int,
    data_dir: Path = DEFAULT_DATA_DIR,
    teams: list[str] | None = None,
    first_week: dt.date | None = None,
    last_week: dt.date | None = None,
    pause: float = 0.15,
    fetch: Fetch | None = None,
) -> None:
    from .panel import load_team_subreddit_rows

    if fetch is None:

        def fetch(url, params, cache):
            return cached_get_json(url, params, cache, pause)

    if first_week is None or last_week is None:
        if season not in SEASON_WEEKS:
            raise SystemExit(
                f"no built-in season window for {season}: give --first-week/--last-week"
            )
        first_week, last_week = (
            first_week or SEASON_WEEKS[season][0],
            last_week or SEASON_WEEKS[season][1],
        )
    all_teams = fetch_teams(season, data_dir, fetch)
    if teams:
        want = {t.upper() for t in teams}
        all_teams = [t for t in all_teams if t["team"].upper() in want or str(t["team_id"]) in want]
        if not all_teams:
            raise SystemExit(f"--teams {teams} matched no team")
    mondays = season_mondays(first_week, last_week)
    entries = fetch_weekly_rosters(all_teams, mondays, data_dir, fetch)
    ids = sorted({e["player_id"] for e in entries})
    people = fetch_people(ids, data_dir, fetch)
    team_sub: dict[int, str] = {}
    for r in load_team_subreddit_rows():
        team_sub.setdefault(int(r["team_id"]), r["subreddit"])
    players, player_teams = build_outputs(entries, people, all_teams, team_sub)
    write_csv(data_dir / TEAMS_FILE, all_teams, TEAM_FIELDS)
    n = write_csv(data_dir / PLAYERS_FILE, players, PLAYER_FIELDS)
    m = write_csv(data_dir / PLAYER_TEAMS_FILE, player_teams, PLAYER_TEAM_FIELDS)
    teams_by_player: dict[str, set] = defaultdict(set)
    for r in player_teams:
        teams_by_player[r["player_id"]].add(r["team_id"])
    changed = sum(1 for v in teams_by_player.values() if len(v) > 1)
    print(
        f"Wrote {n} players to {data_dir / PLAYERS_FILE}, {m} player-weeks to "
        f"{data_dir / PLAYER_TEAMS_FILE} ({len(all_teams)} teams, {len(mondays)} weeks, "
        f"{changed} players on >1 team)"
    )
