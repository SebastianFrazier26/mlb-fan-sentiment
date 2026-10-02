"""Fetch team rosters from the MLB Stats API (free, no key) into players.csv."""

from __future__ import annotations

import requests

from .util import PLAYERS_CSV, write_csv

API = "https://statsapi.mlb.com/api/v1"

# teamId, display name, fan subreddit
TEAMS = {
    112: ("Chicago Cubs", "CHICubs"),
    111: ("Boston Red Sox", "redsox"),
}

FIELDS = [
    "player_id",
    "full_name",
    "last_name",
    "team",
    "team_id",
    "subreddit",
    "position",
    "is_pitcher",
]


def fetch_rosters(season: int, roster_type: str = "fullSeason") -> list[dict]:
    rows = []
    for team_id, (team_name, subreddit) in TEAMS.items():
        resp = requests.get(
            f"{API}/teams/{team_id}/roster",
            params={"rosterType": roster_type, "season": season},
            timeout=30,
        )
        resp.raise_for_status()
        for entry in resp.json().get("roster", []):
            person = entry["person"]
            pos = entry.get("position", {}).get("abbreviation", "")
            rows.append(
                {
                    "player_id": person["id"],
                    "full_name": person["fullName"],
                    "last_name": person["fullName"].split()[-1],
                    "team": team_name,
                    "team_id": team_id,
                    "subreddit": subreddit,
                    "position": pos,
                    "is_pitcher": pos == "P",
                }
            )
    return rows


def run(season: int, roster_type: str = "fullSeason") -> None:
    rows = fetch_rosters(season, roster_type)
    n = write_csv(PLAYERS_CSV, rows, FIELDS)
    per_team = {name: sum(1 for r in rows if r["team"] == name) for _, (name, _) in TEAMS.items()}
    print(f"Wrote {n} players to {PLAYERS_CSV} ({per_team})")
