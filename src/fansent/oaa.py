"""Load a Baseball Savant Outs Above Average CSV export (monthly splits) into a tidy
player-month table. Read-only: downloading belongs in the private collector repo.

Savant's leaderboard export names the id column `player_id` (MLBAM id, the same id
the Stats API uses) and the metric `outs_above_average`. The month-split layout is not
documented, so the month may arrive as `month` ("2026-05" / "2026-05-01" / 5 with a
`year` column) or as a `date`/`game_date`-style column; anything else raises.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from .util import DEFAULT_DATA_DIR, OAA_FILE, write_csv

FIELDS = ["player_id", "full_name", "month", "oaa", "attempts", "fielding_runs_prevented"]
MONTH_COLUMNS = ("month", "game_month", "date", "game_date", "start_date")
NAME_COLUMN = "last_name, first_name"


def _month(row: dict, fieldnames: list[str]) -> dt.date:
    for col in MONTH_COLUMNS:
        raw = (row.get(col) or "").strip()
        if not raw:
            continue
        if raw.isdigit():
            return dt.date(int(row["year"]), int(raw), 1)
        return dt.date.fromisoformat(raw[:7] + "-01")
    raise ValueError(f"no month column among {MONTH_COLUMNS}; got columns {fieldnames}")


def _num(raw: str | None) -> float | None:
    raw = (raw or "").strip()
    return float(raw) if raw else None


def _full_name(raw: str) -> str:
    last, _, first = raw.partition(",")
    return f"{first.strip()} {last.strip()}".strip()


def load_oaa(path: Path | str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        names = [n.strip() for n in reader.fieldnames or []]
        for need in ("player_id", "outs_above_average"):
            if need not in names:
                raise ValueError(f"{path}: missing required column {need!r}; got {names}")
        reader.fieldnames = names
        agg: dict[tuple[str, dt.date], dict] = {}
        for row in reader:
            month = _month(row, names)
            oaa = _num(row["outs_above_average"])
            if oaa is None:
                continue
            key = (row["player_id"].strip(), month)
            # A player can have one export row per position; OAA is additive across them.
            cur = agg.setdefault(
                key,
                {
                    "player_id": key[0],
                    "full_name": _full_name(row.get(NAME_COLUMN, "")),
                    "month": month.isoformat(),
                    "oaa": 0.0,
                    "attempts": None,
                    "fielding_runs_prevented": None,
                },
            )
            cur["oaa"] += oaa
            for src, dst in (("attempts", "attempts"), ("fielding_runs_prevented",) * 2):
                v = _num(row.get(src))
                if v is not None:
                    cur[dst] = (cur[dst] or 0) + v
    return sorted(agg.values(), key=lambda r: (r["month"], r["player_id"]))


def run(path: str, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    rows = load_oaa(path)
    out = data_dir / OAA_FILE
    n = write_csv(out, rows, FIELDS)
    print(f"Wrote {n} player-months to {out}")
