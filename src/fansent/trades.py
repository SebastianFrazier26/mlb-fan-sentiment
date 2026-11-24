"""S6: how old and new fanbases treat a traded player (analyses A21-A22).

Spec: docs/superpowers/specs/2026-11-23-reverse-direction-design.md (S6).

Event = a player's first mid-season team change in `player_teams.csv` (a change into the
season's first roster week is not an event). Event week 0 = first week on the new team.

- A21 honeymoon gap: (new fanbase - r/baseball) sentiment in weeks 0..3 minus (old fanbase -
  r/baseball) in weeks -4..-1, same player, adjusted for the week's game value; pairs need
  >= min_mentions on both sides; CR1 by player.
- A22 early-slope gap: in the new fanbase, the weekly value slope in weeks 0..3 minus the
  slope from week 4 on (unit FE). Positive = new fans react more while they know less.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .reaction import LEAGUE_FANBASE, within_wls

EARLY = range(4)
BEFORE = range(-4, 0)


def trade_events(player_teams: pd.DataFrame) -> pd.DataFrame:
    """(player_id, old_team, new_team, week0) for each player's first mid-season move."""
    pt = player_teams.assign(
        player_id=player_teams["player_id"].astype(str),
        team_id=player_teams["team_id"].astype(int),
        week_start=pd.to_datetime(player_teams["week_start"]),
    ).drop_duplicates(["player_id", "week_start"], keep="last")
    pt = pt.sort_values(["player_id", "week_start"])
    first_week = pt["week_start"].min()
    prev = pt.groupby("player_id")["team_id"].shift()
    move = pt[prev.notna() & (pt["team_id"] != prev) & (pt["week_start"] > first_week)]
    move = move.assign(old_team=prev[move.index].astype(int))
    move = move.drop_duplicates("player_id", keep="first")
    return move.rename(columns={"team_id": "new_team", "week_start": "week0"})[
        ["player_id", "old_team", "new_team", "week0"]
    ].reset_index(drop=True)


def event_panel(wk: pd.DataFrame, events: pd.DataFrame, fb_team: dict[str, int]) -> pd.DataFrame:
    """Weekly rows of traded players with event time and side (new / old / league)."""
    w = wk.merge(events, on="player_id", how="inner")
    w["event_week"] = ((w["week"] - w["week0"]).dt.days // 7).astype(int)
    team = w["fanbase"].str.lower().map(fb_team)
    w["side"] = np.select(
        [w["fanbase"] == LEAGUE_FANBASE, team == w["new_team"], team == w["old_team"]],
        ["league", "new", "old"],
        default="other",
    )
    return w[w["side"] != "other"]


def honeymoon_gap(ev: pd.DataFrame, min_mentions: int = 3) -> tuple[float, float, int, int]:
    ev = ev[ev["n_mentions"] >= min_mentions]
    lg = ev[ev["side"] == "league"][["player_id", "week", "sent", "n_mentions"]]
    parts = []
    for side, weeks in (("new", EARLY), ("old", BEFORE)):
        s = ev[(ev["side"] == side) & ev["event_week"].isin(list(weeks))]
        m = s.merge(lg, on=["player_id", "week"], suffixes=("", "_lg"))
        m["d"] = m["sent"] - m["sent_lg"]
        m["is_new"] = float(side == "new")
        m["w"] = 2 / (1 / m["n_mentions"] + 1 / m["n_mentions_lg"])
        parts.append(m)
    m = pd.concat(parts, ignore_index=True)
    if m["is_new"].nunique() < 2:
        return float("nan"), float("nan"), len(m), m["player_id"].nunique()
    r = within_wls(
        m["d"].to_numpy(),
        m[["is_new", "vz_l0"]].to_numpy(float),
        np.zeros(len(m), int),
        m["w"].to_numpy(),
        m["player_id"].to_numpy(),
        ["is_new", "vz_l0"],
        absorb=False,
    )
    i = r.names.index("is_new")
    return float(r.coef[i]), float(r.se[i]), len(m), int(m["player_id"].nunique())


def early_slope_gap(ev: pd.DataFrame) -> tuple[float, float, int, int]:
    s = ev[(ev["side"] == "new") & (ev["event_week"] >= 0)].copy()
    s["early"] = s["event_week"].isin(list(EARLY)).astype(float)
    s["unit"] = s["fanbase"] + "|" + s["player_id"]
    X = np.column_stack([s["vz_l0"], s["vz_l0"] * s["early"], s["early"]])
    r = within_wls(
        s["sent"].to_numpy(),
        X,
        s["unit"].to_numpy(),
        s["n_mentions"].to_numpy(),
        s["player_id"].to_numpy(),
        ["vz_l0", "vz_l0_x_early", "early"],
    )
    i = r.names.index("vz_l0_x_early")
    return float(r.coef[i]), float(r.se[i]), r.n_obs, int(s["player_id"].nunique())


def run(data_dir, season: int, prereg, out_dir) -> pd.DataFrame:
    from pathlib import Path

    from . import reaction
    from .multitest import benjamini_hochberg, holm
    from .panel import load_fanbase_teams

    reaction.require_open_season(season, prereg)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, wk, _, _ = reaction.build(data_dir, season, 7, 5)
    events = trade_events(pd.read_csv(data_dir / "player_teams.csv", dtype={"player_id": str}))
    fb_team = {fb: next(iter(t)) for fb, t in load_fanbase_teams().items() if len(t) == 1}
    ev = event_panel(wk, events, fb_team)
    rows = []
    est, se, n, k = honeymoon_gap(ev)
    rows.append({"analysis": "A21", "estimate": est, "se": se, "n_pairs": n, "n_players": k})
    est, se, n, k = early_slope_gap(ev)
    rows.append({"analysis": "A22", "estimate": est, "se": se, "n_pairs": n, "n_players": k})
    res = pd.DataFrame(rows)
    res["p"] = 2 * norm.sf(np.abs(res["estimate"] / res["se"]))
    res["n_events"] = len(events)
    res["family"] = "F6"
    res["q_bh"] = benjamini_hochberg(res["p"].to_numpy())
    res["p_holm"] = holm(res["p"].to_numpy())
    res.to_csv(out_dir / "s6_results.csv", index=False)
    return res
