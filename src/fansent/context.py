"""S5: does reaction depend on the team's situation? (analyses A16-A19)

Spec: docs/superpowers/specs/2026-11-23-reverse-direction-design.md (S5).

The S1a daily model (team subs, own-team units) plus interactions of the lag-0 and lag-1 game
value with context, all entered jointly; each estimate is the sum of its two interaction
slopes:

- A16 team won that game (W at the same lag);
- A17 team win% before the fan-day (z-scored over unit-days);
- A18 player new to the team this season (not on its first-week roster);
- A19 rookie (MLB debut in this season, from the people endpoint cache).

Deviation from the spec draft: "games back from a playoff spot" (A20) is not built; it needs
league/division standings by day and is left out rather than approximated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .reaction import LEAGUE_FANBASE, _dow, within_wls

CONTEXT = {"A16": "won", "A17": "winpct_z", "A18": "new_to_team", "A19": "rookie"}


def team_winpct(games: pd.DataFrame) -> pd.DataFrame:
    """(team_id, date, winpct_pre): team record strictly before each game date."""
    t = games.groupby(["team_id", "date"], as_index=False)["team_win"].max()
    t = t.sort_values(["team_id", "date"])
    g = t.groupby("team_id")
    wins = g["team_win"].cumsum() - t["team_win"]
    n = g.cumcount()
    t["winpct_pre"] = (wins / n.replace(0, np.nan)).fillna(0.5)
    return t[["team_id", "date", "winpct_pre"]]


def debut_years(people_dir: Path) -> pd.Series:
    out = {}
    for path in people_dir.glob("*.json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        p = d["people"][0] if "people" in d else d
        debut = p.get("mlbDebutDate") or ""
        out[str(p.get("id", path.stem))] = int(debut[:4]) if debut[:4].isdigit() else np.nan
    return pd.Series(out, dtype=float)


def first_week_pairs(player_teams: pd.DataFrame) -> set[tuple[str, int]]:
    pt = player_teams.assign(week_start=pd.to_datetime(player_teams["week_start"]))
    first = pt[pt["week_start"] == pt["week_start"].min()]
    return set(zip(first["player_id"].astype(str), first["team_id"].astype(int), strict=True))


def add_context(
    panel: pd.DataFrame,
    winpct: pd.DataFrame,
    debut: pd.Series,
    season: int,
    first_week: set[tuple[str, int]],
) -> pd.DataFrame:
    p = panel.sort_values("fan_day").copy()
    p["fan_day"] = p["fan_day"].astype("datetime64[ns]")
    w = winpct.sort_values("date").rename(columns={"date": "fan_day"})
    w["fan_day"] = w["fan_day"].astype("datetime64[ns]")
    p["team_id"] = p["team_id"].astype(int)
    w["team_id"] = w["team_id"].astype(int)
    p = pd.merge_asof(p, w, on="fan_day", by="team_id", direction="backward")
    p["winpct_pre"] = p["winpct_pre"].fillna(0.5)
    p["winpct_z"] = (p["winpct_pre"] - p["winpct_pre"].mean()) / p["winpct_pre"].std(ddof=0)
    pairs = list(zip(p["player_id"].astype(str), p["team_id"].astype(int), strict=True))
    p["new_to_team"] = [float(x not in first_week) for x in pairs]
    p["rookie"] = (p["player_id"].map(debut) == season).astype(float)
    return p


def fit_context(p: pd.DataFrame, n_lags: int):
    base = [f"vz_l{lag}" for lag in range(n_lags + 1)]
    base += [f"played_l{lag}" for lag in range(n_lags + 1)]
    base += [f"team_win_l{lag}" for lag in range(n_lags + 1)]
    cols, X = list(base), [p[base].to_numpy(float)]
    for ctx in CONTEXT.values():
        for lag in (0, 1):
            c = p[f"team_win_l{lag}"] if ctx == "won" else p[ctx]
            X.append((p[f"vz_l{lag}"] * c).to_numpy(float)[:, None])
            cols.append(f"vz_l{lag}_x_{ctx}")
        if ctx != "won":
            X.append(p[ctx].to_numpy(float)[:, None])
            cols.append(ctx)
    X = np.column_stack([*X, _dow(p["fan_day"])])
    cols += [f"dow{k}" for k in range(1, 7)]
    return within_wls(
        p["sent"].to_numpy(),
        X,
        p["unit"].to_numpy(),
        p["n_mentions"].to_numpy(),
        p["player_id"].to_numpy(),
        cols,
    )


def run(data_dir, season: int, prereg, out_dir, n_lags: int = 7) -> pd.DataFrame:
    from . import daypanel as dp
    from . import reaction
    from .multitest import benjamini_hochberg, holm

    reaction.require_open_season(season, prereg)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel, _, _, _ = reaction.build(data_dir, season, n_lags, 5)
    panel = panel[panel["fanbase"] != LEAGUE_FANBASE]
    games = dp.read_game_logs(data_dir / "raw" / "mlb" / "gamelogs" / str(season))
    pt = pd.read_csv(data_dir / "player_teams.csv", dtype={"player_id": str})
    p = add_context(
        panel,
        team_winpct(games),
        debut_years(data_dir / "raw" / "mlb" / "people"),
        season,
        first_week_pairs(pt),
    )
    rows = []
    for group, g in p.groupby("group"):
        r = fit_context(g, n_lags)
        for aid, ctx in CONTEXT.items():
            terms = [f"vz_l0_x_{ctx}", f"vz_l1_x_{ctx}"]
            a = np.zeros(len(r.names))
            a[[r.names.index(t) for t in terms]] = 1.0
            est, se = float(a @ r.coef), float(np.sqrt(a @ r.vcov @ a))
            share = float(g[ctx].mean()) if ctx != "won" else float(g["team_win_l0"].mean())
            rows.append(
                {
                    "analysis": aid,
                    "group": group,
                    "context": ctx,
                    "estimate": est,
                    "se": se,
                    "p": float(2 * norm.sf(abs(est / se))),
                    "n_obs": r.n_obs,
                    "context_mean": share,
                }
            )
    res = pd.DataFrame(rows)
    res["family"] = "F5"
    res["q_bh"] = benjamini_hochberg(res["p"].to_numpy())
    res["p_holm"] = holm(res["p"].to_numpy())
    res.to_csv(out_dir / "s5_results.csv", index=False)
    return res
