"""S2: do fans respond to the persistent (skill) or the transient (luck) parts of a week?

Spec: docs/superpowers/specs/2026-11-23-reverse-direction-design.md (S2, analyses A6-A8).

Per player-week, components from the game logs and Statcast PA rows, z-scored within group
over eligible player-weeks:

- hitters: skill K%, BB%, ISO, xwOBA; luck BABIP, wOBA - xwOBA
- pitchers: skill K%, BB%, FIP, xwOBA-against; luck BABIP-against, LOB%, ERA - FIP

Two slope vectors on the jointly entered components:

- fan slopes b: fan-week sentiment (own-team units) on the week's components, unit FE,
  mention weights, CR1 by player;
- persistence benchmark w: the player's rest-of-season runs per exposure (from the next week
  on) on the same components plus his season-to-date rate before the week, weights = rest-of-
  season exposure, no FE (with a player FE the rest-of-season mean is mechanically tied to
  the week's own value).

dL = L(b) - L(w), L(x) = sum_luck |x_c| / sum_all |x_c|. 0 = fans weigh luck as much as its
predictive value warrants; > 0 = they overweight luck. Player-block bootstrap for the CI and
a two-sided bootstrap p.

Deviation from the spec draft: week components are not EB-shrunk; small weeks are dropped by
an exposure floor instead (MIN_EXPOSURE) and the unit FE absorbs player level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import daypanel as dp
from .reaction import LEAGUE_FANBASE, within_wls

STATCAST_FILE = "statcast_pa.csv"
SKILL = {
    "hitting": ["k_pct", "bb_pct", "iso", "xwoba"],
    "pitching": ["k_pct", "bb_pct", "fip", "xwoba"],
}
LUCK = {
    "hitting": ["babip", "woba_minus_xwoba"],
    "pitching": ["babip", "lob_pct", "era_minus_fip"],
}
MIN_EXPOSURE = {"hitting": 10, "pitching": 12}
MIN_ROS = 50


def week_of(d: pd.Series) -> pd.Series:
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def statcast_daily(pa: pd.DataFrame) -> pd.DataFrame:
    """PA rows -> (player_id, group, date, xw_num, w_num, w_den) for batters and pitchers.

    xwOBA per PA: Savant's estimated wOBA on balls in play, the actual wOBA value otherwise
    (K, BB, HBP), which is how Savant builds xwOBA."""
    d = pa.copy()
    d["date"] = pd.to_datetime(d["game_date"])
    den = pd.to_numeric(d["woba_denom"], errors="coerce").fillna(0)
    wv = pd.to_numeric(d["woba_value"], errors="coerce").fillna(0)
    est = pd.to_numeric(d["estimated_woba_using_speedangle"], errors="coerce")
    d["w_den"], d["w_num"] = den, wv
    d["xw_num"] = est.where(est.notna(), wv) * (den > 0)
    parts = []
    for col, group in (("batter", "hitting"), ("pitcher", "pitching")):
        g = d.groupby([d[col].astype(str), "date"])[["xw_num", "w_num", "w_den"]].sum()
        g = g.reset_index().rename(columns={col: "player_id"})
        parts.append(g.assign(group=group))
    return pd.concat(parts, ignore_index=True)


def statcast_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.assign(week=week_of(daily["date"]))
    return d.groupby(["player_id", "group", "week"], as_index=False)[
        ["xw_num", "w_num", "w_den"]
    ].sum()


def weekly_components(games: pd.DataFrame, sc_week: pd.DataFrame | None) -> pd.DataFrame:
    """Game-log counts (+ Statcast sums) -> one row per player-week with raw components."""
    g = games.assign(week=week_of(games["date"]))
    num = [c for c in g.columns if c not in ("player_id", "date", "group", "team_id", "week")]
    num = [c for c in num if pd.api.types.is_numeric_dtype(g[c]) and c != "team_win"]
    w = g.groupby(["player_id", "group", "week"], as_index=False)[num].sum()
    for c in {*dp.HIT_KEYS.values(), *dp.PITCH_KEYS.values()} - set(w.columns):
        w[c] = 0
    if sc_week is not None:
        w = w.merge(sc_week, on=["player_id", "group", "week"], how="left")
    else:
        w[["xw_num", "w_num", "w_den"]] = np.nan
    hit = w["group"] == "hitting"
    out = pd.DataFrame(index=w.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        xwoba = w["xw_num"] / w["w_den"]
        # Hitters
        out.loc[hit, "k_pct"] = w["k"] / w["pa"]
        out.loc[hit, "bb_pct"] = w["bb"] / w["pa"]
        out.loc[hit, "iso"] = (w["double"] + 2 * w["triple"] + 3 * w["hr"]) / w["ab"]
        out.loc[hit, "babip"] = (w["h"] - w["hr"]) / (w["ab"] - w["k"] - w["hr"] + w["sf"])
        out.loc[hit, "woba_minus_xwoba"] = (w["w_num"] - w["xw_num"]) / w["w_den"]
        # Pitchers
        p = ~hit
        ip = w["outs"] / 3
        fip = (13 * w["hr"] + 3 * (w["bb"] + w["hbp"]) - 2 * w["k"]) / ip
        out.loc[p, "k_pct"] = w["k"] / w["bf"]
        out.loc[p, "bb_pct"] = (w["bb"] + w["hbp"]) / w["bf"]
        out.loc[p, "fip"] = fip
        bip = w["bf"] - w["k"] - w["bb"] - w["hbp"] - w["hr"]
        out.loc[p, "babip"] = (w["h"] - w["hr"]) / bip
        on = w["h"] + w["bb"] + w["hbp"]
        out.loc[p, "lob_pct"] = (on - w["r"]) / (on - 1.4 * w["hr"])
        out.loc[p, "era_minus_fip"] = 9 * w["er"] / ip - fip
        out["xwoba"] = xwoba
    exposure = np.where(hit, w["pa"], w["bf"])
    keep = ["player_id", "group", "week"]
    return pd.concat([w[keep], out], axis=1).assign(exposure=exposure)


def zscore_components(wc: pd.DataFrame) -> pd.DataFrame:
    """Eligible weeks only (exposure floor, all of the group's components finite), z-scored
    within group."""
    parts = []
    for group, g in wc.groupby("group"):
        cols = SKILL[group] + LUCK[group]
        g = g[g["exposure"] >= MIN_EXPOSURE[group]]
        g = g[np.isfinite(g[cols]).all(axis=1)].copy()
        g[cols] = (g[cols] - g[cols].mean()) / g[cols].std(ddof=0)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def rest_of_season(values: pd.DataFrame) -> pd.DataFrame:
    """Per player-week: runs per exposure before the week (`std_rate`) and from the next week
    to season end (`ros_rate`, `ros_n`). values: game_values() output."""
    v = values.assign(week=week_of(values["date"]))
    wk = v.groupby(["player_id", "group", "week"], as_index=False)[["v", "n"]].sum()
    wk = wk.sort_values(["player_id", "group", "week"])
    g = wk.groupby(["player_id", "group"])
    tot_v, tot_n = g["v"].transform("sum"), g["n"].transform("sum")
    cum_v, cum_n = g["v"].cumsum(), g["n"].cumsum()
    wk["std_n"] = cum_n - wk["n"]
    wk["std_rate"] = (cum_v - wk["v"]) / wk["std_n"].replace(0, np.nan)
    wk["ros_n"] = tot_n - cum_n
    wk["ros_rate"] = (tot_v - cum_v) / wk["ros_n"].replace(0, np.nan)
    return wk[["player_id", "group", "week", "std_rate", "std_n", "ros_rate", "ros_n"]]


def luck_share(coef: pd.Series, group: str) -> float:
    a = coef.abs()
    return float(a[LUCK[group]].sum() / a[SKILL[group] + LUCK[group]].sum())


@dataclass
class S2Fit:
    b: pd.Series
    b_se: pd.Series
    w: pd.Series
    w_se: pd.Series
    dL: float
    n_fan: int
    n_bench: int


def fit_fan(fan: pd.DataFrame, group: str):
    cols = SKILL[group] + LUCK[group]
    return within_wls(
        fan["sent"].to_numpy(),
        fan[cols].to_numpy(float),
        fan["unit"].to_numpy(),
        fan["n_mentions"].to_numpy(),
        fan["player_id"].to_numpy(),
        cols,
    )


def fit_bench(bench: pd.DataFrame, group: str):
    cols = SKILL[group] + LUCK[group]
    X = np.column_stack([bench[cols].to_numpy(float), bench["std_rate"].to_numpy(float)])
    return within_wls(
        bench["ros_rate"].to_numpy(),
        X,
        np.zeros(len(bench), dtype=int),
        bench["ros_n"].to_numpy(),
        bench["player_id"].to_numpy(),
        [*cols, "std_rate"],
        absorb=False,
    )


def fit_s2(fan: pd.DataFrame, bench: pd.DataFrame, group: str) -> S2Fit:
    cols = SKILL[group] + LUCK[group]
    rf, rb = fit_fan(fan, group), fit_bench(bench, group)
    b = pd.Series(rf.coef, index=rf.names)[cols]
    b_se = pd.Series(rf.se, index=rf.names)[cols]
    w = pd.Series(rb.coef, index=rb.names)[cols]
    w_se = pd.Series(rb.se, index=rb.names)[cols]
    return S2Fit(b, b_se, w, w_se, luck_share(b, group) - luck_share(w, group), rf.n_obs, rb.n_obs)


def _resample(df: pd.DataFrame, pick: np.ndarray, by: dict) -> pd.DataFrame:
    parts, reps = [], []
    for rep, p in enumerate(pick):
        idx = by.get(p)
        if idx is not None:
            parts.append(idx)
            reps.append(np.full(len(idx), rep))
    if not parts:
        return df.iloc[:0]
    out = df.iloc[np.concatenate(parts)].copy()
    tag = "#" + np.concatenate(reps).astype(str)
    out["player_id"] = out["player_id"].astype(str).to_numpy() + tag
    if "unit" in out:
        out["unit"] = out["unit"].astype(str).to_numpy() + tag
    return out


def bootstrap_dl(
    fan: pd.DataFrame,
    bench: pd.DataFrame,
    group: str,
    n_boot: int,
    seed: int,
    by_fanbase: bool = False,
) -> pd.DataFrame:
    """Player-block bootstrap of dL (one row per draw); with by_fanbase also dL_k for every
    fanbase, all from the same resample so per-fanbase draws share the benchmark draw."""
    rng = np.random.default_rng(seed)
    players = np.union1d(fan["player_id"].unique(), bench["player_id"].unique())
    fan_by = fan.groupby("player_id").indices
    bench_by = bench.groupby("player_id").indices
    fanbases = sorted(fan["fanbase"].unique()) if by_fanbase else []
    rows = []
    for b in range(n_boot):
        pick = rng.choice(players, size=len(players), replace=True)
        f, bn = _resample(fan, pick, fan_by), _resample(bench, pick, bench_by)
        rb = fit_bench(bn, group)
        lw = luck_share(pd.Series(rb.coef, index=rb.names), group)
        rf = fit_fan(f, group)
        row = {"draw": b, "dL": luck_share(pd.Series(rf.coef, index=rf.names), group) - lw}
        for fb in fanbases:
            sub = f[f["fanbase"] == fb]
            if sub["player_id"].nunique() < 5:
                row[fb] = np.nan
                continue
            r = fit_fan(sub, group)
            row[fb] = luck_share(pd.Series(r.coef, index=r.names), group) - lw
        rows.append(row)
    return pd.DataFrame(rows)


def split_half_reliability(games: pd.DataFrame, sc_daily: pd.DataFrame | None) -> pd.DataFrame:
    """Odd/even game split per player: Spearman-Brown corrected rank correlation of each
    component across players with enough exposure in both halves. Skill should beat luck."""
    g = games.sort_values(["player_id", "group", "date"]).copy()
    g["half"] = g.groupby(["player_id", "group"]).cumcount() % 2
    halves = []
    for h in (0, 1):
        sub = g[g["half"] == h]
        sc = None
        if sc_daily is not None:
            keys = sub[["player_id", "group", "date"]]
            sc = sc_daily.merge(keys, on=["player_id", "group", "date"], how="inner")
            sc = sc.groupby(["player_id", "group"], as_index=False)[
                ["xw_num", "w_num", "w_den"]
            ].sum()
            sc["week"] = pd.Timestamp("2000-01-03")
        one = sub.drop(columns="half").assign(date=pd.Timestamp("2000-01-03"))
        halves.append(weekly_components(one, sc))
    m = halves[0].merge(halves[1], on=["player_id", "group"], suffixes=("_a", "_b"))
    rows = []
    for group in ("hitting", "pitching"):
        floor = 10 * MIN_EXPOSURE[group]
        mg = m[(m["group"] == group) & (m["exposure_a"] >= floor) & (m["exposure_b"] >= floor)]
        for c in SKILL[group] + LUCK[group]:
            x, y = mg[f"{c}_a"], mg[f"{c}_b"]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 20:
                continue
            r = float(np.corrcoef(x[ok].rank(), y[ok].rank())[0, 1])
            rows.append(
                {
                    "group": group,
                    "component": c,
                    "kind": "skill" if c in SKILL[group] else "luck",
                    "n_players": int(ok.sum()),
                    "r_half": r,
                    "reliability": 2 * r / (1 + r),
                }
            )
    return pd.DataFrame(rows)


def build(data_dir: Path, season: int, cutoff_hour: int = 5):
    """(fan-week panel with components, benchmark panel, component table, games)."""
    from .panel import load_fanbase_teams, load_team_subreddits
    from .reaction import load_inputs

    inp = load_inputs(data_dir)
    games = dp.read_game_logs(data_dir / "raw" / "mlb" / "gamelogs" / str(season))
    group = dp.player_group(inp["players"])
    games = games[games["group"] == games["player_id"].map(group)].reset_index(drop=True)
    sc_path = data_dir / STATCAST_FILE
    if not sc_path.exists():
        raise SystemExit(f"{sc_path} missing: run fansent-collect xstats-download first")
    sc_daily = statcast_daily(pd.read_csv(sc_path, dtype={"batter": str, "pitcher": str}))
    sc_week = statcast_weekly(sc_daily)
    comps = zscore_components(weekly_components(games, sc_week))
    ros = rest_of_season(dp.game_values(games))
    bench = comps.merge(ros, on=["player_id", "group", "week"], how="inner")
    bench = bench[(bench["ros_n"] >= MIN_ROS) & bench["std_rate"].notna()]
    days = dp.mention_days(
        inp["mentions"],
        inp["player_teams"],
        load_team_subreddits(),
        load_fanbase_teams(),
        cutoff_hour,
    )
    days = days.assign(cs=days["sent"] * days["n_mentions"])
    days["week"] = week_of(days["fan_day"])
    fw = days.groupby(["fanbase", "player_id", "week"], as_index=False).agg(
        cs=("cs", "sum"), n_mentions=("n_mentions", "sum")
    )
    fw["sent"] = fw["cs"] / fw["n_mentions"]
    fan = fw.drop(columns="cs").merge(comps, on=["player_id", "week"], how="inner")
    fan["unit"] = fan["fanbase"] + "|" + fan["player_id"]
    return fan, bench, games, sc_daily


def run(
    data_dir: Path,
    season: int,
    prereg: Path | None,
    n_boot: int = 199,
    seed: int = 0,
    out_dir: Path | None = None,
    min_mentions: int = 3,
) -> pd.DataFrame:
    import datetime as dt

    from .multitest import benjamini_hochberg, holm
    from .reaction import require_open_season

    require_open_season(season, prereg)
    out_dir = out_dir or data_dir / "runs" / f"{dt.datetime.now(dt.UTC).date()}-skill-luck"
    out_dir.mkdir(parents=True, exist_ok=True)
    fan, bench, games, sc_daily = build(data_dir, season)
    fan = fan[fan["n_mentions"] >= min_mentions]
    team = fan[fan["fanbase"] != LEAGUE_FANBASE]
    results, comps, fanbase_draws = [], [], []
    for aid, group in (("A6", "hitting"), ("A7", "pitching")):
        f, bn = team[team["group"] == group], bench[bench["group"] == group]
        fit = fit_s2(f, bn, group)
        boot = bootstrap_dl(f, bn, group, n_boot, seed, by_fanbase=True)
        d = boot["dL"].to_numpy()
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / (n_boot + 1)))
        lo, hi = np.nanquantile(d, [0.025, 0.975])
        results.append(
            {
                "analysis": aid,
                "group": group,
                "dL": fit.dL,
                "L_fan": luck_share(fit.b, group),
                "L_bench": luck_share(fit.w, group),
                "se_boot": float(np.nanstd(d, ddof=1)),
                "lo95": lo,
                "hi95": hi,
                "p_boot": p,
                "n_fan_weeks": fit.n_fan,
                "n_bench_weeks": fit.n_bench,
            }
        )
        for c in fit.b.index:
            comps.append(
                {
                    "analysis": "A8",
                    "group": group,
                    "component": c,
                    "kind": "skill" if c in SKILL[group] else "luck",
                    "b": fit.b[c],
                    "b_se": fit.b_se[c],
                    "w": fit.w[c],
                    "w_se": fit.w_se[c],
                }
            )
        fanbase_draws.append(boot.assign(group=group))
    res = pd.DataFrame(results)
    res["family"] = "F2"
    res["q_bh"] = benjamini_hochberg(res["p_boot"].to_numpy())
    res["p_holm"] = holm(res["p_boot"].to_numpy())
    comp = pd.DataFrame(comps)
    from scipy.stats import norm

    comp["p_b"] = 2 * norm.sf(np.abs(comp["b"] / comp["b_se"]))
    comp["family"] = "F2c"
    comp["q_bh_b"] = benjamini_hochberg(comp["p_b"].to_numpy())
    res.to_csv(out_dir / "s2_results.csv", index=False)
    comp.to_csv(out_dir / "s2_components.csv", index=False)
    pd.concat(fanbase_draws).to_csv(out_dir / "s2_boot_draws.csv", index=False)
    split_half_reliability(games, sc_daily).to_csv(out_dir / "s2_reliability.csv", index=False)
    return res
