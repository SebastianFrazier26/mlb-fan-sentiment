"""Per-game values and per-fan-day sentiment for the reverse-direction study.

Spec: docs/superpowers/specs/2026-11-23-reverse-direction-design.md ("Building blocks").

Fan-day: a comment's `created_utc` in the fanbase's home time zone, shifted back by
`cutoff_hour` (05:00 default), so post-midnight discussion of a night game stays on the game's
day. MLB game-log dates are the local official date, so a game and its discussion share a
fan-day. r/baseball (league-wide) uses the mentioned player's team zone that week.

Game value is runs above average in the game: hitters wRAA from fixed linear weights against
the season's own league wOBA; pitchers ER-based runs saved against the season's league ERA.
Doubleheaders are summed into one player-day.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TIMEZONES_PATH = Path(__file__).with_name("team_timezones.csv")
REDDIT_SOURCE = "reddit_arctic"

# FanGraphs-style weights held fixed across seasons (2023-ish values); the league mean is
# recomputed per season, so only the relative event weights are borrowed.
WOBA_WEIGHTS = {
    "ubb": 0.696,
    "hbp": 0.726,
    "single": 0.883,
    "double": 1.244,
    "triple": 1.569,
    "hr": 2.004,
}
WOBA_SCALE = 1.204

HIT_KEYS = {
    "plateAppearances": "pa",
    "atBats": "ab",
    "hits": "h",
    "doubles": "double",
    "triples": "triple",
    "homeRuns": "hr",
    "baseOnBalls": "bb",
    "intentionalWalks": "ibb",
    "hitByPitch": "hbp",
    "sacFlies": "sf",
    "strikeOuts": "k",
}
PITCH_KEYS = {
    "outs": "outs",
    "battersFaced": "bf",
    "earnedRuns": "er",
    "homeRuns": "hr",
    "strikeOuts": "k",
    "baseOnBalls": "bb",
    "hitByPitch": "hbp",
    "hits": "h",
}
GROUP_FILES = {"hitting": HIT_KEYS, "pitching": PITCH_KEYS}


def load_team_timezones(path: Path = TIMEZONES_PATH) -> dict[int, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["team_id"]): r["tz"] for r in csv.DictReader(f)}


def fan_day(created_utc: pd.Series, tz: pd.Series, cutoff_hour: int = 5) -> pd.Series:
    """Local fan-day (datetime64, midnight) per row; rows with a missing zone get NaT."""
    ts = pd.to_datetime(created_utc.astype("int64"), unit="s", utc=True)
    out = pd.Series(pd.NaT, index=created_utc.index, dtype="datetime64[ns]")
    shift = pd.Timedelta(hours=cutoff_hour)
    for zone, idx in tz.dropna().groupby(tz.dropna()).groups.items():
        # Shift wall-clock time, not absolute time: on a DST-change night the two differ.
        local = ts.loc[idx].dt.tz_convert(zone).dt.tz_localize(None) - shift
        out.loc[idx] = local.dt.normalize().astype("datetime64[ns]")
    return out


def read_game_logs(gamelog_dir: Path) -> pd.DataFrame:
    """Regular-season hitting and pitching splits -> one row per player, date, group (summed)."""
    rows = []
    for group, keys in GROUP_FILES.items():
        for path in sorted(gamelog_dir.glob(f"*_{group}.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for block in data.get("stats", []):
                for sp in block.get("splits", []):
                    if sp.get("gameType", "R") != "R":
                        continue
                    st = sp.get("stat", {})
                    row = {
                        "player_id": str(sp["player"]["id"]),
                        "date": sp["date"],
                        "group": group,
                        "team_id": int(sp["team"]["id"]),
                        "team_win": bool(sp.get("isWin", False)),
                        "games": 1,
                    }
                    row.update({v: int(st.get(k, 0) or 0) for k, v in keys.items()})
                    rows.append(row)
    if not rows:
        raise ValueError(f"no game logs under {gamelog_dir}")
    df = pd.DataFrame(rows).fillna(0)
    df["date"] = pd.to_datetime(df["date"])
    num = [c for c in df.columns if c not in ("player_id", "date", "group", "team_id", "team_win")]
    agg = {c: "sum" for c in num} | {"team_id": "last", "team_win": "max"}
    return df.groupby(["player_id", "date", "group"], as_index=False).agg(agg)


def woba_parts(g: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(linear-weights numerator, wOBA denominator) per hitting row."""
    single = g["h"] - g["double"] - g["triple"] - g["hr"]
    w = WOBA_WEIGHTS
    num = (
        w["ubb"] * (g["bb"] - g["ibb"])
        + w["hbp"] * g["hbp"]
        + w["single"] * single
        + w["double"] * g["double"]
        + w["triple"] * g["triple"]
        + w["hr"] * g["hr"]
    )
    den = g["ab"] + g["bb"] - g["ibb"] + g["sf"] + g["hbp"]
    return num, den


def game_values(games: pd.DataFrame) -> pd.DataFrame:
    """Add `v` (runs above average in the game) and `n` (exposure: PA or batters faced)."""
    out = games.copy()
    out["v"] = np.nan
    out["n"] = np.nan
    hit = out["group"] == "hitting"
    num, den = woba_parts(out[hit])
    lg_woba = num.sum() / den.sum()
    out.loc[hit, "v"] = (num - lg_woba * den) / WOBA_SCALE
    out.loc[hit, "n"] = out.loc[hit, "pa"]
    pit = out["group"] == "pitching"
    lg_er_per_out = out.loc[pit, "er"].sum() / out.loc[pit, "outs"].sum()
    out.loc[pit, "v"] = lg_er_per_out * out.loc[pit, "outs"] - out.loc[pit, "er"]
    out.loc[pit, "n"] = out.loc[pit, "bf"]
    return out[out["n"] > 0].reset_index(drop=True)


@dataclass(frozen=True)
class EBHyper:
    m: float  # prior mean talent, runs per exposure
    tau2: float  # between-player talent variance
    sigma2: float  # per-exposure noise variance
    min_exposure: float

    @property
    def k(self) -> float:
        return self.sigma2 / self.tau2


def eb_hyper(games: pd.DataFrame, min_exposure: float = 50.0) -> EBHyper:
    """Method-of-moments talent prior for one group's player-games (`v`, `n`).

    Noise: within-player residuals of v around n * r_i, with the weighted-variance correction
    sum(n) - sum(n^2)/N per player. Talent: between-player variance of season rates minus the
    expected sampling part sigma^2 / N, floored to keep k finite."""
    tot = games.groupby("player_id").agg(V=("v", "sum"), N=("n", "sum"))
    tot = tot[tot["N"] >= min_exposure]
    g = games[games["player_id"].isin(tot.index)].join(tot, on="player_id")
    r = g["V"] / g["N"]
    resid2 = (g["v"] - g["n"] * r) ** 2
    dof = (g["n"] - g["n"] ** 2 / g["N"]).sum()
    sigma2 = float(resid2.sum() / dof)
    rate = tot["V"] / tot["N"]
    m = float(rate.mean())
    tau2 = float(((rate - m) ** 2).mean() - sigma2 * (1.0 / tot["N"]).mean())
    tau2 = max(tau2, 1e-3 * sigma2 / tot["N"].mean())
    return EBHyper(m=m, tau2=tau2, sigma2=sigma2, min_exposure=min_exposure)


def add_surprise(games: pd.DataFrame, hyper: EBHyper) -> pd.DataFrame:
    """Per player-game, in date order: pre-game exposure `N_pre`, EB talent `theta_pre`,
    surprise `s = v - n theta_pre` and talent update `u = s / (N_pre + n + k)`."""
    out = games.sort_values(["player_id", "date"]).copy()
    grp = out.groupby("player_id")
    out["N_pre"] = grp["n"].cumsum() - out["n"]
    v_pre = grp["v"].cumsum() - out["v"]
    k = hyper.k
    out["theta_pre"] = (v_pre + k * hyper.m) / (out["N_pre"] + k)
    out["s"] = out["v"] - out["n"] * out["theta_pre"]
    out["u"] = out["s"] / (out["N_pre"] + out["n"] + k)
    return out


def player_group(players: pd.DataFrame) -> pd.Series:
    """player_id -> 'pitching' / 'hitting' from players.csv `is_pitcher`."""
    is_p = players["is_pitcher"].astype(str).str.lower().isin(["true", "1"])
    return pd.Series(np.where(is_p, "pitching", "hitting"), index=players["player_id"].astype(str))


def mention_days(
    mentions: pd.DataFrame,
    player_teams: pd.DataFrame,
    subreddit_fanbase: dict[str, str],
    fanbase_teams: dict[str, frozenset[int]],
    cutoff_hour: int = 5,
    timezones: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Kept Reddit mentions -> mean sentiment per (fanbase, player_id, fan_day).

    mentions: text_id, subreddit, player_id, created_utc, compound (relevance-filtered).
    player_teams: player_id, team_id, week_start (weekly 40-man rosters).
    Team fanbases keep only their own players, judged by the roster of the fan-day's Monday
    week; league-wide subs use the player's team that week for the time zone and are dropped
    when the player is on no roster that week."""
    tz_of = timezones or load_team_timezones()
    df = mentions.copy()
    df["player_id"] = df["player_id"].astype(str)
    sub = df["subreddit"].astype(str).str.lower()
    df["fanbase"] = sub.map(subreddit_fanbase).fillna(df["subreddit"].astype(str))
    fb_team = {
        fb: next(iter(t)) for fb, t in fanbase_teams.items() if t is not None and len(t) == 1
    }
    home = df["fanbase"].str.lower().map(fb_team)
    pt = player_teams.assign(
        player_id=player_teams["player_id"].astype(str),
        week_start=pd.to_datetime(player_teams["week_start"]),
        team_id=player_teams["team_id"].astype(int),
    ).drop_duplicates(["player_id", "week_start"], keep="last")

    utc_day = pd.to_datetime(df["created_utc"].astype("int64"), unit="s").dt.normalize()
    approx_week = utc_day - pd.to_timedelta(utc_day.dt.weekday, unit="D")
    roster = (
        df[["player_id"]]
        .assign(week_start=approx_week)
        .merge(pt, on=["player_id", "week_start"], how="left")["team_id"]
    )
    zone_team = home.where(home.notna(), pd.Series(roster.to_numpy(), index=df.index))
    df["fan_day"] = fan_day(df["created_utc"], zone_team.map(tz_of), cutoff_hour)
    df = df[df["fan_day"].notna()]

    df["week_start"] = df["fan_day"] - pd.to_timedelta(df["fan_day"].dt.weekday, unit="D")
    df = df.merge(pt, on=["player_id", "week_start"], how="left")
    team_fb = df["fanbase"].str.lower().isin(fb_team)
    own = df["team_id"] == df["fanbase"].str.lower().map(fb_team)
    df = df[~team_fb | own]
    df = df[df["team_id"].notna()]

    g = df.groupby(["fanbase", "player_id", "fan_day"], sort=True)["compound"]
    out = g.agg(sent="mean", n_mentions="size").reset_index()
    team = df.groupby(["fanbase", "player_id", "fan_day"], sort=True)["team_id"].last()
    out["team_id"] = team.to_numpy().astype(int)
    return out


def daily_lags(
    games: pd.DataFrame, days: pd.DataFrame, cols: list[str], n_lags: int
) -> pd.DataFrame:
    """Attach `{col}_l{l}` (value of the player's game on fan_day - l, 0 if no game) and
    `played_l{l}` for l = 0..n_lags to each row of `days` (player_id, fan_day)."""
    g = games[["player_id", "date", *cols]].rename(columns={"date": "fan_day"})
    out = days.copy()
    for lag in range(n_lags + 1):
        shifted = g.assign(fan_day=g["fan_day"] + pd.Timedelta(days=lag))
        shifted = shifted.rename(columns={c: f"{c}_l{lag}" for c in cols})
        shifted[f"played_l{lag}"] = 1.0
        out = out.merge(shifted, on=["player_id", "fan_day"], how="left")
    fill = [f"{c}_l{lag}" for c in cols for lag in range(n_lags + 1)]
    fill += [f"played_l{lag}" for lag in range(n_lags + 1)]
    out[fill] = out[fill].fillna(0.0)
    return out


def weekly(days: pd.DataFrame, games: pd.DataFrame, cols: list[str], n_lags: int) -> pd.DataFrame:
    """Fan-week (Mon-Sun) panel: mention-weighted mean sentiment per unit-week, with the
    player's weekly sums of `cols` at week lags 0..n_lags (0 when no games)."""
    d = days.assign(week=days["fan_day"] - pd.to_timedelta(days["fan_day"].dt.weekday, unit="D"))
    d = d.assign(cs=d["sent"] * d["n_mentions"])
    w = d.groupby(["fanbase", "player_id", "week"], as_index=False).agg(
        cs=("cs", "sum"), n_mentions=("n_mentions", "sum")
    )
    w["sent"] = w["cs"] / w["n_mentions"]
    w = w.drop(columns="cs")
    gw = games.assign(week=games["date"] - pd.to_timedelta(games["date"].dt.weekday, unit="D"))
    gw = gw.groupby(["player_id", "week"], as_index=False)[[*cols, "n"]].sum()
    for lag in range(n_lags + 1):
        sh = gw.assign(week=gw["week"] + pd.Timedelta(weeks=lag))
        sh = sh.rename(columns={c: f"{c}_l{lag}" for c in [*cols, "n"]})
        w = w.merge(sh, on=["player_id", "week"], how="left")
    lagged = [f"{c}_l{lag}" for c in [*cols, "n"] for lag in range(n_lags + 1)]
    w[lagged] = w[lagged].fillna(0.0)
    return w
