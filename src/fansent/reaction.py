"""S1: how fan sentiment reacts to game results (reverse-direction study, exploratory).

Spec: docs/superpowers/specs/2026-11-23-reverse-direction-design.md (S1, analyses A1-A5).

Every model is within-unit (unit = fanbase x player) weighted least squares: unit fixed effects
absorbed by weighted demeaning, weights = mentions on the day/week, CR1 standard errors
clustered by player (a player appears in his team sub and in r/baseball, so the player, not
the unit, is the independent block).

S1c overreaction: sentiment on the player's EB talent update `u` and raw surprise `s` at lags
0..L. A fan tracking expected future value responds only through u, so B_s = sum of the s
lags is 0. OI = B_s / (B_s + B_u / (N_med + k)) is the share of the response to a one-run
surprise, at median pre-game exposure, that the talent update does not justify.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import daypanel as dp
from .multitest import benjamini_hochberg, holm
from .panel import load_fanbase_teams, load_team_subreddits
from .schedule import SCHEDULE_FILE
from .util import (
    PLAYER_TEAMS_FILE,
    PLAYERS_FILE,
    RELEVANCE_FILE,
    SENTIMENT_FILE,
    TEXTS_FILE,
)

EXPLORATORY_SEASON = 2026
FROZEN_MARK = "Status: FROZEN"
LEAGUE_FANBASE = "baseball"


def require_open_season(season: int, prereg: Path | None) -> None:
    """Integrity guard: only 2026 is exploratory; other seasons need a frozen pre-registration."""
    if season == EXPLORATORY_SEASON:
        return
    if prereg is None or not prereg.exists():
        raise SystemExit(
            f"season {season} is confirmatory: pass --prereg with the frozen pre-registration"
        )
    if FROZEN_MARK not in prereg.read_text(encoding="utf-8"):
        raise SystemExit(f"{prereg} is not frozen (needs a line '{FROZEN_MARK}')")


@dataclass
class WLSResult:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    vcov: np.ndarray
    n_obs: int
    n_units: int
    n_clusters: int

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"term": self.names, "coef": self.coef, "se": self.se})


def _demean(a: np.ndarray, unit: np.ndarray, w: np.ndarray) -> np.ndarray:
    n_units = int(unit.max()) + 1
    wsum = np.bincount(unit, weights=w, minlength=n_units)
    if a.ndim == 1:
        return a - (np.bincount(unit, weights=w * a, minlength=n_units) / wsum)[unit]
    means = (
        np.stack(
            [np.bincount(unit, weights=w * a[:, j], minlength=n_units) for j in range(a.shape[1])],
            axis=1,
        )
        / wsum[:, None]
    )
    return a - means[unit]


def within_wls(
    y: np.ndarray,
    X: np.ndarray,
    unit: np.ndarray,
    w: np.ndarray,
    cluster: np.ndarray,
    names: list[str],
    absorb: bool = True,
) -> WLSResult:
    """WLS of y on X with unit FE absorbed (or a plain intercept when absorb=False), CR1 SEs."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    unit = np.unique(np.asarray(unit), return_inverse=True)[1]
    cl = np.unique(np.asarray(cluster), return_inverse=True)[1]
    if absorb:
        yd, Xd = _demean(y, unit, w), _demean(X, unit, w)
        n_fe = int(unit.max()) + 1
    else:
        Xd = np.column_stack([np.ones(len(y)), X])
        yd = y
        names = ["const", *names]
        n_fe = 0
    sw = np.sqrt(w)
    Xw, yw = Xd * sw[:, None], yd * sw
    xtx_inv = np.linalg.pinv(Xw.T @ Xw)
    beta = xtx_inv @ (Xw.T @ yw)
    resid = yw - Xw @ beta
    G = int(cl.max()) + 1
    scores = np.zeros((G, Xw.shape[1]))
    np.add.at(scores, cl, Xw * resid[:, None])
    n, p = Xw.shape
    adj = G / (G - 1) * (n - 1) / max(n - p - n_fe, 1)
    vcov = adj * xtx_inv @ (scores.T @ scores) @ xtx_inv
    return WLSResult(names, beta, np.sqrt(np.clip(np.diag(vcov), 0, None)), vcov, n, n_fe, G)


def _lag_names(cols: list[str], n_lags: int) -> list[str]:
    return [f"{c}_l{lag}" for c in cols for lag in range(n_lags + 1)]


def _sum_stat(res: WLSResult, terms: list[str]) -> tuple[float, float]:
    idx = [res.names.index(t) for t in terms]
    a = np.zeros(len(res.names))
    a[idx] = 1.0
    return float(a @ res.coef), float(np.sqrt(a @ res.vcov @ a))


def _dow(days: pd.Series) -> np.ndarray:
    d = days.dt.weekday.to_numpy()
    return np.stack([(d == k).astype(float) for k in range(1, 7)], axis=1)


def fit_daily_dl(panel: pd.DataFrame, n_lags: int, value: str = "vz") -> WLSResult:
    """S1a: y_{j,d} on value lags 0..L, played-flags 0..L and day of week."""
    cols = _lag_names([value], n_lags) + [f"played_l{lag}" for lag in range(n_lags + 1)]
    X = np.column_stack([panel[cols].to_numpy(float), _dow(panel["fan_day"])])
    names = cols + [f"dow{k}" for k in range(1, 7)]
    return within_wls(
        panel["sent"].to_numpy(),
        X,
        panel["unit"].to_numpy(),
        panel["n_mentions"].to_numpy(),
        panel["player_id"].to_numpy(),
        names,
    )


def fit_weekly_dl(wk: pd.DataFrame, n_lags: int, value: str = "vz") -> WLSResult:
    """S1b: fan-week sentiment on weekly summed value lags 0..L and exposure lags."""
    cols = _lag_names([value, "n"], n_lags)
    return within_wls(
        wk["sent"].to_numpy(),
        wk[cols].to_numpy(float),
        wk["unit"].to_numpy(),
        wk["n_mentions"].to_numpy(),
        wk["player_id"].to_numpy(),
        cols,
    )


def fit_overreaction(panel: pd.DataFrame, n_lags: int, control_win: bool = True) -> WLSResult:
    """S1c: y_{j,d} on u and s lags 0..L plus played-flags and day of week. control_win adds
    the team-won flags at each lag (owner decision 2026-11-23: the primary OI is net of the
    team result; the uncontrolled one is secondary)."""
    cols = _lag_names(["u", "s"], n_lags) + [f"played_l{lag}" for lag in range(n_lags + 1)]
    if control_win:
        cols += [f"team_win_l{lag}" for lag in range(n_lags + 1)]
    X = np.column_stack([panel[cols].to_numpy(float), _dow(panel["fan_day"])])
    return within_wls(
        panel["sent"].to_numpy(),
        X,
        panel["unit"].to_numpy(),
        panel["n_mentions"].to_numpy(),
        panel["player_id"].to_numpy(),
        cols + [f"dow{k}" for k in range(1, 7)],
    )


def oi_from(res: WLSResult, n_lags: int, scale: float) -> tuple[float, float, float]:
    """(OI, B_u, B_s); scale = N_med + k converts a one-run surprise into its talent update."""
    b_u, _ = _sum_stat(res, [f"u_l{lag}" for lag in range(n_lags + 1)])
    b_s, _ = _sum_stat(res, [f"s_l{lag}" for lag in range(n_lags + 1)])
    denom = b_s + b_u / scale
    return (b_s / denom if denom != 0 else float("nan")), b_u, b_s


def block_bootstrap_oi(
    panel: pd.DataFrame,
    n_lags: int,
    scale: float,
    n_boot: int,
    seed: int,
    control_win: bool = True,
) -> np.ndarray:
    """OI over player-block bootstrap resamples (the whole player history in every unit)."""
    rng = np.random.default_rng(seed)
    players = panel["player_id"].unique()
    by_player = {p: idx for p, idx in panel.groupby("player_id").indices.items()}
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(players, size=len(players), replace=True)
        parts, units = [], []
        for rep, p in enumerate(pick):
            idx = by_player[p]
            parts.append(idx)
            units.append(np.full(len(idx), rep))
        rows = np.concatenate(parts)
        sub = panel.iloc[rows].copy()
        rep_id = np.concatenate(units)
        sub["unit"] = sub["unit"].astype(str).to_numpy() + "#" + rep_id.astype(str)
        sub["player_id"] = sub["player_id"].astype(str).to_numpy() + "#" + rep_id.astype(str)
        out[b] = oi_from(fit_overreaction(sub, n_lags, control_win), n_lags, scale)[0]
    return out


def load_inputs(data_dir: Path) -> dict[str, pd.DataFrame]:
    sent = pd.read_csv(
        data_dir / SENTIMENT_FILE,
        usecols=["text_id", "source", "subreddit", "player_id", "compound"],
        dtype={"text_id": str, "player_id": str},
    )
    sent = sent[sent["source"] == dp.REDDIT_SOURCE]
    rel_path = data_dir / RELEVANCE_FILE
    if rel_path.exists():
        rel = pd.read_csv(
            rel_path,
            usecols=["text_id", "player_id", "kept"],
            dtype={"text_id": str, "player_id": str},
        )
        sent = sent.merge(rel, on=["text_id", "player_id"], how="left")
        sent = sent[sent["kept"].fillna(1) == 1].drop(columns="kept")
    ts = pd.read_csv(
        data_dir / TEXTS_FILE,
        usecols=["text_id", "player_id", "created_utc"],
        dtype={"text_id": str, "player_id": str},
    ).drop_duplicates(["text_id", "player_id"])
    mentions = sent.merge(ts, on=["text_id", "player_id"], how="inner")
    return {
        "mentions": mentions,
        "players": pd.read_csv(data_dir / PLAYERS_FILE, dtype={"player_id": str}),
        "player_teams": pd.read_csv(data_dir / PLAYER_TEAMS_FILE, dtype={"player_id": str}),
    }


def build(
    data_dir: Path, season: int, n_lags: int, cutoff_hour: int, drop_pregame: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dp.EBHyper], dict[str, float]]:
    """(day panel, week panel, EB hyperparameters and median pre-game exposure per group)."""
    inp = load_inputs(data_dir)
    games = dp.game_values(dp.read_game_logs(data_dir / "raw" / "mlb" / "gamelogs" / str(season)))
    group = dp.player_group(inp["players"])
    games = games[games["group"] == games["player_id"].map(group)]
    hypers, parts = {}, []
    for gname, g in games.groupby("group"):
        hypers[gname] = dp.eb_hyper(g)
        g = dp.add_surprise(g, hypers[gname])
        g["vz"] = g["v"] / g["v"].std(ddof=0)
        g["team_win"] = g["team_win"].astype(float)
        parts.append(g)
    games = pd.concat(parts, ignore_index=True)
    pitch_times = None
    if drop_pregame:
        pitch_times = dp.first_pitch(pd.read_csv(data_dir / SCHEDULE_FILE))
    days = dp.mention_days(
        inp["mentions"],
        inp["player_teams"],
        load_team_subreddits(),
        load_fanbase_teams(),
        cutoff_hour,
        pitch_times=pitch_times,
    )
    days["group"] = days["player_id"].map(group)
    days = days[days["group"].notna()]
    panel = dp.daily_lags(games, days, ["vz", "u", "s", "team_win"], n_lags)
    panel["unit"] = panel["fanbase"] + "|" + panel["player_id"]
    wk = dp.weekly(days, games, ["vz"], 3)
    wk["group"] = wk["player_id"].map(group)
    wk["unit"] = wk["fanbase"] + "|" + wk["player_id"]
    n_med = {str(g): float(x["N_pre"].median()) for g, x in games.groupby("group")}
    return panel, wk, hypers, n_med


def run(
    data_dir: Path,
    season: int = EXPLORATORY_SEASON,
    prereg: Path | None = None,
    n_lags: int = 7,
    cutoff_hour: int = 5,
    n_boot: int = 199,
    seed: int = 0,
    out_dir: Path | None = None,
    drop_pregame: bool = False,
) -> pd.DataFrame:
    require_open_season(season, prereg)
    panel, wk, hypers, n_med = build(data_dir, season, n_lags, cutoff_hour, drop_pregame)
    out_dir = out_dir or data_dir / "runs" / f"{dt.datetime.now(dt.UTC).date()}-react"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, coef_tables = [], []

    def add(aid: str, label: str, est: float, se: float, n_obs: int, extra: dict | None = None):
        z = est / se if se > 0 else float("nan")
        rows.append(
            {
                "analysis": aid,
                "label": label,
                "estimate": est,
                "se": se,
                "z": z,
                "p_test": float(2 * norm.sf(abs(z))),
                "n_obs": n_obs,
                **(extra or {}),
            }
        )

    team = panel["fanbase"] != LEAGUE_FANBASE
    for aid, grp in (("A1", "hitting"), ("A2", "pitching")):
        sub = panel[team & (panel["group"] == grp)]
        res = fit_daily_dl(sub, n_lags)
        coef_tables.append(res.table().assign(analysis=aid))
        b0 = res.names.index("vz_l0")
        add(aid, f"{grp} b_0", float(res.coef[b0]), float(res.se[b0]), res.n_obs)
        add(
            aid,
            f"{grp} B_{n_lags}",
            *_sum_stat(res, [f"vz_l{lag}" for lag in range(n_lags + 1)]),
            res.n_obs,
        )
    for grp in ("hitting", "pitching"):
        sub = wk[(wk["fanbase"] != LEAGUE_FANBASE) & (wk["group"] == grp)]
        res = fit_weekly_dl(sub, 3)
        coef_tables.append(res.table().assign(analysis=f"A3-{grp}"))
        add("A3", f"{grp} B_3", *_sum_stat(res, [f"vz_l{lag}" for lag in range(4)]), res.n_obs)
    oi_specs = [
        ("A4", team, "team subs", True),
        ("A5", ~team, "r/baseball", True),
        ("A4u", team, "team subs, no win control", False),
        ("A5u", ~team, "r/baseball, no win control", False),
    ]
    for aid, mask, label, control in oi_specs:
        for grp in ("hitting", "pitching"):
            sub = panel[mask & (panel["group"] == grp)]
            h = hypers[grp]
            scale = n_med[grp] + h.k
            res = fit_overreaction(sub, n_lags, control)
            coef_tables.append(res.table().assign(analysis=f"{aid}-{grp}"))
            oi, b_u, b_s = oi_from(res, n_lags, scale)
            boot = block_bootstrap_oi(sub, n_lags, scale, n_boot, seed, control) if n_boot else None
            se = float(np.nanstd(boot, ddof=1)) if boot is not None else float("nan")
            lo, hi = (
                np.nanquantile(boot, [0.025, 0.975])
                if boot is not None
                else (float("nan"), float("nan"))
            )
            # The null of interest is OI = 0 (rational), tested through B_s and its CR1 SE
            # rather than through the bootstrap SE of a ratio.
            _, bs_se = _sum_stat(res, [f"s_l{lag}" for lag in range(n_lags + 1)])
            add(
                aid,
                f"{label} {grp} OI",
                oi,
                se,
                res.n_obs,
                {
                    "oi_lo95": lo,
                    "oi_hi95": hi,
                    "B_u": b_u,
                    "B_s": b_s,
                    "B_s_se": bs_se,
                    "k": h.k,
                    "N_med": n_med[grp],
                },
            )
            rows[-1]["z"] = b_s / bs_se
            rows[-1]["p_test"] = float(2 * norm.sf(abs(b_s / bs_se)))
    res_df = pd.DataFrame(rows)
    # Uncontrolled OI rows are the secondary analysis: their own BH family, so they neither
    # dilute nor borrow from the primary F1 tests.
    res_df["family"] = np.where(res_df["analysis"].str.endswith("u"), "F1s", "F1")
    res_df["q_bh"] = np.nan
    res_df["p_holm"] = np.nan
    for idx in res_df.groupby("family").groups.values():
        p = res_df.loc[idx, "p_test"].to_numpy(float)
        res_df.loc[idx, "q_bh"] = benjamini_hochberg(p)
        res_df.loc[idx, "p_holm"] = holm(p)
    res_df.to_csv(out_dir / "s1_results.csv", index=False)
    pd.concat(coef_tables).to_csv(out_dir / "s1_coefficients.csv", index=False)
    hyper_rows = [
        {"group": g, "m": h.m, "tau2": h.tau2, "sigma2": h.sigma2, "k": h.k, "N_med": n_med[g]}
        for g, h in hypers.items()
    ]
    pd.DataFrame(hyper_rows).to_csv(out_dir / "s1_eb_hyper.csv", index=False)
    meta = {
        "season": season,
        "n_lags": n_lags,
        "cutoff_hour": cutoff_hour,
        "drop_pregame": drop_pregame,
        "n_boot": n_boot,
        "seed": seed,
        "day_rows": len(panel),
        "week_rows": len(wk),
        "units": int(panel["unit"].nunique()),
        "players": int(panel["player_id"].nunique()),
    }
    pd.Series(meta).to_csv(out_dir / "s1_meta.csv", header=False)
    return res_df
