"""Build the model panel: weekly (or monthly) sentiment joined to next-period performance.

One row per (metric, fanbase, player, period t): sentiment and performance at t, the
performance outcome at t+1 (consecutive periods required). Values stay raw; `higher_is_better`
rides along and `oriented()` applies the sign only when a model is fed, so effects read as
"better performance" for every metric.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .composite import DEFAULT_COMPOSITE_WEIGHTS, composite_index

NEWS_SOURCE = "gdelt_headline"
NEWS_FANBASE = "news"
INHERITED = "reply_inherited"

TEAM_SUBREDDITS_CSV = Path(__file__).with_name("team_subreddits.csv")


def load_team_subreddits(path: Path = TEAM_SUBREDDITS_CSV) -> dict[str, frozenset[int] | None]:
    """subreddit (lower-cased: Reddit names are case-insensitive) -> {team_id}."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["subreddit"].lower(): frozenset({int(r["team_id"])}) for r in csv.DictReader(f)}


# Weekly weight units as written by stats.py: PA (OPS), AB (AVG), IP (ERA, WHIP), chances,
# SB+CS attempts. Unconfirmed assumptions, not tuned values.
DEFAULT_MIN_WEIGHT: dict[str, float] = {
    "OPS": 10,
    "AVG": 10,
    "ERA": 3.0,
    "WHIP": 3.0,
    "FLD_PCT": 3,
    "ERRORS": 3,
    "CS_PCT": 2,
}

PANEL_COLS = [
    "metric",
    "fanbase",
    "player_id",
    "full_name",
    "period",
    "sent",
    "n_mentions",
    "y_next",
    "perf",
    "std",
    "weight",
    "higher_is_better",
]


@dataclass(frozen=True)
class PanelConfig:
    min_weight: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_MIN_WEIGHT))
    min_mentions: int = 3
    # Team fanbases only count their own team's players (decision 2026-10-13). Fanbase -> allowed
    # team ids, keys lower-case; a fanbase absent from the map or mapped to None is league-wide
    # (baseball, news) rather than an error, so non-team sources and new subs keep working. {}
    # disables the filter (`--all-teams`).
    fanbase_teams: Mapping[str, frozenset[int] | None] = field(default_factory=load_team_subreddits)
    min_weeks: int = 8
    min_units_per_fanbase: int = 3
    # Reply comments inherit the player from the parent comment; 1.0 treats them like explicit
    # mentions, 0 drops them.
    reply_inherited_weight: float = 1.0
    composite_weights: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {r: dict(w) for r, w in DEFAULT_COMPOSITE_WEIGHTS.items()}
    )
    role_min_pa: float = 20.0
    # OAA monthly panel
    min_month_mentions: int = 10
    min_months: int = 3
    min_oaa_attempts: float = 10.0


def _day(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=False).dt.tz_localize(None).dt.normalize()


def fanbase_of(source: pd.Series, subreddit: pd.Series) -> pd.Series:
    sub = subreddit.fillna("").astype(str).str.strip()
    named = sub.where(sub != "", source.astype(str))
    return named.where(source != NEWS_SOURCE, NEWS_FANBASE)


def filter_own_team(
    scores: pd.DataFrame, cfg: PanelConfig, player_teams: Mapping[str, int] | None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows where a team fanbase mentions a player outside its team(s).

    player_teams: player_id -> team_id from players.csv (current roster, so a traded player
    leaves his old team's fanbase). Players missing from the map count as off-team. Returns
    (kept rows, dropped row count per fanbase)."""
    fanbase = fanbase_of(scores["source"], scores["subreddit"])
    allowed = fanbase.str.lower().map(dict(cfg.fanbase_teams))
    restricted = allowed.notna()
    if not restricted.any():
        return scores, {}
    if player_teams is None:
        raise ValueError("player_teams is required when a team fanbase filter applies")
    team = scores["player_id"].astype(str).map(dict(player_teams))
    own = pd.Series(
        [
            pd.notna(t) and t in a if isinstance(a, frozenset) else False
            for t, a in zip(team, allowed, strict=True)
        ],
        index=scores.index,
    )
    drop = restricted & ~own
    counts = fanbase[drop].value_counts()
    return scores[~drop], {str(k): int(v) for k, v in counts.items()}


def attach_attribution(
    scores: pd.DataFrame, sidecars: list[pd.DataFrame], inherited_weight: float
) -> pd.DataFrame:
    """Add `w` per text-player row. Sidecars (text_id, player_id, attribution) may be empty;
    rows without a sidecar entry count as explicit mentions."""
    out = scores.copy()
    out["text_id"] = out["text_id"].astype(str)
    out["player_id"] = out["player_id"].astype(str)
    attr = [
        s[["text_id", "player_id", "attribution"]]
        .astype(str)
        .drop_duplicates(["text_id", "player_id"])
        for s in sidecars
        if s is not None and len(s)
    ]
    if attr:
        out = out.merge(
            pd.concat(attr).drop_duplicates(["text_id", "player_id"]),
            on=["text_id", "player_id"],
            how="left",
        )
    else:
        out["attribution"] = np.nan
    out["w"] = np.where(out["attribution"] == INHERITED, float(inherited_weight), 1.0)
    return out


def _aggregate(rows: pd.DataFrame, period: pd.Series) -> pd.DataFrame:
    df = rows.assign(
        period=period,
        wc=rows["compound"].astype(float) * rows["w"],
        pos=(rows["w"] > 0).astype(int),
    )
    g = df.groupby(["fanbase", "player_id", "period"], sort=True)
    out = pd.DataFrame(
        {"wsum": g["w"].sum(), "csum": g["wc"].sum(), "n_mentions": g["pos"].sum()}
    ).reset_index()
    out = out[out["wsum"] > 0]
    out["sent"] = out["csum"] / out["wsum"]
    return out[["fanbase", "player_id", "period", "sent", "n_mentions"]]


def sentiment_periods(
    scores: pd.DataFrame, sidecars: list[pd.DataFrame], cfg: PanelConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(weekly, monthly) mean VADER compound per (fanbase, player, period).

    scores: sentiment.csv columns text_id, source, subreddit, date, player_id, compound.
    Weekly periods are Mon-Sun, labelled by Monday (matches stats.csv `week_start`); monthly
    periods by first of month. Mean is weighted by attribution (see PanelConfig)."""
    rows = attach_attribution(scores, sidecars, cfg.reply_inherited_weight)
    rows["fanbase"] = fanbase_of(rows["source"], rows["subreddit"])
    day = _day(rows["date"])
    weekly = _aggregate(rows, day - pd.to_timedelta(day.dt.weekday, unit="D"))
    monthly = _aggregate(rows, day.dt.to_period("M").dt.to_timestamp())
    return weekly, monthly


def load_stats(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["period"] = pd.to_datetime(df["week_start"])
    df["higher_is_better"] = df["higher_is_better"].astype(str).str.lower().eq("true")
    return df


def eligible_stats(stats: pd.DataFrame, cfg: PanelConfig) -> pd.DataFrame:
    floor = stats["metric"].map(dict(cfg.min_weight))
    return stats[floor.notna() & (stats["weight"] >= floor)]


def pair_with_next(
    perf: pd.DataFrame, step_back: Callable[[pd.Series], pd.Series], keys: list[str]
) -> pd.DataFrame:
    """Attach the outcome at t+1 to each row at t; rows with no consecutive successor drop."""
    nxt = perf[[*keys, "period", "value", "weight"]].rename(
        columns={"value": "y_next", "weight": "weight_next"}
    )
    nxt["period"] = step_back(nxt["period"])
    out = perf.merge(nxt, on=[*keys, "period"], how="inner")
    return (
        out.rename(columns={"value": "perf", "std_value": "std"})
        .drop(columns="weight")
        .rename(columns={"weight_next": "weight"})
    )


def _finish(
    paired: pd.DataFrame, sent: pd.DataFrame, min_mentions: int, min_obs: int, cfg: PanelConfig
) -> pd.DataFrame:
    df = paired.merge(sent, on=["player_id", "period"], how="inner")
    df = df[df["n_mentions"] >= min_mentions]
    size = df.groupby(["metric", "fanbase", "player_id"])["period"].transform("size")
    df = df[size >= min_obs]
    units = df.groupby(["metric", "fanbase"])["player_id"].transform("nunique")
    df = df[units >= cfg.min_units_per_fanbase]
    return df[PANEL_COLS].reset_index(drop=True)


def week_back(p: pd.Series) -> pd.Series:
    return p - pd.Timedelta(days=7)


def month_back(p: pd.Series) -> pd.Series:
    return p - pd.offsets.MonthBegin(1)


def build_weekly_panel(
    stats: pd.DataFrame,
    weekly_sent: pd.DataFrame,
    cfg: PanelConfig,
    metrics: list[str] | None = None,
    include_composite: bool = True,
) -> pd.DataFrame:
    """Long panel over every metric present in `stats` (plus COMPOSITE) that has a threshold."""
    st = load_stats(stats)
    elig = eligible_stats(st, cfg)
    parts = [elig]
    if include_composite:
        comp = composite_index(elig, cfg.composite_weights, cfg.role_min_pa, st)
        if len(comp):
            parts.append(comp)
    perf = pd.concat(parts, ignore_index=True)
    if metrics is not None:
        perf = perf[perf["metric"].isin(metrics)]
    perf = perf[
        [
            "player_id",
            "full_name",
            "period",
            "metric",
            "value",
            "std_value",
            "weight",
            "higher_is_better",
        ]
    ]
    paired = pair_with_next(perf, week_back, ["player_id", "metric"])
    return _finish(paired, weekly_sent, cfg.min_mentions, cfg.min_weeks, cfg)


def build_oaa_panel(
    oaa: pd.DataFrame, monthly_sent: pd.DataFrame, cfg: PanelConfig
) -> pd.DataFrame:
    """Monthly sentiment vs next-month OAA; `std` is season-to-date cumulative OAA."""
    df = oaa.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["period"] = pd.to_datetime(df["month"])
    df = df[df["attempts"] >= cfg.min_oaa_attempts].sort_values(["player_id", "period"])
    df["std_value"] = df.groupby(["player_id", df["period"].dt.year])["oaa"].cumsum()
    df = df.rename(columns={"oaa": "value", "attempts": "weight"})
    df["metric"] = "OAA"
    df["higher_is_better"] = True
    df = df[
        [
            "player_id",
            "full_name",
            "period",
            "metric",
            "value",
            "std_value",
            "weight",
            "higher_is_better",
        ]
    ]
    paired = pair_with_next(df, month_back, ["player_id", "metric"])
    return _finish(paired, monthly_sent, cfg.min_month_mentions, cfg.min_months, cfg)


def oriented(panel: pd.DataFrame) -> pd.DataFrame:
    """Flip y_next/perf/std for lower-is-better metrics so higher always means better."""
    sign = np.where(panel["higher_is_better"], 1.0, -1.0)
    out = panel.copy()
    for c in ("y_next", "perf", "std"):
        out[c] = out[c] * sign
    return out
