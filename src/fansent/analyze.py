"""Lagged correlation: does fan sentiment in week t predict performance in
week t+1, beyond what week t performance already predicts?

For each (fanbase, player) we residualize both sentiment_t and perf_{t+1}
against perf_t and correlate the residuals (a partial correlation). Positive r
means that fanbase's mood carries real predictive signal ("the fans are
right"); negative r means fans overreact and performance mean-reverts against
their mood. Per-fanbase scores aggregate player r values via Fisher z.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from .util import DATA_DIR, NEWS_TONE_CSV, SENTIMENT_CSV, STATS_CSV, week_start

MIN_PA_WEEK = 10
MIN_IP_WEEK = 3.0
PLAYERS_OUT = DATA_DIR / "analysis_players.csv"
FANBASES_OUT = DATA_DIR / "analysis_fanbases.csv"


def load_perf() -> pd.DataFrame:
    df = pd.read_csv(STATS_CSV)
    df = df[df["metric"].isin(["OPS", "ERA"])]
    df["week_start"] = pd.to_datetime(df["week_start"])
    # Orient so higher == better for every player (ERA gets negated).
    df["perf"] = np.where(df["higher_is_better"], df["value"], -df["value"])
    ok = np.where(df["metric"] == "OPS", df["weight"] >= MIN_PA_WEEK, df["weight"] >= MIN_IP_WEEK)
    return df[ok][["player_id", "full_name", "team", "week_start", "perf"]]


def load_sentiment(min_mentions: int) -> pd.DataFrame:
    """Weekly mean sentiment per (fanbase, player). Fanbase = subreddit, plus
    a 'news' pseudo-fanbase from GDELT tone if news_tone.csv exists."""
    frames = []
    if SENTIMENT_CSV.exists():
        s = pd.read_csv(SENTIMENT_CSV)
        s["week_start"] = pd.to_datetime(s["date"]).dt.date.map(week_start)
        g = (
            s.groupby(["subreddit", "player_id", "week_start"])["compound"]
            .agg(sent="mean", mentions="size")
            .reset_index()
            .rename(columns={"subreddit": "fanbase"})
        )
        frames.append(g[g["mentions"] >= min_mentions])
    if NEWS_TONE_CSV.exists():
        n = pd.read_csv(NEWS_TONE_CSV)
        n["week_start"] = pd.to_datetime(n["date"]).dt.date.map(week_start)
        g = (
            n.groupby(["player_id", "week_start"])["tone"]
            .agg(sent="mean", mentions="size")
            .reset_index()
        )
        g["fanbase"] = "news"
        frames.append(g)
    if not frames:
        raise SystemExit("No sentiment data found — run `fansent score` and/or `fansent news`.")
    out = pd.concat(frames, ignore_index=True)
    out["week_start"] = pd.to_datetime(out["week_start"])
    return out


def partial_corr(sent: np.ndarray, perf_next: np.ndarray, perf_now: np.ndarray):
    """Correlate sent and perf_next after removing what perf_now explains."""
    X = np.column_stack([np.ones_like(perf_now), perf_now])
    res_sent = sent - X @ np.linalg.lstsq(X, sent, rcond=None)[0]
    res_next = perf_next - X @ np.linalg.lstsq(X, perf_next, rcond=None)[0]
    if res_sent.std() < 1e-12 or res_next.std() < 1e-12:
        return None
    return sps.pearsonr(res_sent, res_next)


def run(min_weeks: int = 8, min_mentions: int = 3) -> None:
    perf = load_perf()
    sent = load_sentiment(min_mentions)

    perf_next = perf.copy()
    perf_next["week_start"] -= pd.Timedelta(weeks=1)  # join perf_{t+1} onto week t
    perf_next = perf_next.rename(columns={"perf": "perf_next"})[
        ["player_id", "week_start", "perf_next"]
    ]

    panel = (
        sent.merge(perf, on=["player_id", "week_start"])
        .merge(perf_next, on=["player_id", "week_start"])
        .dropna(subset=["sent", "perf", "perf_next"])
    )

    player_rows = []
    for (fanbase, pid), g in panel.groupby(["fanbase", "player_id"]):
        if len(g) < min_weeks:
            continue
        result = partial_corr(g["sent"].to_numpy(), g["perf_next"].to_numpy(), g["perf"].to_numpy())
        if result is None:
            continue
        player_rows.append(
            {
                "fanbase": fanbase,
                "player_id": pid,
                "full_name": g["full_name"].iloc[0],
                "team": g["team"].iloc[0],
                "weeks": len(g),
                "r_partial": round(result.statistic, 4),
                "p_value": round(result.pvalue, 4),
            }
        )

    if not player_rows:
        raise SystemExit(
            f"No (fanbase, player) pair had >= {min_weeks} usable weeks — "
            "collect more data or lower --min-weeks/--min-mentions."
        )

    players_df = pd.DataFrame(player_rows).sort_values(["fanbase", "r_partial"], ascending=False)
    players_df.to_csv(PLAYERS_OUT, index=False)

    fanbase_rows = []
    for fanbase, g in players_df.groupby("fanbase"):
        z = np.arctanh(g["r_partial"].clip(-0.999, 0.999))
        fanbase_rows.append(
            {
                "fanbase": fanbase,
                "players": len(g),
                "mean_r": round(math.tanh(z.mean()), 4),
                "pct_positive": round(100 * (g["r_partial"] > 0).mean(), 1),
                "median_weeks": g["weeks"].median(),
            }
        )
    fanbases_df = pd.DataFrame(fanbase_rows).sort_values("mean_r", ascending=False)
    fanbases_df.to_csv(FANBASES_OUT, index=False)

    print("\n=== Which fanbase is most 'right' about its players? ===")
    print(fanbases_df.to_string(index=False))
    print(f"\nPer-player detail: {PLAYERS_OUT}")
    print("mean_r > 0: fan mood adds predictive signal beyond recent stats.")
    print("mean_r < 0: fans overreact; performance mean-reverts against their mood.")
