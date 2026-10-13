"""Composite per-player performance index.

Each constituent metric is oriented (higher = better) and z-scored pooled over all eligible
player-weeks of that metric; a player-week's composite is the weighted mean of the z-scores
available that week. The composite's season-to-date control is built the same way from
`std_value` (z-scored against its own pooled distribution, since cumulative counts such as
ERRORS live on a different scale than weekly ones). A week needs at least one anchor metric
(hitting for hitters, pitching for pitchers) so fielding alone never makes a composite.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

DEFAULT_COMPOSITE_WEIGHTS: dict[str, dict[str, float]] = {
    "hitter": {"OPS": 0.375, "AVG": 0.375, "FLD_PCT": 0.125, "ERRORS": 0.125},
    "pitcher": {"ERA": 0.5, "WHIP": 0.5},
}
ANCHORS = {"hitter": ("OPS", "AVG"), "pitcher": ("ERA", "WHIP")}


def player_roles(stats: pd.DataFrame, min_pa: float) -> pd.Series:
    """'hitter' if the player ever reached `min_pa` season PA; else 'pitcher' if he has pitching
    rows. Position players mopping up an inning still have the PA, pitchers have ~none."""
    ops = stats[stats["metric"] == "OPS"].groupby("player_id")["std_weight"].max()
    pitches = set(stats.loc[stats["group"] == "pitching", "player_id"])
    roles = {}
    for pid in stats["player_id"].unique():
        if ops.get(pid, 0) >= min_pa:
            roles[pid] = "hitter"
        elif pid in pitches:
            roles[pid] = "pitcher"
    return pd.Series(roles, name="role")


def composite_index(
    eligible: pd.DataFrame,
    weights: Mapping[str, Mapping[str, float]],
    role_min_pa: float,
    all_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Rows in the long stats format with metric 'COMPOSITE' (value/std_value oriented, weight NaN)."""
    roles = player_roles(all_stats, role_min_pa)
    used = {m for w in weights.values() for m, x in w.items() if x > 0}
    df = eligible[eligible["metric"].isin(used)].copy()
    df["role"] = df["player_id"].map(roles)
    df = df.dropna(subset=["role"])
    sign = df["higher_is_better"].map({True: 1.0, False: -1.0})
    for src, dst in (("value", "z"), ("std_value", "zs")):
        o = df[src] * sign
        grp = o.groupby(df["metric"])
        sd = grp.transform("std").replace(0, float("nan"))
        df[dst] = (o - grp.transform("mean")) / sd
    df["cw"] = [weights[r].get(m, 0.0) for r, m in zip(df["role"], df["metric"], strict=True)]
    df["anchor"] = [m in ANCHORS[r] for r, m in zip(df["role"], df["metric"], strict=True)]
    df = df[(df["cw"] > 0) & df["z"].notna() & df["zs"].notna()]
    df["wz"] = df["cw"] * df["z"]
    df["wzs"] = df["cw"] * df["zs"]
    g = df.groupby(["player_id", "period"], sort=True)
    out = pd.DataFrame(
        {
            "wz": g["wz"].sum(),
            "wzs": g["wzs"].sum(),
            "cw": g["cw"].sum(),
            "anchor": g["anchor"].any(),
            "full_name": g["full_name"].first(),
        }
    ).reset_index()
    out = out[out["anchor"]]
    out["value"] = out["wz"] / out["cw"]
    out["std_value"] = out["wzs"] / out["cw"]
    out["metric"] = "COMPOSITE"
    out["weight"] = float("nan")
    out["higher_is_better"] = True
    return out[
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
