"""S4: homers vs neutrals (reverse-direction study, analyses A12-A15).

Spec: docs/superpowers/specs/2026-11-23-reverse-direction-design.md (S4).

Pairs are player-weeks with >= `min_mentions` kept mentions both in the player's own team
subreddit and in r/baseball. Within a pair the week's news is the same, so:

- A12 level gap: mention-weighted mean of (team - r/baseball) sentiment over pairs, CR1 by
  player (classifier calibration per source is not yet applied: raw scale);
- A13 reaction gap: stacked pair rows, pair fixed effects, sentiment on team x value lags
  0..3 (the value lags themselves are constant within a pair and drop out); gap = sum of the
  interaction slopes;
- A14 / A15: overreaction and luck gaps, differences of the S1c and S2 estimates for team
  subs and r/baseball with the two SEs combined as if independent. Both sources mention the
  same players, so the true covariance is positive and this SE is conservative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .reaction import LEAGUE_FANBASE, within_wls

N_LAGS = 3


def pairs(wk: pd.DataFrame, min_mentions: int = 3) -> pd.DataFrame:
    w = wk[wk["n_mentions"] >= min_mentions]
    team = w[w["fanbase"] != LEAGUE_FANBASE]
    lg = w[w["fanbase"] == LEAGUE_FANBASE]
    keys = ["player_id", "week"]
    lag_cols = [c for c in w.columns if c.startswith(("vz_l", "n_l"))]
    m = team.merge(
        lg[[*keys, "sent", "n_mentions"]], on=keys, suffixes=("_team", "_lg"), how="inner"
    )
    return m[
        [
            *keys,
            "fanbase",
            "group",
            "sent_team",
            "n_mentions_team",
            "sent_lg",
            "n_mentions_lg",
            *lag_cols,
        ]
    ]


def level_gap(p: pd.DataFrame) -> tuple[float, float, int]:
    """Weighted mean gap and CR1 SE; weight = harmonic mean of the two mention counts."""
    d = (p["sent_team"] - p["sent_lg"]).to_numpy()
    w = 2 / (1 / p["n_mentions_team"] + 1 / p["n_mentions_lg"])
    r = within_wls(
        d,
        np.zeros((len(d), 0)),
        np.zeros(len(d), int),
        w.to_numpy(),
        p["player_id"].to_numpy(),
        [],
        absorb=False,
    )
    return float(r.coef[0]), float(r.se[0]), len(d)


def reaction_gap(p: pd.DataFrame) -> tuple[float, float, int]:
    lags = [f"vz_l{lag}" for lag in range(N_LAGS + 1)]
    pair_id = p["player_id"].astype(str) + "|" + p["week"].astype(str)
    stacked = pd.concat(
        [
            pd.DataFrame(
                {
                    "pair": pair_id,
                    "player_id": p["player_id"],
                    "team": 1.0,
                    "sent": p["sent_team"],
                    "w": p["n_mentions_team"],
                    **{c: p[c] for c in lags},
                }
            ),
            pd.DataFrame(
                {
                    "pair": pair_id,
                    "player_id": p["player_id"],
                    "team": 0.0,
                    "sent": p["sent_lg"],
                    "w": p["n_mentions_lg"],
                    **{c: p[c] for c in lags},
                }
            ),
        ],
        ignore_index=True,
    )
    X = np.column_stack(
        [stacked["team"].to_numpy()] + [(stacked["team"] * stacked[c]).to_numpy() for c in lags]
    )
    names = ["team", *[f"team_x_{c}" for c in lags]]
    r = within_wls(
        stacked["sent"].to_numpy(),
        X,
        stacked["pair"].to_numpy(),
        stacked["w"].to_numpy(),
        stacked["player_id"].to_numpy(),
        names,
    )
    a = np.array([0.0] + [1.0] * len(lags))
    return float(a @ r.coef), float(np.sqrt(a @ r.vcov @ a)), r.n_obs


def diff(a: float, se_a: float, b: float, se_b: float) -> tuple[float, float]:
    return a - b, float(np.hypot(se_a, se_b))


def p_two_sided(est: float, se: float) -> float:
    return float(2 * norm.sf(abs(est / se))) if se > 0 else float("nan")


def run(data_dir, season: int, prereg, s1_dir, s2_dir, out_dir, n_boot: int = 199, seed=0):
    """A12-A15. A14 reads the S1 run (s1_results.csv, primary OI rows); A15 reads the S2 run
    (team dL) and fits r/baseball dL here with the same bootstrap."""
    from pathlib import Path

    from . import reaction, skill_luck
    from .multitest import benjamini_hochberg, holm

    reaction.require_open_season(season, prereg)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, wk, _, _ = reaction.build(data_dir, season, 7, 5)
    rows = []
    for group in ("hitting", "pitching"):
        p = pairs(wk[wk["group"] == group])
        g, se, n = level_gap(p)
        rows.append({"analysis": "A12", "group": group, "estimate": g, "se": se, "n": n})
        g, se, n = reaction_gap(p)
        rows.append({"analysis": "A13", "group": group, "estimate": g, "se": se, "n": n})
    s1 = pd.read_csv(Path(s1_dir) / "s1_results.csv")
    s2 = pd.read_csv(Path(s2_dir) / "s2_results.csv").set_index("group")
    fan, bench, _, _ = skill_luck.build(data_dir, season)
    lg = fan[(fan["fanbase"] == LEAGUE_FANBASE) & (fan["n_mentions"] >= 3)]
    for group in ("hitting", "pitching"):
        t = s1[(s1["analysis"] == "A4") & s1["label"].str.contains(group)].iloc[0]
        b = s1[(s1["analysis"] == "A5") & s1["label"].str.contains(group)].iloc[0]
        est, se = diff(t["estimate"], t["se"], b["estimate"], b["se"])
        rows.append({"analysis": "A14", "group": group, "estimate": est, "se": se, "n": 0})
        f, bn = lg[lg["group"] == group], bench[bench["group"] == group]
        fit = skill_luck.fit_s2(f, bn, group)
        boot = skill_luck.bootstrap_dl(f, bn, group, n_boot, seed)
        est, se = diff(
            s2.loc[group, "dL"],
            s2.loc[group, "se_boot"],
            fit.dL,
            float(np.nanstd(boot["dL"], ddof=1)),
        )
        rows.append(
            {
                "analysis": "A15",
                "group": group,
                "estimate": est,
                "se": se,
                "n": fit.n_fan,
                "dL_baseball": fit.dL,
            }
        )
    res = pd.DataFrame(rows)
    res["p"] = [p_two_sided(e, s) for e, s in zip(res["estimate"], res["se"], strict=True)]
    res["family"] = "F4"
    res["q_bh"] = benjamini_hochberg(res["p"].to_numpy())
    res["p_holm"] = holm(res["p"].to_numpy())
    res.to_csv(out_dir / "s4_results.csv", index=False)
    return res
