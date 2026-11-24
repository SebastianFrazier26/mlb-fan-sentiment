"""S3: which fanbase "knows ball" (reverse-direction study, analyses A9-A11).

Spec: docs/superpowers/specs/2026-11-23-reverse-direction-design.md (S3).

Per fanbase k and player group, two indices from the own-team units only:

- dL_k (primary): S2 luck overweight, fan slopes fitted on fanbase k against the league-wide
  persistence benchmark;
- OI_k (secondary): S1c overreaction index with the team-won control.

The group estimates are combined per fanbase weighted by observations. Then:

- heterogeneity test: S = sum_k n_k (est_k - mean_n)^2 with n_k = units in fanbase k, null by
  permuting fanbase labels across units within player group (unit counts per fanbase are kept,
  so n_k is fixed under the null); p = (1 + #{S_null >= S}) / (1 + B);
- partial pooling: est_k ~ N(theta_k, se_k^2), theta_k ~ N(m, omega^2), half-Cauchy(A) on
  omega via the inverse-gamma mixture, Gibbs-sampled; se_k from a player-block bootstrap,
  taken as IQR / 1.349 because ratio indices have heavy bootstrap tails;
- knows-ball score C_k = -(2 z(dL_k) + z(OI_k)) / 3 per posterior draw, ranks (1 = best),
  tiers by P(rank in top / bottom 10) >= 0.8. Descriptive only on 2026.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

TIER_P = 0.8
TIER_SIZE = 10
SCORE_WEIGHTS = {"dL": 2.0, "OI": 1.0}


def het_stat(est: pd.Series, n: pd.Series) -> float:
    ok = est.notna() & np.isfinite(est)
    e, w = est[ok], n[ok].astype(float)
    mean = float((w * e).sum() / w.sum())
    return float((w * (e - mean) ** 2).sum())


def permute_units(panel: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Reassign fanbase labels across units, separately within each player group."""
    out = panel.copy()
    for idx in out.groupby("group").groups.values():
        sub = out.loc[idx]
        units = sub[["unit", "fanbase"]].drop_duplicates("unit")
        new = dict(zip(units["unit"], rng.permutation(units["fanbase"].to_numpy()), strict=True))
        out.loc[idx, "fanbase"] = sub["unit"].map(new).to_numpy()
    return out


def permutation_test(
    panel: pd.DataFrame,
    estimate: Callable[[pd.DataFrame], pd.Series],
    n_boot: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    """(observed S, permutation p, null draws). estimate(panel) -> per-fanbase index."""
    n_units = panel.drop_duplicates("unit").groupby("fanbase").size()
    obs = het_stat(estimate(panel), n_units)
    rng = np.random.default_rng(seed)
    null = np.empty(n_boot)
    for b in range(n_boot):
        null[b] = het_stat(estimate(permute_units(panel, rng)), n_units)
    p = float((1 + np.sum(null >= obs)) / (1 + n_boot))
    return obs, p, null


def normal_normal_gibbs(
    est: np.ndarray,
    se: np.ndarray,
    *,
    n_chains: int = 4,
    n_iter: int = 6000,
    burn: int = 2000,
    scale: float | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Draws (chains, kept, ...) of theta (K), m and omega for est_k ~ N(theta_k, se_k^2),
    theta_k ~ N(m, omega^2), flat m, omega ~ half-Cauchy(scale) written as
    omega^2 | a ~ IG(1/2, 1/a), a ~ IG(1/2, 1/scale^2) (Wand et al. 2011)."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    K = len(est)
    scale = scale or float(np.std(est, ddof=1) or 1.0)
    rng = np.random.default_rng(seed)
    kept = n_iter - burn
    out = {
        "theta": np.empty((n_chains, kept, K)),
        "m": np.empty((n_chains, kept)),
        "omega": np.empty((n_chains, kept)),
    }
    v = se**2
    for c in range(n_chains):
        m = float(np.mean(est)) + rng.normal(0, scale)
        om2 = float(np.var(est)) * rng.uniform(0.5, 2) + 1e-12
        a = 1.0
        for it in range(n_iter):
            prec = 1 / v + 1 / om2
            theta = rng.normal((est / v + m / om2) / prec, np.sqrt(1 / prec))
            m = rng.normal(theta.mean(), np.sqrt(om2 / K))
            ss = float(np.sum((theta - m) ** 2))
            om2 = 1 / rng.gamma((K + 1) / 2, 1 / (ss / 2 + 1 / a))
            a = 1 / rng.gamma(1.0, 1 / (1 / om2 + 1 / scale**2))
            if it >= burn:
                j = it - burn
                out["theta"][c, j], out["m"][c, j], out["omega"][c, j] = theta, m, np.sqrt(om2)
    return out


def robust_se(draws: pd.DataFrame) -> pd.Series:
    q1, q3 = draws.quantile(0.25), draws.quantile(0.75)
    return (q3 - q1) / 1.349


def score_draws(theta: dict[str, np.ndarray]) -> np.ndarray:
    """C draws (D, K) from pooled theta draws per index; higher = knows ball."""

    def z(x):
        return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)

    tot = sum(SCORE_WEIGHTS.values())
    return -sum(SCORE_WEIGHTS[k] * z(theta[k]) for k in SCORE_WEIGHTS) / tot


def rank_summary(values: np.ndarray, names: list[str], higher_better: bool) -> pd.DataFrame:
    """Posterior rank summary (1 = best) and tiers from draws (D, K)."""
    order = -values if higher_better else values
    ranks = order.argsort(axis=1).argsort(axis=1) + 1
    K = len(names)
    top = (ranks <= TIER_SIZE).mean(axis=0)
    bottom = (ranks > K - TIER_SIZE).mean(axis=0)
    tier = np.where(top >= TIER_P, "top", np.where(bottom >= TIER_P, "bottom", "middle"))
    lo, med, hi = np.quantile(ranks, [0.1, 0.5, 0.9], axis=0)
    return pd.DataFrame(
        {
            "fanbase": names,
            "rank_median": med,
            "rank_lo80": lo,
            "rank_hi80": hi,
            "p_top10": top,
            "p_bottom10": bottom,
            "tier": tier,
        }
    ).sort_values("rank_median")


MIN_PLAYERS = 10
RELIABILITY_FLOOR = 0.3


def split_half_reliability(mentions: pd.DataFrame, seed: int = 0, min_half: int = 3) -> pd.Series:
    """Per fanbase: Spearman-Brown reliability of weekly mean sentiment from a random half
    split of each player-week's mentions, after removing each unit's mean (the within-unit
    signal the S1-S3 models use)."""
    rng = np.random.default_rng(seed)
    m = mentions.assign(
        week=mentions["fan_day"] - pd.to_timedelta(mentions["fan_day"].dt.weekday, unit="D"),
        half=rng.integers(0, 2, len(mentions)),
    )
    g = m.groupby(["fanbase", "player_id", "week", "half"])["compound"].agg(["mean", "size"])
    g = g.unstack("half")
    g = g[(g[("size", 0)] >= min_half) & (g[("size", 1)] >= min_half)]
    d = pd.DataFrame({"a": g[("mean", 0)], "b": g[("mean", 1)]}).reset_index()
    for c in "ab":
        d[c] -= d.groupby(["fanbase", "player_id"])[c].transform("mean")
    r = d.groupby("fanbase").apply(
        lambda x: np.corrcoef(x["a"], x["b"])[0, 1], include_groups=False
    )
    return 2 * r / (1 + r)


def _combine(per_group: dict[str, pd.Series], counts: dict[str, pd.Series]) -> pd.Series:
    """Observation-weighted mean over player groups, per fanbase (NaN groups dropped)."""
    est = pd.DataFrame(per_group)
    w = pd.DataFrame(counts).reindex_like(est).where(est.notna(), 0.0).fillna(0.0)
    tot = w.sum(axis=1)
    return (est.fillna(0.0) * w).sum(axis=1) / tot.where(tot > 0)


def dl_estimator(lw: dict[str, float]) -> Callable[[pd.DataFrame], pd.Series]:
    from .skill_luck import fit_fan, luck_share

    def est(fan: pd.DataFrame) -> pd.Series:
        per, cnt = {}, {}
        for group, g in fan.groupby("group"):
            vals, ns = {}, {}
            for fb, sub in g.groupby("fanbase"):
                if sub["player_id"].nunique() < MIN_PLAYERS:
                    continue
                r = fit_fan(sub, str(group))
                vals[fb] = luck_share(pd.Series(r.coef, index=r.names), str(group)) - lw[group]
                ns[fb] = r.n_obs
            per[group], cnt[group] = pd.Series(vals, dtype=float), pd.Series(ns, dtype=float)
        return _combine(per, cnt)

    return est


def oi_estimator(n_lags: int, scales: dict[str, float]) -> Callable[[pd.DataFrame], pd.Series]:
    from .reaction import fit_overreaction, oi_from

    def est(panel: pd.DataFrame) -> pd.Series:
        per, cnt = {}, {}
        for group, g in panel.groupby("group"):
            vals, ns = {}, {}
            for fb, sub in g.groupby("fanbase"):
                if sub["player_id"].nunique() < MIN_PLAYERS:
                    continue
                r = fit_overreaction(sub, n_lags, control_win=True)
                vals[fb] = oi_from(r, n_lags, scales[group])[0]
                ns[fb] = r.n_obs
            per[group], cnt[group] = pd.Series(vals, dtype=float), pd.Series(ns, dtype=float)
        return _combine(per, cnt)

    return est


def bootstrap_by_fanbase(
    panel: pd.DataFrame, estimate: Callable[[pd.DataFrame], pd.Series], n_boot: int, seed: int
) -> pd.DataFrame:
    """Player-block resamples within each fanbase (one row per draw, column per fanbase)."""
    rng = np.random.default_rng(seed)
    blocks = {fb: g.groupby("player_id").indices for fb, g in panel.groupby("fanbase")}
    offsets = {fb: panel.index.get_indexer(g.index) for fb, g in panel.groupby("fanbase")}
    rows = []
    for _ in range(n_boot):
        parts, tags = [], []
        for fb, by in blocks.items():
            players = list(by)
            for rep, p in enumerate(rng.choice(len(players), size=len(players), replace=True)):
                idx = offsets[fb][by[players[p]]]
                parts.append(idx)
                tags.append(np.full(len(idx), rep))
        sub = panel.iloc[np.concatenate(parts)].copy()
        tag = "#" + np.concatenate(tags).astype(str)
        sub["player_id"] = sub["player_id"].astype(str).to_numpy() + tag
        sub["unit"] = sub["unit"].astype(str).to_numpy() + tag
        rows.append(estimate(sub))
    return pd.DataFrame(rows)


def run(
    data_dir,
    season: int,
    prereg=None,
    n_perm: int = 999,
    n_boot: int = 199,
    seed: int = 0,
    out_dir=None,
    n_lags: int = 7,
    min_mentions: int = 3,
) -> pd.DataFrame:
    import datetime as dt
    from pathlib import Path

    from . import reaction, skill_luck
    from .multitest import benjamini_hochberg, holm

    reaction.require_open_season(season, prereg)
    data_dir = Path(data_dir)
    out_dir = Path(out_dir or data_dir / "runs" / f"{dt.datetime.now(dt.UTC).date()}-fanbases")
    out_dir.mkdir(parents=True, exist_ok=True)

    day, _, hypers, n_med = reaction.build(data_dir, season, n_lags, 5)
    day = day[day["fanbase"] != reaction.LEAGUE_FANBASE].reset_index(drop=True)
    scales = {g: n_med[g] + h.k for g, h in hypers.items()}
    fan, bench, _, _ = skill_luck.build(data_dir, season)
    fan = fan[(fan["fanbase"] != reaction.LEAGUE_FANBASE) & (fan["n_mentions"] >= min_mentions)]
    fan = fan.reset_index(drop=True)
    lw = {}
    for group, b in bench.groupby("group"):
        r = skill_luck.fit_bench(b, str(group))
        lw[group] = skill_luck.luck_share(pd.Series(r.coef, index=r.names), str(group))

    specs = {
        "dL": (fan, dl_estimator(lw), "A10"),
        "OI": (day, oi_estimator(n_lags, scales), "A9"),
    }
    tests, est_tables, theta = [], [], {}
    for name, (panel, fn, aid) in specs.items():
        obs, p, _ = permutation_test(panel, fn, n_perm, seed)
        est = fn(panel)
        boot = bootstrap_by_fanbase(panel, fn, n_boot, seed + 1)
        se = robust_se(boot).reindex(est.index)
        ok = est.notna() & se.notna() & (se > 0)
        post = normal_normal_gibbs(est[ok].to_numpy(), se[ok].to_numpy(), seed=seed)
        th = post["theta"].reshape(-1, int(ok.sum()))
        theta[name] = pd.DataFrame(th, columns=est[ok].index)
        om = post["omega"].ravel()
        tests.append(
            {
                "analysis": aid,
                "index": name,
                "S_obs": obs,
                "p_perm": p,
                "n_fanbases": int(ok.sum()),
                "omega_median": float(np.median(om)),
                "omega_lo95": float(np.quantile(om, 0.025)),
                "omega_hi95": float(np.quantile(om, 0.975)),
                "m_mean": float(post["m"].mean()),
            }
        )
        lo, hi = np.quantile(th, [0.025, 0.975], axis=0)
        est_tables.append(
            pd.DataFrame(
                {
                    "index": name,
                    "fanbase": est[ok].index,
                    "estimate": est[ok].to_numpy(),
                    "se_boot": se[ok].to_numpy(),
                    "theta_mean": th.mean(axis=0),
                    "theta_lo95": lo,
                    "theta_hi95": hi,
                }
            )
        )
    from . import daypanel as dp
    from .panel import load_fanbase_teams as _lft
    from .panel import load_team_subreddits as _lts

    inp = reaction.load_inputs(data_dir)
    ment = dp.mention_days(inp["mentions"], inp["player_teams"], _lts(), _lft(), 5, aggregate=False)
    rel = split_half_reliability(ment[ment["fanbase"] != reaction.LEAGUE_FANBASE], seed)
    res = pd.DataFrame(tests)
    res["family"] = "F3"
    res["q_bh"] = benjamini_hochberg(res["p_perm"].to_numpy())
    res["p_holm"] = holm(res["p_perm"].to_numpy())
    common = sorted(set(theta["dL"].columns) & set(theta["OI"].columns))
    c = score_draws({k: theta[k][common].to_numpy() for k in SCORE_WEIGHTS})
    ranks = rank_summary(c, common, higher_better=True).assign(index="knows_ball")
    ranks["sb_rel"] = ranks["fanbase"].map(rel)
    ranks["too_noisy"] = ranks["sb_rel"] < RELIABILITY_FLOOR
    ranks_dl = rank_summary(theta["dL"].to_numpy(), list(theta["dL"].columns), False)
    res.to_csv(out_dir / "s3_tests.csv", index=False)
    pd.concat(est_tables).to_csv(out_dir / "s3_fanbase_estimates.csv", index=False)
    pd.concat([ranks, ranks_dl.assign(index="dL")]).to_csv(out_dir / "s3_ranks.csv", index=False)
    return res
