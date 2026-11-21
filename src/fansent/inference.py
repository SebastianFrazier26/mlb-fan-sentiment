"""Per-metric hierarchical fit plus a permutation null for each fanbase effect.

Two nulls, picked by `InferenceConfig.null`:

- "freedman-lane" (default): within each unit, sentiment is regressed on the model's own
  controls (week-t perf and season-to-date std; common slopes, unit-specific intercepts), the
  residuals are permuted within unit and added back to the fitted values. This keeps the
  sentiment-from-performance dependence (fans react to results) and destroys only the part of
  sentiment that could predict next-week outcome beyond the controls.
- "shuffle": sentiment itself is shuffled within each unit. It is exchangeable only if sentiment
  is independent of perf/std; when fans react to performance the null is shifted off 0 and the
  p-values are biased (2026-11-18 diagnosis: null mean ~+0.03 for AVG/OPS/ERA/COMPOSITE).
  Kept for reproducing the registered `confirm` statistic.

Estimator (`InferenceConfig.estimator`): "pooled" (default, the registered one) regresses on raw
sentiment, so mu mixes the between-unit association with the within-unit effect. "mundlak" splits
sentiment, perf and std into unit mean + within-unit deviation, applies gamma_u / mu to the
deviation and adds the three unit means as fixed-coefficient covariates; mu is then the
within-player effect (sd of next outcome per sd of within-unit sentiment, a different estimand
and scale), and beta_between is the standardized coefficient on the sentiment unit mean. Both
nulls work unchanged: the model inputs are already deviations, so Freedman-Lane regresses the
sentiment deviation on the demeaned controls and permutes within unit, and the unit means (the
fixed `extras`) never move.

The statistic is the posterior mean of mu_k from a light fit; the observed value uses the
identical light settings (and its own seed slot) so the comparison is like for like.
perm_p is two-sided and centered on the null median c:
(1 + #{|null - c| >= |obs - c|}) / (1 + B); an uncentered |null| >= |obs| assumes a null
symmetric about 0, which neither null guarantees. perm_p_less / perm_p_greater are the
one-sided (1 + #{null <= obs}) / (1 + B) and its mirror; they compare obs with the null's own
distribution, so a shifted null is already accounted for and no centering applies.

Seeds: one SeedSequence child per replicate (slot 0 = observed), so results do not depend on
worker count.
"""

from __future__ import annotations

import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import hierarchical as h
from .model_data import ESTIMATORS, ModelInputs, panel_to_inputs
from .panel import oriented


@dataclass(frozen=True)
class InferenceConfig:
    n_chains: int = 4
    n_iter: int = 6000
    burn: int = 3000
    perm_b: int = 1000
    perm_chains: int = 2
    perm_iter: int = 600
    perm_burn: int = 300
    seed: int = 0
    workers: int = 1
    # Fit all units as one group (single mu) instead of one mu per fanbase.
    pool_fanbases: bool = False
    # Which permutation p feeds the multiple-comparison correction and the table marker.
    alternative: str = "two-sided"
    # "pooled" (registered) or "mundlak" (within-unit; see module docstring).
    estimator: str = "pooled"
    # "freedman-lane" or "shuffle" (see module docstring).
    null: str = "freedman-lane"
    # Also compute this null and report its one-sided and centered p as supplementary columns.
    supplementary_null: str | None = None
    priors: h.Priors = field(default_factory=h.Priors)

    def __post_init__(self) -> None:
        # analyze() files per-metric ValueErrors under "skipped", which would hide a typo here.
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"unknown estimator {self.estimator!r}; expected one of {ESTIMATORS}")


@dataclass
class MetricResult:
    metric: str
    fanbases: pd.DataFrame
    units: pd.DataFrame


def within_unit_permutation(unit_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Index array p such that sent[p] shuffles sentiment within each unit.

    Sorting by (unit, random key) and by unit alone yields the same unit blocks, so pairing the
    two orderings positionwise permutes inside blocks only."""
    base = np.argsort(unit_idx, kind="stable")
    shuffled = np.lexsort((rng.random(unit_idx.size), unit_idx))
    p = np.empty_like(base)
    p[base] = shuffled
    return p


def demean_by_unit(x: np.ndarray, unit_idx: np.ndarray) -> np.ndarray:
    n = np.bincount(unit_idx)
    return x - (np.bincount(unit_idx, weights=x) / np.maximum(n, 1))[unit_idx]


def freedman_lane_parts(inp: ModelInputs) -> tuple[np.ndarray, np.ndarray]:
    """(fitted, resid) with fitted + resid == sent. Sentiment is regressed on perf and std with
    common slopes and unit intercepts (via within-unit demeaning); fitted keeps the unit mean."""
    u = inp.unit_idx
    xd = np.column_stack([demean_by_unit(inp.perf, u), demean_by_unit(inp.std, u)])
    sd = demean_by_unit(inp.sent, u)
    coef, *_ = np.linalg.lstsq(xd, sd, rcond=None)
    resid = sd - xd @ coef
    return inp.sent - resid, resid


def _light_mu(inp: ModelInputs, sent: np.ndarray, cfg: InferenceConfig, seed) -> np.ndarray:
    res = h.fit(
        inp.y_next,
        inp.perf,
        inp.std,
        sent,
        inp.unit_idx,
        inp.fanbase_idx_of_unit,
        inp.weights,
        n_chains=cfg.perm_chains,
        n_iter=cfg.perm_iter,
        burn=cfg.perm_burn,
        seed=seed,
        priors=cfg.priors,
        store_units=False,
        extra=inp.extras,
        extra_names=list(inp.extra_names),
    )
    return res.pooled("mu").mean(axis=0)


NULLS = ("freedman-lane", "shuffle")
_STATE: dict = {}


def _init_worker(inp: ModelInputs, cfg: InferenceConfig, kind: str) -> None:
    if kind not in NULLS:
        raise ValueError(f"unknown null {kind!r}; expected one of {NULLS}")
    _STATE["inp"], _STATE["cfg"] = inp, cfg
    _STATE["parts"] = freedman_lane_parts(inp) if kind == "freedman-lane" else None


def _null_rep(seed_seq: np.random.SeedSequence) -> np.ndarray:
    inp, cfg, parts = _STATE["inp"], _STATE["cfg"], _STATE["parts"]
    shuffle_seed, fit_seed = seed_seq.spawn(2)
    perm = within_unit_permutation(inp.unit_idx, np.random.default_rng(shuffle_seed))
    sent = inp.sent[perm] if parts is None else parts[0] + parts[1][perm]
    return _light_mu(inp, sent, cfg, fit_seed)


def permutation_null(
    inp: ModelInputs, cfg: InferenceConfig, seed_base: int, kind: str | None = None
):
    """Returns (observed light-fit mu_k, null array of shape (B, K)); kind defaults to cfg.null."""
    kind = kind or cfg.null
    seeds = np.random.SeedSequence([cfg.seed, seed_base]).spawn(cfg.perm_b + 1)
    obs = _light_mu(inp, inp.sent, cfg, seeds[0])
    if cfg.perm_b == 0:
        return obs, np.empty((0, obs.size))
    if cfg.workers > 1:
        with ProcessPoolExecutor(
            cfg.workers, initializer=_init_worker, initargs=(inp, cfg, kind)
        ) as ex:
            nulls = list(
                ex.map(_null_rep, seeds[1:], chunksize=max(1, cfg.perm_b // (4 * cfg.workers)))
            )
    else:
        _init_worker(inp, cfg, kind)
        nulls = [_null_rep(s) for s in seeds[1:]]
    return obs, np.vstack(nulls)


def permutation_p(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    c = np.median(null, axis=0)
    dev = np.abs(null - c[None, :]) >= np.abs(obs - c)[None, :]
    return (1 + dev.sum(axis=0)) / (1 + null.shape[0])


def permutation_p_less(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    return (1 + (null <= obs[None, :]).sum(axis=0)) / (1 + null.shape[0])


def permutation_p_greater(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    return (1 + (null >= obs[None, :]).sum(axis=0)) / (1 + null.shape[0])


def analyze_metric(panel: pd.DataFrame, metric: str, cfg: InferenceConfig) -> MetricResult:
    """Fit one metric's panel (rows of a single metric); effects are in the better-performance
    direction on the z-scored scale (sd of oriented next outcome per sd of sentiment)."""
    sub = oriented(panel[panel["metric"] == metric])
    use_w = sub["weight"].notna().all()
    inp = panel_to_inputs(
        sub,
        y_next="y_next",
        perf="perf",
        std="std",
        sent="sent",
        weight="weight" if use_w else None,
        pool=cfg.pool_fanbases,
        estimator=cfg.estimator,
    )
    res = h.fit(
        inp.y_next,
        inp.perf,
        inp.std,
        inp.sent,
        inp.unit_idx,
        inp.fanbase_idx_of_unit,
        inp.weights,
        n_chains=cfg.n_chains,
        n_iter=cfg.n_iter,
        burn=cfg.burn,
        seed=[cfg.seed, zlib.crc32(metric.encode())],
        priors=cfg.priors,
        **inp.fit_kwargs(),
    )
    seed_base = zlib.crc32(metric.encode())
    obs, null = permutation_null(inp, cfg, seed_base)
    supp = None
    if cfg.supplementary_null and cfg.supplementary_null != cfg.null:
        supp = permutation_null(inp, cfg, seed_base, cfg.supplementary_null)

    fb = res.fanbase_summary()
    diag = res.diagnostics()
    rhat_max = []
    for name in inp.fanbase_names:
        shared = diag["param"].isin(
            [f"mu[{name}]", f"tau[{name}]", "sigma", *(f"beta_{b}" for b in res.beta_names)]
        )
        rhat_max.append(diag.loc[shared, "rhat"].max())
    nobs = np.bincount(inp.fanbase_idx_of_unit[inp.unit_idx], minlength=len(inp.fanbase_names))
    fanbases = pd.DataFrame(
        {
            "metric": metric,
            "fanbase": fb["fanbase"],
            "mu_mean": fb["mu_mean"],
            "ci_low": fb["mu_lo95"],
            "ci_high": fb["mu_hi95"],
            "p_pos": fb["p_positive"],
            "perm_p": permutation_p(obs, null),
            "perm_p_less": permutation_p_less(obs, null),
            "perm_p_greater": permutation_p_greater(obs, null),
            "n_units": fb["n_units"],
            "n_obs": nobs,
            "rhat_max": rhat_max,
            "null_mean": null.mean(axis=0),
        }
    )
    if "between" in res.beta_names:
        fanbases["beta_between"] = res.pooled("beta")[:, res.beta_names.index("between")].mean()
    if supp is not None:
        fanbases["supp_perm_p_less"] = permutation_p_less(*supp)
        fanbases["supp_perm_p"] = permutation_p(*supp)
        fanbases["supp_null_mean"] = supp[1].mean(axis=0)
    u = res.unit_summary()
    pids = [p for _, p in inp.unit_names]
    names = sub.drop_duplicates("player_id").set_index("player_id")["full_name"]
    units = pd.DataFrame(
        {
            "metric": metric,
            "fanbase": [f for f, _ in inp.unit_names],
            "player_id": pids,
            "full_name": [names.get(p, "") for p in pids],
            "n_obs": u["n_obs"],
            "gamma_mean": u["gamma_mean"],
            "gamma_lo95": u["gamma_lo95"],
            "gamma_hi95": u["gamma_hi95"],
            "p_pos": u["p_positive"],
        }
    )
    return MetricResult(metric, fanbases, units)
