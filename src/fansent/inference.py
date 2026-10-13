"""Per-metric hierarchical fit plus a within-unit permutation null for each fanbase effect.

The permutation shuffles sentiment across each unit's own periods, holding y_next, perf and std
fixed, so unit-level sentiment and outcome distributions (marginals) are untouched and only the
sentiment-to-next-outcome pairing is destroyed. The statistic is the posterior mean of mu_k from a
light fit; the observed value uses the identical light settings (and its own seed slot) so the
comparison is like for like. p = (1 + #{|null| >= |obs|}) / (1 + B), two-sided.

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
from .model_data import ModelInputs, panel_to_inputs
from .panel import oriented


@dataclass(frozen=True)
class InferenceConfig:
    n_chains: int = 4
    n_iter: int = 2000
    burn: int = 1000
    perm_b: int = 1000
    perm_chains: int = 2
    perm_iter: int = 600
    perm_burn: int = 300
    seed: int = 0
    workers: int = 1
    priors: h.Priors = field(default_factory=h.Priors)


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
    )
    return res.pooled("mu").mean(axis=0)


_STATE: dict = {}


def _init_worker(inp: ModelInputs, cfg: InferenceConfig) -> None:
    _STATE["inp"], _STATE["cfg"] = inp, cfg


def _null_rep(seed_seq: np.random.SeedSequence) -> np.ndarray:
    inp, cfg = _STATE["inp"], _STATE["cfg"]
    shuffle_seed, fit_seed = seed_seq.spawn(2)
    perm = within_unit_permutation(inp.unit_idx, np.random.default_rng(shuffle_seed))
    return _light_mu(inp, inp.sent[perm], cfg, fit_seed)


def permutation_null(inp: ModelInputs, cfg: InferenceConfig, seed_base: int):
    """Returns (observed light-fit mu_k, null array of shape (B, K))."""
    seeds = np.random.SeedSequence([cfg.seed, seed_base]).spawn(cfg.perm_b + 1)
    obs = _light_mu(inp, inp.sent, cfg, seeds[0])
    if cfg.perm_b == 0:
        return obs, np.empty((0, obs.size))
    if cfg.workers > 1:
        with ProcessPoolExecutor(cfg.workers, initializer=_init_worker, initargs=(inp, cfg)) as ex:
            nulls = list(
                ex.map(_null_rep, seeds[1:], chunksize=max(1, cfg.perm_b // (4 * cfg.workers)))
            )
    else:
        _init_worker(inp, cfg)
        nulls = [_null_rep(s) for s in seeds[1:]]
    return obs, np.vstack(nulls)


def permutation_p(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    b = null.shape[0]
    return (1 + (np.abs(null) >= np.abs(obs)[None, :]).sum(axis=0)) / (1 + b)


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
    obs, null = permutation_null(inp, cfg, zlib.crc32(metric.encode()))
    perm_p = permutation_p(obs, null)

    fb = res.fanbase_summary()
    diag = res.diagnostics()
    rhat_max = []
    for name in inp.fanbase_names:
        shared = diag["param"].isin(
            [f"mu[{name}]", f"tau[{name}]", "sigma", "beta_perf", "beta_std"]
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
            "perm_p": perm_p,
            "n_units": fb["n_units"],
            "n_obs": nobs,
            "rhat_max": rhat_max,
        }
    )
    u = res.unit_summary()
    pids = [p for _, p in inp.unit_names]
    names = sub.drop_duplicates("player_id").set_index("player_id")["full_name"]
    units = pd.DataFrame(
        {
            "metric": metric,
            "fanbase": u["fanbase"],
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
