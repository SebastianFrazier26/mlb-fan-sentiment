"""`fansent power`: Monte-Carlo power and minimum detectable effect (MDE) of the fanbase test.

Panels are simulated from the hierarchical model's own generative structure (hierarchical.py
docstring) at the sizes of the real analysis, with a known true fanbase effect mu, then refit with
the same sampler. A fit "detects" when the 95% credible interval of mu_k excludes 0 (two-sided).

That is a cheap proxy for the real decision rule. The shipped test is a within-unit permutation
p-value, BH-corrected within family, which is stricter than an uncorrected 95% CrI, so the power
reported here is an upper bound on the real test's power and the MDE a lower bound. `--perm-b`
re-runs the permutation test on a few replicates per point; uncorrected it matched the CrI, so
the gap is the multiplicity correction.

DGP per unit (all in z-score units, since the fit z-scores its inputs):
    perf_t = a_u + N(0,1);  std_t = running mean of perf;  sent_t = m_u + N(0,1)
    y_t    = alpha_u + b_perf*perf_t + b_std*std_t + gamma_u*sent_t + sigma*eps_t
    gamma_u ~ N(mu, tau^2)
gamma is rescaled so mu is in sd(y) per sd(sent) as in the real output (exact up to the usual
second-order term, via sd(y) ~ sd(y without sentiment)/sqrt(1 - mu^2 - tau^2)).
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import hierarchical as h
from .inference import InferenceConfig, permutation_null, permutation_p
from .model_data import ModelInputs

FANBASES_FILE = "analysis_fanbases.csv"
CURVES_OUT = "power_curves.csv"
MDE_OUT = "power_mde.csv"
DEFAULT_MUS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
DEFAULT_POOLED_MUS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1)
MIN_OBS_PER_UNIT = 8  # PanelConfig.min_weeks


@dataclass(frozen=True)
class Dgp:
    sigma: float = 1.0
    tau: float = 0.1
    beta_perf: float = 0.05
    beta_std: float = 0.1
    alpha_sd: float = 0.3
    unit_sent_sd: float = 0.5
    unit_perf_sd: float = 0.5


@dataclass(frozen=True)
class Scenario:
    name: str
    n_fanbases: int
    units_per_fanbase: int
    obs_per_unit: float
    mus: tuple[float, ...] = DEFAULT_MUS
    pooled: bool = False  # refit all units as one fanbase (shared mu)


@dataclass
class PowerConfig:
    reps: int = 100
    seed: int = 0
    workers: int = 1
    dgp: Dgp = field(default_factory=Dgp)
    fit_chains: int = 2
    fit_iter: int = 600
    fit_burn: int = 300
    perm_b: int = 0
    perm_reps: int = 10
    target: float = 0.8
    alpha: float = 0.05


def simulate(
    scn: Scenario, mu: float, dgp: Dgp, rng: np.random.Generator
) -> tuple[ModelInputs, np.ndarray]:
    """One synthetic panel with true fanbase effect `mu` in every fanbase. Returns the model
    inputs and the (fanbases,) array of true mu."""
    if mu * mu + dgp.tau * dgp.tau >= 1.0:
        raise ValueError("mu^2 + tau^2 must be < 1 (effects are in sd units)")
    n_units = scn.n_fanbases * scn.units_per_fanbase
    spread = 0.25 * scn.obs_per_unit
    t_u = np.maximum(
        MIN_OBS_PER_UNIT, np.rint(rng.normal(scn.obs_per_unit, spread, n_units))
    ).astype(int)
    unit_idx = np.repeat(np.arange(n_units), t_u)
    n = unit_idx.size
    starts = np.concatenate([[0], np.cumsum(t_u)[:-1]])

    perf = rng.normal(0, dgp.unit_perf_sd, n_units)[unit_idx] + rng.normal(size=n)
    csum = np.cumsum(perf)
    base = np.repeat(csum[starts] - perf[starts], t_u)
    pos = np.arange(n) - np.repeat(starts, t_u) + 1
    std = (csum - base) / pos
    sent = rng.normal(0, dgp.unit_sent_sd, n_units)[unit_idx] + rng.normal(size=n)
    alpha = rng.normal(0, dgp.alpha_sd, n_units)[unit_idx]
    y0 = alpha + dgp.beta_perf * perf + dgp.beta_std * std + dgp.sigma * rng.normal(size=n)

    gamma_z = rng.normal(mu, dgp.tau, n_units)
    scale = y0.std() / sent.std() / np.sqrt(1.0 - mu * mu - dgp.tau * dgp.tau)
    y = y0 + (gamma_z * scale)[unit_idx] * sent

    k_of_unit = (
        np.zeros(n_units, dtype=int)
        if scn.pooled
        else np.repeat(np.arange(scn.n_fanbases), scn.units_per_fanbase)
    )
    n_fb = 1 if scn.pooled else scn.n_fanbases
    inp = ModelInputs(
        y_next=y,
        perf=perf,
        std=std,
        sent=sent,
        unit_idx=unit_idx,
        fanbase_idx_of_unit=k_of_unit,
        weights=None,
        fanbase_names=[str(k) for k in range(n_fb)],
        unit_names=[(str(k), str(j)) for j, k in enumerate(k_of_unit)],
    )
    return inp, np.full(n_fb, mu)


def credible_detect(draws_mu: np.ndarray, alpha: float) -> np.ndarray:
    """(draws, fanbases) -> bool per fanbase: the central (1-alpha) CrI excludes 0."""
    lo, hi = np.quantile(draws_mu, [alpha / 2, 1 - alpha / 2], axis=0)
    return (lo > 0) | (hi < 0)


def mde(mus, power, target: float = 0.8) -> float:
    """Smallest effect where the power curve first reaches `target`, by linear interpolation
    between grid points. NaN if the curve never reaches it (MDE beyond the grid)."""
    mus = np.asarray(mus, dtype=float)
    power = np.asarray(power, dtype=float)
    order = np.argsort(mus)
    mus, power = mus[order], power[order]
    hit = np.flatnonzero(power >= target)
    if hit.size == 0:
        return float("nan")
    i = int(hit[0])
    if i == 0 or power[i] == target:
        return float(mus[i])
    p0, p1 = power[i - 1], power[i]
    return float(mus[i - 1] + (target - p0) / (p1 - p0) * (mus[i] - mus[i - 1]))


_STATE: dict = {}


def _init_worker(scenarios: list[Scenario], cfg: PowerConfig) -> None:
    _STATE["scenarios"], _STATE["cfg"] = scenarios, cfg


def _run_rep(task: tuple[int, int, int]) -> dict:
    si, mi, rep = task
    scn, cfg = _STATE["scenarios"][si], _STATE["cfg"]
    mu = scn.mus[mi]
    sim_seed, fit_seed, perm_seed = np.random.SeedSequence([cfg.seed, si, mi, rep]).spawn(3)
    inp, _ = simulate(scn, mu, cfg.dgp, np.random.default_rng(sim_seed))
    res = h.fit(
        inp.y_next,
        inp.perf,
        inp.std,
        inp.sent,
        inp.unit_idx,
        inp.fanbase_idx_of_unit,
        None,
        n_chains=cfg.fit_chains,
        n_iter=cfg.fit_iter,
        burn=cfg.fit_burn,
        seed=fit_seed,
        store_units=False,
    )
    out = {
        "si": si,
        "mi": mi,
        "detected": credible_detect(res.pooled("mu"), cfg.alpha),
        "perm_detected": None,
    }
    if cfg.perm_b > 0 and rep < cfg.perm_reps:
        icfg = InferenceConfig(perm_b=cfg.perm_b, workers=1, seed=cfg.seed)
        obs, null = permutation_null(inp, icfg, int(perm_seed.generate_state(1)[0]))
        out["perm_detected"] = permutation_p(obs, null) < cfg.alpha
    return out


def run_power(scenarios: list[Scenario], cfg: PowerConfig) -> pd.DataFrame:
    """Power curve table: scenario, mu, power, mcse, n_rep, n_tests, power_perm, n_perm_tests."""
    tasks = [
        (si, mi, r)
        for si, s in enumerate(scenarios)
        for mi in range(len(s.mus))
        for r in range(cfg.reps)
    ]
    # Heaviest fits first so the pool does not end on a single long straggler.
    tasks.sort(key=lambda t: -scenarios[t[0]].n_fanbases * scenarios[t[0]].units_per_fanbase)
    if cfg.workers > 1:
        with ProcessPoolExecutor(
            cfg.workers, initializer=_init_worker, initargs=(scenarios, cfg)
        ) as ex:
            results = list(ex.map(_run_rep, tasks, chunksize=1))
    else:
        _init_worker(scenarios, cfg)
        results = [_run_rep(t) for t in tasks]

    rows = []
    for si, s in enumerate(scenarios):
        for mi, mu in enumerate(s.mus):
            sel = [r for r in results if r["si"] == si and r["mi"] == mi]
            det = np.concatenate([r["detected"] for r in sel])
            pdet = [r["perm_detected"] for r in sel if r["perm_detected"] is not None]
            perm = np.concatenate(pdet) if pdet else np.empty(0, dtype=bool)
            p = det.mean()
            rows.append(
                {
                    "scenario": s.name,
                    "mu": mu,
                    "power": p,
                    # Fanbases within a dataset share sigma/beta, so this is slightly optimistic
                    # for the multi-fanbase scenarios; the rep count is the honest n.
                    "mcse": float(np.sqrt(p * (1 - p) / len(sel))),
                    "n_rep": len(sel),
                    "n_tests": det.size,
                    "power_perm": perm.mean() if perm.size else np.nan,
                    "n_perm_tests": perm.size,
                }
            )
    return pd.DataFrame(rows)


def mde_table(curves: pd.DataFrame, target: float) -> pd.DataFrame:
    rows = []
    for name, g in curves.groupby("scenario", sort=False):
        perm = g.dropna(subset=["power_perm"])
        rows.append(
            {
                "scenario": name,
                "mde": mde(g["mu"], g["power"], target),
                "fpr_at_0": g.loc[g["mu"] == 0, "power"].mean(),
                "mde_perm": mde(perm["mu"], perm["power_perm"], target)
                if len(perm) > 1
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def real_sizes(data_dir: Path, metric: str = "COMPOSITE") -> dict[str, tuple[float, float]]:
    """fanbase -> (units, obs per unit) from a previous correlate run; empty if absent."""
    path = data_dir / FANBASES_FILE
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[df["metric"] == metric]
    return {
        str(r.fanbase): (float(r.n_units), float(r.n_obs) / float(r.n_units))
        for r in df.itertuples()
        if r.n_units > 0
    }


def build_scenarios(
    sizes: dict[str, tuple[float, float]],
    league_units: int,
    league_fanbases: int,
    mus: tuple[float, ...],
    pooled_mus: tuple[float, ...],
    names: list[str],
) -> list[Scenario]:
    team = [sizes[k] for k in ("CHICubs", "redsox") if k in sizes]
    t_units, t_opu = (
        (round(float(np.mean([u for u, _ in team]))), float(np.mean([o for _, o in team])))
        if team
        else (23, 16.5)
    )
    b_units, b_opu = (
        (round(sizes["baseball"][0]), sizes["baseball"][1]) if "baseball" in sizes else (47, 16.2)
    )
    league = Scenario("league_per_fanbase", league_fanbases, league_units, t_opu, mus)
    registry = {
        "team": Scenario("team_sub", 1, t_units, t_opu, mus),
        "baseball": Scenario("r_baseball", 1, b_units, b_opu, mus),
        "league": league,
        "pooled": replace(league, name="league_pooled", mus=pooled_mus, pooled=True),
        "pitching": Scenario("team_pitching", 1, 10, 13.0, mus),
    }
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; choose from {sorted(registry)}")
    return [registry[n] for n in names]


def format_report(curves: pd.DataFrame, mdes: pd.DataFrame, target: float, seconds: float) -> str:
    blocks = []
    for name, g in curves.groupby("scenario", sort=False):
        t = g.drop(columns=["scenario"]).copy()
        t["power"] = t["power"].map("{:.2f}".format)
        t["mcse"] = t["mcse"].map("{:.2f}".format)
        t["power_perm"] = t["power_perm"].map("{:.2f}".format).replace("nan", "-")
        blocks.append(f"-- {name}\n{t.to_string(index=False)}")
    m = mdes.copy()
    for c in ("mde", "fpr_at_0", "mde_perm"):
        m[c] = m[c].map("{:.3f}".format).replace("nan", ">grid" if c != "fpr_at_0" else "-")
    blocks.append(
        f"-- MDE at {target:.0%} power (sd(next perf) per sd(sentiment))\n{m.to_string(index=False)}"
    )
    blocks.append(f"{seconds:.0f}s")
    return "\n\n".join(blocks)


def run(
    data_dir: Path,
    cfg: PowerConfig,
    names: list[str],
    mus: tuple[float, ...],
    pooled_mus: tuple[float, ...],
    league_units: int,
    league_fanbases: int,
    out_dir: Path | None = None,
) -> None:
    sizes = real_sizes(data_dir)
    print(
        f"sizes from {data_dir / FANBASES_FILE}: {sizes}" if sizes else "sizes: built-in defaults"
    )
    scenarios = build_scenarios(sizes, league_units, league_fanbases, mus, pooled_mus, names)
    for s in scenarios:
        print(
            f"{s.name}: {s.n_fanbases} fanbase(s) x {s.units_per_fanbase} units, "
            f"{s.obs_per_unit:.1f} obs/unit{' (pooled)' if s.pooled else ''}"
        )
    t0 = time.perf_counter()
    curves = run_power(scenarios, cfg)
    seconds = time.perf_counter() - t0
    mdes = mde_table(curves, cfg.target)
    print(format_report(curves, mdes, cfg.target, seconds))
    out = out_dir or data_dir
    curves.to_csv(out / CURVES_OUT, index=False)
    mdes.to_csv(out / MDE_OUT, index=False)
    print(f"\n{out / CURVES_OUT}, {out / MDE_OUT}")
