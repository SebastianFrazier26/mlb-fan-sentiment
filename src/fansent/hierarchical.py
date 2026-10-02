"""Hierarchical Bayesian model of fan sentiment -> next-week performance, fit by Gibbs.

Per metric. Unit j is a (fanbase, player) pair, k(j) its fanbase, obs i belongs to unit u(i):

    y_i = alpha_u + beta1*perf_i + beta2*std_i + gamma_u*sent_i + eps_i,  eps_i ~ N(0, sigma^2/w_i)
    alpha_j ~ N(mu_alpha, tau_alpha^2)        gamma_j ~ N(mu_k(j), tau_k(j)^2)
    mu_k ~ N(0, s^2)   mu_alpha ~ N(0, s_a^2)   beta ~ N(0, s_b^2 I)   sigma^2 ~ IG(a, b)
    tau^2 ~ IG(nu/2, nu/a_tau),  a_tau ~ IG(1/2, 1/A^2)   => tau ~ half-t(nu, A)  (Makalic-Schmidt)

Full conditionals (everything else held fixed; w_i-weighted sums S over a unit's obs, z_i=(1, sent_i),
r_i = y_i - beta.x_i):
  (alpha_j, gamma_j) | . ~ N(P^-1 m, P^-1)  jointly, with
        P = diag(1/tau_alpha^2, 1/tau_k^2) + (1/sigma^2) * S[w z z'],
        m = (mu_alpha/tau_alpha^2, mu_k/tau_k^2) + (1/sigma^2) * S[w z r]
  beta | .     ~ N(Q^-1 c, Q^-1),  Q = I/s_b^2 + X'WX/sigma^2,  c = X'W(y - alpha_u - gamma_u*sent)/sigma^2
  sigma^2 | .  ~ IG(a + N/2, b + sum_i w_i e_i^2 / 2)
  mu_alpha | . ~ N(prec^-1 * sum(alpha)/tau_alpha^2, prec^-1),  prec = 1/s_a^2 + J/tau_alpha^2
  mu_k | .     ~ N(prec^-1 * sum_{j in k} gamma_j/tau_k^2, prec^-1),  prec = 1/s^2 + n_k/tau_k^2
  tau^2 | .    ~ IG((nu + n)/2, nu/a_tau + sum d^2/2),  d = deviations from the group mean, n = group size
  a_tau | .    ~ IG((nu + 1)/2, nu/tau^2 + 1/A^2)

Centered sampling of (mu_k, tau_k, gamma) funnels when tau_k is small, so each sweep also does an
ASIS step (Yu & Meng 2011): gamma_j = mu_k + theta_k*eta_j with theta_k ~ N(0, xi_k^2),
xi_k^2 ~ IG(nu/2, nu*A^2/2) (so |theta_k| = tau_k is half-t). Given alpha, beta, sigma^2, eta the
likelihood is linear in (mu_k, theta_k), giving a bivariate normal conditional per fanbase;
xi_k^2 | theta_k ~ IG((nu+1)/2, nu*A^2/2 + theta_k^2/2).

Chains are vectorized along a leading axis; per-unit sums use reduceat on obs sorted by unit.
Variables are z-scored (y, perf, std, sent separately) before fitting, so gamma is in
sd(y) per sd(sent) and weights are rescaled to mean 1 (sigma^2 is the variance of an average-weight obs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .diagnostics import ess_bulk, split_rhat

_FLOOR = 1e-12


@dataclass(frozen=True)
class Priors:
    """Defaults assume z-scored data (see module docstring)."""

    beta_sd: float = 5.0
    mu_alpha_sd: float = 10.0
    mu_sd: float = 1.0
    tau_df: float = 1.0
    tau_scale: float = 1.0
    sigma_a: float = 0.01
    sigma_b: float = 0.01


@dataclass
class FitResult:
    """Posterior draws, each shaped (n_chains, n_draws, ...). mu/tau are per fanbase; beta is
    (perf, std). alpha/gamma (per unit) are None when fit(store_units=False)."""

    draws: dict[str, np.ndarray]
    scales: dict[str, tuple[float, float]]  # name -> (mean, sd) used to z-score
    fanbase_names: list[str] | None = None
    unit_names: list[str] | None = None
    unit_fanbase: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    unit_nobs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    @property
    def n_chains(self) -> int:
        return self.draws["sigma"].shape[0]

    @property
    def n_draws(self) -> int:
        return self.draws["sigma"].shape[1]

    def pooled(self, name: str) -> np.ndarray:
        """Draws with chains merged: (n_chains * n_draws, ...)."""
        a = self.draws[name]
        return a.reshape(-1, *a.shape[2:])

    def _scalar_series(self) -> dict[str, np.ndarray]:
        d = self.draws
        out: dict[str, np.ndarray] = {}
        for k in range(d["mu"].shape[2]):
            label = self._fanbase_label(k)
            out[f"mu[{label}]"] = d["mu"][:, :, k]
            out[f"tau[{label}]"] = d["tau"][:, :, k]
        out["sigma"] = d["sigma"]
        out["beta_perf"] = d["beta"][:, :, 0]
        out["beta_std"] = d["beta"][:, :, 1]
        return out

    def _fanbase_label(self, k: int) -> str:
        return str(self.fanbase_names[k]) if self.fanbase_names is not None else str(k)

    def diagnostics(self) -> pd.DataFrame:
        """Split-R-hat (rank-normalized) and bulk ESS for mu_k, tau_k, sigma, beta."""
        rows = [
            {"param": p, "rhat": split_rhat(x), "ess_bulk": ess_bulk(x)}
            for p, x in self._scalar_series().items()
        ]
        return pd.DataFrame(rows)

    def fanbase_summary(self) -> pd.DataFrame:
        """Per fanbase: posterior mean / 95% CrI of mu_k, P(mu_k > 0), tau_k mean, R-hat, ESS."""
        mu = self.pooled("mu")
        tau = self.pooled("tau")
        lo, hi = np.quantile(mu, [0.025, 0.975], axis=0)
        rows = []
        for k in range(mu.shape[1]):
            x = self.draws["mu"][:, :, k]
            rows.append(
                {
                    "fanbase": self._fanbase_label(k),
                    "n_units": int(np.sum(self.unit_fanbase == k)),
                    "mu_mean": mu[:, k].mean(),
                    "mu_sd": mu[:, k].std(ddof=1),
                    "mu_lo95": lo[k],
                    "mu_hi95": hi[k],
                    "p_positive": (mu[:, k] > 0).mean(),
                    "tau_mean": tau[:, k].mean(),
                    "rhat": split_rhat(x),
                    "ess_bulk": ess_bulk(x),
                }
            )
        return pd.DataFrame(rows)

    def unit_summary(self) -> pd.DataFrame:
        """Per unit: shrunk gamma_j posterior mean / 95% CrI, P(gamma_j > 0), obs count."""
        if "gamma" not in self.draws:
            raise ValueError("unit draws were not stored (fit(..., store_units=False))")
        g = self.pooled("gamma")
        lo, hi = np.quantile(g, [0.025, 0.975], axis=0)
        names = self.unit_names if self.unit_names is not None else list(range(g.shape[1]))
        return pd.DataFrame(
            {
                "unit": names,
                "fanbase": [self._fanbase_label(int(k)) for k in self.unit_fanbase],
                "n_obs": self.unit_nobs,
                "gamma_mean": g.mean(axis=0),
                "gamma_lo95": lo,
                "gamma_hi95": hi,
                "p_positive": (g > 0).mean(axis=0),
            }
        )


def _inv_gamma(rng: np.random.Generator, shape, scale) -> np.ndarray:
    return 1.0 / rng.gamma(shape, 1.0 / np.asarray(scale))


def _draw_unit_effects(
    rng, sw_r, sws_r, sw, sws, swss, inv_sig2, mu_alpha, inv_tau_a2, mu_g, inv_tau_g2
):
    """Joint (alpha_j, gamma_j) draw. All inputs broadcast to (chains, units)."""
    return _draw_bivariate_normal(
        rng,
        inv_tau_a2 + sw * inv_sig2,
        sws * inv_sig2,
        inv_tau_g2 + swss * inv_sig2,
        mu_alpha * inv_tau_a2 + sw_r * inv_sig2,
        mu_g * inv_tau_g2 + sws_r * inv_sig2,
    )


def _draw_bivariate_normal(rng, p11, p12, p22, m1, m2):
    """Draw from N(P^-1 m, P^-1) elementwise for symmetric 2x2 precisions P."""
    det = p11 * p22 - p12 * p12
    c11, c12, c22 = p22 / det, -p12 / det, p11 / det
    mean1 = c11 * m1 + c12 * m2
    mean2 = c12 * m1 + c22 * m2
    l11 = np.sqrt(c11)
    l21 = c12 / l11
    l22 = np.sqrt(np.maximum(c22 - l21 * l21, _FLOOR))
    z1 = rng.standard_normal(mean1.shape)
    z2 = rng.standard_normal(mean1.shape)
    return mean1 + l11 * z1, mean2 + l21 * z1 + l22 * z2


def _draw_noncentered_group(rng, s1, swss, eta, onehot, inv_sig2, tau2, prior_sd, df, scale):
    """ASIS step. Reparameterize gamma_j = mu_k + theta_k*eta_j, eta_j ~ N(0,1), theta_k ~ t_df(scale)
    (|theta| = tau). Given alpha, beta and eta, (mu_k, theta_k) enter the likelihood linearly, so they
    are drawn jointly per fanbase. s1 = per-unit sum w*sent*(y - beta.x - alpha_u); swss = per-unit
    sum w*sent^2; both (chains, units) / (1, units). Returns (mu, theta) of shape (chains, fanbases)."""
    xi2 = _inv_gamma(rng, (df + 1.0) / 2.0, df * scale**2 / 2.0 + tau2 / 2.0)
    p11 = 1.0 / prior_sd**2 + (swss @ onehot) * inv_sig2
    p12 = ((eta * swss) @ onehot) * inv_sig2
    p22 = 1.0 / xi2 + ((eta * eta * swss) @ onehot) * inv_sig2
    return _draw_bivariate_normal(
        rng, p11, p12, p22, (s1 @ onehot) * inv_sig2, ((s1 * eta) @ onehot) * inv_sig2
    )


def _draw_beta(rng, resid, X, w, sigma2, prior_sd):
    """beta | . for resid = y - alpha_u - gamma_u*sent, shape (chains, N); sigma2 (chains, 1)."""
    xtwx = (X * w[:, None]).T @ X
    xtwr = resid @ (X * w[:, None])
    prec = xtwx[None] / sigma2[:, :, None] + np.eye(X.shape[1]) / prior_sd**2
    cov = np.linalg.inv(prec)
    mean = np.einsum("cij,cj->ci", cov, xtwr / sigma2)
    chol = np.linalg.cholesky(cov)
    return mean + np.einsum("cij,cj->ci", chol, rng.standard_normal(mean.shape))


def _draw_normal_mean(rng, total, tau2, n, prior_sd):
    """Mean of N(mu, tau2) group under N(0, prior_sd^2) prior; total = sum of group values."""
    prec = 1.0 / prior_sd**2 + n / tau2
    return total / tau2 / prec + rng.standard_normal(np.shape(prec)) / np.sqrt(prec)


def _draw_scale(rng, ss, n, aux, df, scale):
    """Half-t(df, scale) std dev via parameter expansion; returns (tau2, aux)."""
    tau2 = _inv_gamma(rng, (df + n) / 2.0, df / aux + ss / 2.0)
    aux = _inv_gamma(rng, (df + 1.0) / 2.0, df / tau2 + 1.0 / scale**2)
    return tau2, aux


def _zscore(x: np.ndarray, name: str) -> tuple[np.ndarray, tuple[float, float]]:
    m, s = float(x.mean()), float(x.std())
    if s < _FLOOR:
        raise ValueError(f"{name} has zero variance; cannot standardize")
    return (x - m) / s, (m, s)


def fit(
    y_next,
    perf,
    std,
    sent,
    unit_idx,
    fanbase_idx_of_unit,
    weights=None,
    *,
    n_chains: int = 4,
    n_iter: int = 2000,
    burn: int = 1000,
    thin: int = 1,
    seed: int | None = None,
    priors: Priors | None = None,
    store_units: bool = True,
    fanbase_names: list[str] | None = None,
    unit_names: list[str] | None = None,
) -> FitResult:
    """Fit the hierarchical model with `n_chains` vectorized Gibbs chains.

    Args:
        y_next, perf, std, sent: (N,) raw observations (z-scored internally).
        unit_idx: (N,) int in [0, J), the unit of each observation.
        fanbase_idx_of_unit: (J,) int in [0, K), fanbase of each unit.
        weights: optional (N,) positive precision weights (e.g. PA / IP).
        n_iter: total iterations per chain including `burn`; kept draws = (n_iter-burn)//thin.
        store_units: keep per-unit alpha/gamma draws; disable for cheap repeated fits.
    """
    p = priors or Priors()
    y, perf, std, sent = (np.asarray(a, dtype=float) for a in (y_next, perf, std, sent))
    unit_idx = np.asarray(unit_idx)
    k_of_unit = np.asarray(fanbase_idx_of_unit)
    n = y.shape[0]
    if not (perf.shape == std.shape == sent.shape == unit_idx.shape == (n,)) or y.ndim != 1:
        raise ValueError("y_next, perf, std, sent, unit_idx must be 1-D with equal length")
    if not all(np.isfinite(a).all() for a in (y, perf, std, sent)):
        raise ValueError("non-finite values in inputs")
    if unit_idx.dtype.kind not in "iu" or k_of_unit.dtype.kind not in "iu":
        raise ValueError("unit_idx and fanbase_idx_of_unit must be integer arrays")
    n_units = k_of_unit.shape[0]
    if unit_idx.min() < 0 or unit_idx.max() >= n_units:
        raise ValueError("unit_idx out of range for fanbase_idx_of_unit")
    n_fb = int(k_of_unit.max()) + 1
    if k_of_unit.min() < 0:
        raise ValueError("negative fanbase index")
    nobs = np.bincount(unit_idx, minlength=n_units)
    if (nobs == 0).any():
        raise ValueError("every unit needs at least one observation")
    if (np.bincount(k_of_unit, minlength=n_fb) == 0).any():
        raise ValueError("fanbase indices must be contiguous (every fanbase needs a unit)")
    if n_chains < 1 or burn < 0 or thin < 1 or n_iter <= burn or (n_iter - burn) // thin < 1:
        raise ValueError("need n_chains >= 1, 0 <= burn < n_iter, thin >= 1")
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,) or not np.isfinite(w).all() or (w <= 0).any():
            raise ValueError("weights must be positive, finite, length N")
        w = w / w.mean()

    y, sy = _zscore(y, "y_next")
    perf, sp = _zscore(perf, "perf")
    std, ss_ = _zscore(std, "std")
    sent, sn = _zscore(sent, "sent")

    order = np.argsort(unit_idx, kind="stable")
    y, perf, std, sent, w, u = (
        y[order],
        perf[order],
        std[order],
        sent[order],
        w[order],
        unit_idx[order],
    )
    X = np.column_stack([perf, std])
    starts = np.concatenate([[0], np.cumsum(nobs)[:-1]])
    ws = w * sent
    sw = np.add.reduceat(w, starts)[None]
    sws = np.add.reduceat(ws, starts)[None]
    swss = np.add.reduceat(ws * sent, starts)[None]
    onehot = np.zeros((n_units, n_fb))
    onehot[np.arange(n_units), k_of_unit] = 1.0
    n_k = onehot.sum(axis=0)[None]

    rng = np.random.default_rng(seed)
    C = n_chains
    alpha = rng.normal(0, 0.5, (C, n_units))
    gamma = rng.normal(0, 0.5, (C, n_units))
    beta = rng.normal(0, 0.5, (C, 2))
    sigma2 = np.exp(rng.normal(0, 0.3, (C, 1)))
    mu_alpha = rng.normal(0, 0.5, (C, 1))
    mu = rng.normal(0, 0.5, (C, n_fb))
    tau_a2 = np.exp(rng.normal(-1, 0.5, (C, 1)))
    tau2 = np.exp(rng.normal(-1, 0.5, (C, n_fb)))
    aux_a = np.ones((C, 1))
    aux = np.ones((C, n_fb))

    n_keep = (n_iter - burn) // thin
    out = {
        "mu": np.empty((C, n_keep, n_fb)),
        "tau": np.empty((C, n_keep, n_fb)),
        "sigma": np.empty((C, n_keep)),
        "beta": np.empty((C, n_keep, 2)),
        "mu_alpha": np.empty((C, n_keep)),
        "tau_alpha": np.empty((C, n_keep)),
    }
    if store_units:
        out["alpha"] = np.empty((C, n_keep, n_units))
        out["gamma"] = np.empty((C, n_keep, n_units))

    kept = 0
    for it in range(n_iter):
        r = y - beta @ X.T
        alpha, gamma = _draw_unit_effects(
            rng,
            np.add.reduceat(w * r, starts, axis=1),
            np.add.reduceat(ws * r, starts, axis=1),
            sw, sws, swss,
            1.0 / sigma2,
            mu_alpha,
            1.0 / tau_a2,
            mu[:, k_of_unit],
            1.0 / tau2[:, k_of_unit],
        )  # fmt: skip
        resid = y - alpha[:, u] - gamma[:, u] * sent
        beta = _draw_beta(rng, resid, X, w, sigma2, p.beta_sd)
        e = resid - beta @ X.T
        sigma2 = _inv_gamma(
            rng, p.sigma_a + n / 2.0, p.sigma_b + 0.5 * (w * e * e).sum(1, keepdims=True)
        )
        mu_alpha = _draw_normal_mean(
            rng, alpha.sum(1, keepdims=True), tau_a2, n_units, p.mu_alpha_sd
        )
        mu = _draw_normal_mean(rng, gamma @ onehot, tau2, n_k, p.mu_sd)
        tau_a2, aux_a = _draw_scale(
            rng,
            ((alpha - mu_alpha) ** 2).sum(1, keepdims=True),
            n_units,
            aux_a,
            p.tau_df,
            p.tau_scale,
        )
        tau2, aux = _draw_scale(
            rng, ((gamma - mu[:, k_of_unit]) ** 2) @ onehot, n_k, aux, p.tau_df, p.tau_scale
        )

        # Interweave with the non-centered parameterization: with small tau_k the centered
        # chain on (mu_k, tau_k, gamma) funnels and mixes badly; this step breaks it.
        eta = (gamma - mu[:, k_of_unit]) / np.sqrt(tau2[:, k_of_unit])
        s1 = np.add.reduceat(ws * (y - beta @ X.T), starts, axis=1) - alpha * sws
        mu, theta = _draw_noncentered_group(
            rng, s1, swss, eta, onehot, 1.0 / sigma2, tau2, p.mu_sd, p.tau_df, p.tau_scale
        )
        tau2 = np.maximum(theta * theta, _FLOOR)
        gamma = mu[:, k_of_unit] + theta[:, k_of_unit] * eta

        if it >= burn and (it - burn) % thin == 0 and kept < n_keep:
            out["mu"][:, kept] = mu
            out["tau"][:, kept] = np.sqrt(tau2)
            out["sigma"][:, kept] = np.sqrt(sigma2[:, 0])
            out["beta"][:, kept] = beta
            out["mu_alpha"][:, kept] = mu_alpha[:, 0]
            out["tau_alpha"][:, kept] = np.sqrt(tau_a2[:, 0])
            if store_units:
                out["alpha"][:, kept] = alpha
                out["gamma"][:, kept] = gamma
            kept += 1

    return FitResult(
        draws=out,
        scales={"y_next": sy, "perf": sp, "std": ss_, "sent": sn},
        fanbase_names=fanbase_names,
        unit_names=unit_names,
        unit_fanbase=k_of_unit.copy(),
        unit_nobs=nobs,
    )
