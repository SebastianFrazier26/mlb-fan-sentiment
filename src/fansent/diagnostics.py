"""MCMC convergence diagnostics (Vehtari et al. 2021, "Rank-normalization, folding, and
localization"). All inputs are scalar-parameter draws shaped (n_chains, n_draws)."""

from __future__ import annotations

import numpy as np
from scipy import special
from scipy import stats as sps


def _split(x: np.ndarray) -> np.ndarray:
    n = x.shape[1] // 2
    if n < 2:
        raise ValueError("need at least 4 draws per chain for split diagnostics")
    return np.concatenate([x[:, :n], x[:, -n:]], axis=0)


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Replace draws by normal scores of their pooled ranks (fractional offset 3/8)."""
    ranks = sps.rankdata(x, method="average").reshape(x.shape)
    return special.ndtri((ranks - 0.375) / (x.size + 0.25))


def split_rhat(x: np.ndarray, *, rank: bool = True) -> float:
    """Split-R-hat: each chain is halved so within-chain drift also inflates R-hat.
    rank=True uses rank-normalized draws (the bulk R-hat); rank=False is classic Gelman-Rubin."""
    x = np.asarray(x, dtype=float)
    z = _split(rank_normalize(x) if rank else x)
    n = z.shape[1]
    within = z.var(axis=1, ddof=1).mean()
    between = n * z.mean(axis=1).var(ddof=1)
    if within <= 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def ess_bulk(x: np.ndarray) -> float:
    """Bulk effective sample size of rank-normalized, split chains. Autocorrelations are
    summed with Geyer's initial monotone positive sequence."""
    z = _split(rank_normalize(np.asarray(x, dtype=float)))
    m, n = z.shape
    centered = z - z.mean(axis=1, keepdims=True)
    f = np.fft.rfft(centered, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
    chain_var = acov[:, 0] * n / (n - 1)
    within = chain_var.mean()
    var_plus = within * (n - 1) / n + n * z.mean(axis=1).var(ddof=1) / n
    if var_plus <= 0:
        return float("nan")
    rho = 1.0 - (within - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0

    npairs = (n - 1) // 2
    pairs = rho[: 2 * npairs].reshape(npairs, 2).sum(axis=1)
    neg = np.flatnonzero(pairs < 0)
    if neg.size:
        pairs = pairs[: neg[0]]
    pairs = np.minimum.accumulate(pairs)
    tau = -1.0 + 2.0 * pairs.sum()
    tau = max(tau, 1.0 / np.log10(m * n))
    return float(m * n / tau)
