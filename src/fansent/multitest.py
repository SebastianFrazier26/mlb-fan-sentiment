"""Multiple-comparison corrections, implemented directly (no statsmodels)."""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p) -> np.ndarray:
    """BH step-up FDR-adjusted p-values (q-values), valid under independence / PRDS."""
    p = np.asarray(p, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def holm(p) -> np.ndarray:
    """Holm step-down FWER-adjusted p-values."""
    p = np.asarray(p, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    adj = np.maximum.accumulate(p[order] * (m - np.arange(m)))
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out
