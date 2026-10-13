"""Multinomial logistic regression with L2, fit by full-batch L-BFGS.

Objective (bias unpenalised):  C * sum_i s_i * (-log p_{i,y_i})  +  0.5 * ||W||^2
Gradient w.r.t. logits for row i is s_i * (p_i - onehot(y_i)), so
    dL/dW = C * X^T R + W,   dL/db = C * 1^T R,   R = s * (P - Y).
The loss is convex and smooth, so L-BFGS (scipy.optimize as the line-search/quasi-Newton
driver; model and gradients are ours) converges deterministically in tens of iterations, with no
learning rate or epoch count to tune. Adam would need both and gives noisier hyperparameter CV.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import minimize


def class_weights(y: np.ndarray, k: int, mode: str) -> np.ndarray:
    """Per-row weights s_i; 'balanced' = n / (k * n_c), which sums to n over the rows."""
    if mode == "none":
        return np.ones(len(y))
    if mode != "balanced":
        raise ValueError(f"class_weight must be 'balanced' or 'none', got {mode!r}")
    counts = np.bincount(y, minlength=k).astype(float)
    w = np.where(counts > 0, len(y) / (k * np.maximum(counts, 1)), 0.0)
    return w[y]


def _split(theta: np.ndarray, d: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    return theta[: d * k].reshape(d, k), theta[d * k :]


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def loss_grad(
    theta: np.ndarray, x: sparse.spmatrix, y: np.ndarray, s: np.ndarray, c: float, k: int
) -> tuple[float, np.ndarray]:
    d = x.shape[1]
    w, b = _split(theta, d, k)
    logp = log_softmax(x @ w + b)
    n = len(y)
    nll = -(s * logp[np.arange(n), y]).sum()
    loss = c * nll + 0.5 * float((w * w).sum())
    r = np.exp(logp)
    r[np.arange(n), y] -= 1.0
    r *= s[:, None]
    gw = c * np.asarray(x.T @ r) + w
    gb = c * r.sum(axis=0)
    return loss, np.concatenate([gw.ravel(), gb])


def fit_softmax(
    x: sparse.spmatrix,
    y: np.ndarray,
    k: int,
    c: float = 10.0,
    class_weight: str = "balanced",
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    d = x.shape[1]
    s = class_weights(y, k, class_weight)
    res = minimize(
        loss_grad,
        np.zeros(d * k + k),
        args=(x, y, s, c, k),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    return _split(res.x, d, k)


def predict_proba(x: sparse.spmatrix, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(x @ w + b))
