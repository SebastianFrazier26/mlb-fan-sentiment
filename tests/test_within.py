import numpy as np
import pandas as pd
import pytest

from fansent import hierarchical as h
from fansent.inference import InferenceConfig, analyze_metric, freedman_lane_parts
from fansent.model_data import MUNDLAK_EXTRAS, panel_to_inputs

CFG = {
    "n_chains": 2,
    "n_iter": 1500,
    "burn": 700,
    "perm_b": 40,
    "perm_chains": 2,
    "perm_iter": 300,
    "perm_burn": 150,
}


def make_panel(within, between, n_units=60, weeks=17, seed=0, noise=1.0, m_sd=0.6):
    """Sentiment = unit mean m_u + weekly deviation d. Next-week outcome level a_u = between*m_u
    + noise; outcome response to d is `within`."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        m = m_sd * rng.normal()
        a = between * m
        d = rng.normal(size=weeks)
        perf = rng.normal(size=weeks)
        y = a + within * d + noise * rng.normal(size=weeks)
        for t in range(weeks):
            rows.append(
                {
                    "metric": "OPS",
                    "fanbase": "pooled",
                    "player_id": str(u),
                    "full_name": f"p{u}",
                    "y_next": y[t],
                    "perf": perf[t],
                    "std": perf[: t + 1].mean(),
                    "sent": m + d[t],
                    "weight": 1.0,
                    "higher_is_better": True,
                }
            )
    return pd.DataFrame(rows)


def fit_mu(panel, estimator, **kw):
    cfg = InferenceConfig(estimator=estimator, **{**CFG, **kw})
    return analyze_metric(panel, "OPS", cfg).fanbases.iloc[0]


def z_scale(panel, estimator):
    """sd(sent) / sd(y) on the scale mu is reported in, to convert a raw coefficient."""
    inp = panel_to_inputs(
        panel, y_next="y_next", perf="perf", std="std", sent="sent", estimator=estimator
    )
    return inp.sent.std() / inp.y_next.std()


def test_pooled_mu_picks_up_between_unit_association_but_mundlak_does_not():
    panel = make_panel(within=0.0, between=0.5)
    pooled = fit_mu(panel, "pooled")
    mundlak = fit_mu(panel, "mundlak")
    assert pooled["ci_low"] > 0
    assert mundlak["ci_low"] < 0 < mundlak["ci_high"]
    assert abs(mundlak["mu_mean"]) < 0.5 * pooled["mu_mean"]
    assert mundlak["beta_between"] > 0.15
    assert "beta_between" not in pooled.index


def test_mundlak_recovers_planted_within_effect():
    panel = make_panel(within=0.3, between=0.5, seed=1)
    row = fit_mu(panel, "mundlak")
    truth = 0.3 * z_scale(panel, "mundlak")
    assert row["ci_low"] < truth < row["ci_high"]
    assert row["ci_low"] > 0


def test_mundlak_freedman_lane_null_centered_on_zero_with_between_structure():
    panel = make_panel(within=0.0, between=0.5, seed=2)
    row = fit_mu(panel, "mundlak", perm_b=60)
    assert abs(row["null_mean"]) < 0.03
    assert 0.02 < row["perm_p"] <= 1.0


def test_mundlak_inputs_are_deviations_with_means_as_extras():
    panel = make_panel(within=0.1, between=0.5, n_units=5, weeks=6)
    inp = panel_to_inputs(
        panel, y_next="y_next", perf="perf", std="std", sent="sent", estimator="mundlak"
    )
    assert inp.extra_names == MUNDLAK_EXTRAS
    for col in (inp.sent, inp.perf, inp.std):
        assert np.abs(np.bincount(inp.unit_idx, weights=col)).max() < 1e-10
    raw = panel.groupby("player_id").sent.mean().to_numpy()
    np.testing.assert_allclose(inp.extras[::6, 0], raw)
    fitted, resid = freedman_lane_parts(inp)
    np.testing.assert_allclose(fitted + resid, inp.sent)
    assert np.abs(np.bincount(inp.unit_idx, weights=fitted)).max() < 1e-10
    with pytest.raises(ValueError, match="estimator"):
        panel_to_inputs(panel, y_next="y_next", perf="perf", std="std", sent="sent", estimator="x")
    with pytest.raises(ValueError, match="estimator"):
        InferenceConfig(estimator="x")


def test_extra_covariates_recovered_by_gibbs_beta_block():
    rng = np.random.default_rng(5)
    J, T = 30, 25
    u = np.repeat(np.arange(J), T)
    n = u.size
    perf, std, sent = rng.normal(size=(3, n))
    extra = rng.normal(size=(J, 2))[u]  # unit-level, like the Mundlak means
    alpha = rng.normal(0, 0.3, J)
    y = alpha[u] + 0.2 * perf + extra @ np.array([0.7, -0.4]) + 0.3 * sent + rng.normal(size=n)
    res = h.fit(
        y, perf, std, sent, u, np.zeros(J, int), n_chains=2, n_iter=1500, burn=700, seed=1,
        extra=extra, extra_names=["a", "b"],
    )  # fmt: skip
    assert res.beta_names == ("perf", "std", "a", "b")
    assert res.pooled("beta").shape[1] == 4
    b = res.pooled("beta").mean(axis=0)
    sy = res.scales["y_next"][1]
    for i, (name, truth) in enumerate([("a", 0.7), ("b", -0.4)], start=2):
        assert b[i] == pytest.approx(truth * res.scales[name][1] / sy, abs=0.08)
    assert {"beta_a", "beta_b"} <= set(res.diagnostics()["param"])


def test_beta_conditional_generic_in_column_count():
    rng = np.random.default_rng(6)
    n, p = 40, 5
    X = rng.normal(size=(n, p))
    w = rng.uniform(0.5, 2, n)
    resid = rng.normal(size=n)
    sig2, sd = 0.7, 2.0
    cov = np.linalg.inv(np.eye(p) / sd**2 + X.T @ (X * w[:, None]) / sig2)
    mean = cov @ (X.T @ (w * resid)) / sig2
    C = 100_000
    draws = h._draw_beta(
        np.random.default_rng(3), np.tile(resid, (C, 1)), X, w, np.full((C, 1), sig2), sd
    )
    np.testing.assert_allclose(draws.mean(0), mean, atol=5e-3)
    np.testing.assert_allclose(np.cov(draws.T), cov, atol=5e-3)
