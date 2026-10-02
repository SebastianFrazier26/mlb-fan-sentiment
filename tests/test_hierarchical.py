import numpy as np
import pandas as pd
import pytest

from fansent import hierarchical as h
from fansent.diagnostics import ess_bulk, split_rhat
from fansent.model_data import panel_to_inputs


def simulate(
    mu_true=(0.3, 0.0, -0.3),
    units_per_fb=15,
    weeks=26,
    tau_g=0.1,
    tau_a=0.5,
    sigma=1.0,
    beta=(0.2, 0.1),
    seed=0,
    unit_weeks=None,
):
    rng = np.random.default_rng(seed)
    k_of_unit = np.repeat(np.arange(len(mu_true)), units_per_fb)
    J = k_of_unit.size
    gamma = rng.normal(np.asarray(mu_true)[k_of_unit], tau_g)
    alpha = rng.normal(0, tau_a, J)
    counts = np.full(J, weeks) if unit_weeks is None else np.asarray(unit_weeks)
    u = np.repeat(np.arange(J), counts)
    n = u.size
    perf, std, sent = rng.normal(size=(3, n))
    y = alpha[u] + beta[0] * perf + beta[1] * std + gamma[u] * sent + rng.normal(0, sigma, n)
    return {"y": y, "perf": perf, "std": std, "sent": sent, "u": u, "k": k_of_unit, "gamma": gamma}


def run(d, **kw):
    kw = {"n_iter": 1200, "burn": 600, **kw}
    return h.fit(d["y"], d["perf"], d["std"], d["sent"], d["u"], d["k"], seed=1, **kw)


@pytest.fixture(scope="module")
def recovery():
    d = simulate(units_per_fb=25, weeks=30)
    return d, run(d)


def test_recovers_fanbase_effects(recovery):
    _, res = recovery
    s = res.fanbase_summary()
    # Draws live on the z-scored scale, so compare against truth rescaled by sd(y)/sd(sent).
    scale = res.scales["sent"][1] / res.scales["y_next"][1]
    truth = np.array([0.3, 0.0, -0.3]) * scale
    assert ((s.mu_lo95 < truth) & (truth < s.mu_hi95)).all()
    assert s.p_positive[0] > 0.99 and s.p_positive[2] < 0.01
    assert s.mu_mean[0] > s.mu_mean[1] > s.mu_mean[2]


def test_converged_on_well_posed_case(recovery):
    _, res = recovery
    diag = res.diagnostics()
    assert (diag.rhat < 1.05).all()
    assert (diag.ess_bulk > 200).all()


def test_beta_and_sigma_recovered(recovery):
    _, res = recovery
    sy, sp, ss = res.scales["y_next"][1], res.scales["perf"][1], res.scales["std"][1]
    b = res.pooled("beta").mean(axis=0)
    assert b[0] == pytest.approx(0.2 * sp / sy, abs=0.04)
    assert b[1] == pytest.approx(0.1 * ss / sy, abs=0.04)


def test_null_effect_cri_covers_zero():
    d = simulate(mu_true=(0.0, 0.0, 0.0), tau_g=0.05, seed=3)
    s = run(d).fanbase_summary()
    assert ((s.mu_lo95 < 0) & (0 < s.mu_hi95)).all()


def test_shrinkage_pulls_sparse_units_toward_fanbase_mean():
    # Same strong true slope for every unit in one fanbase; the 3-obs units are noisy, so
    # their raw OLS slopes scatter widely while the posterior means sit near mu_k.
    J = 20
    weeks = np.array([40] * 10 + [3] * 10)
    d = simulate(mu_true=(0.0,), units_per_fb=J, tau_g=0.02, unit_weeks=weeks, seed=5)
    res = run(d)
    g = res.pooled("gamma").mean(axis=0)
    mu = res.pooled("mu").mean()
    raw = np.array(
        [np.polyfit(d["sent"][d["u"] == j], d["y"][d["u"] == j], 1)[0] for j in range(J)]
    )
    sparse = slice(10, 20)
    raw_dev = np.abs(raw[sparse] / res.scales["y_next"][1] * res.scales["sent"][1] - mu)
    post_dev = np.abs(g[sparse] - mu)
    assert post_dev.mean() < 0.5 * raw_dev.mean()


def test_deterministic_given_seed():
    d = simulate(units_per_fb=5, weeks=10)
    a = run(d, n_iter=200, burn=100)
    b = run(d, n_iter=200, burn=100)
    np.testing.assert_array_equal(a.draws["mu"], b.draws["mu"])
    c = h.fit(d["y"], d["perf"], d["std"], d["sent"], d["u"], d["k"], seed=2, n_iter=200, burn=100)
    assert not np.array_equal(a.draws["mu"], c.draws["mu"])


def test_weights_downweight_noisy_obs():
    d = simulate(mu_true=(0.0,), units_per_fb=10, weeks=40, seed=7)
    noisy = np.arange(d["y"].size) % 2 == 0
    d["y"] = d["y"] + np.where(noisy, np.random.default_rng(0).normal(0, 4, d["y"].size), 0)
    w = np.where(noisy, 1 / 16, 1.0)
    sig_w = run(d, weights=w, n_iter=600, burn=300).pooled("sigma").mean()
    sig_u = run(d, n_iter=600, burn=300).pooled("sigma").mean()
    assert sig_w < sig_u


def test_validation_errors():
    d = simulate(units_per_fb=2, weeks=5)
    base = (d["y"], d["perf"], d["std"], d["sent"], d["u"], d["k"])
    with pytest.raises(ValueError):
        h.fit(*base[:5], d["k"][:-1])
    with pytest.raises(ValueError):
        h.fit(base[0], base[1], base[2], np.zeros_like(base[3]), d["u"], d["k"])
    with pytest.raises(ValueError):
        h.fit(*base, weights=-np.ones_like(d["y"]))
    with pytest.raises(ValueError):
        h.fit(*base, n_iter=10, burn=10)


def test_store_units_false_and_summaries():
    d = simulate(units_per_fb=3, weeks=8)
    res = run(d, n_iter=100, burn=50, store_units=False)
    assert "gamma" not in res.draws
    with pytest.raises(ValueError):
        res.unit_summary()
    full = run(d, n_iter=100, burn=50, thin=5)
    assert full.n_draws == 10
    assert len(full.unit_summary()) == 9


# --- conditional updates against closed forms -------------------------------------------


def test_unit_effects_conditional_matches_bayes_regression():
    rng = np.random.default_rng(0)
    n = 12
    s = rng.normal(size=n)
    w = rng.uniform(0.5, 2, n)
    r = rng.normal(size=n)
    sig2, mu_a, ta2, mu_g, tg2 = 0.8, 0.3, 0.5, -0.2, 0.2
    Z = np.column_stack([np.ones(n), s])
    prec = np.diag([1 / ta2, 1 / tg2]) + Z.T @ (Z * w[:, None]) / sig2
    cov = np.linalg.inv(prec)
    mean = cov @ (np.array([mu_a / ta2, mu_g / tg2]) + Z.T @ (w * r) / sig2)

    C = 200_000
    ones = np.ones((C, 1))
    a, g = h._draw_unit_effects(
        np.random.default_rng(1),
        ones * (w * r).sum(), ones * (w * s * r).sum(),
        ones * w.sum(), ones * (w * s).sum(), ones * (w * s * s).sum(),
        ones / sig2, ones * mu_a, ones / ta2, ones * mu_g, ones / tg2,
    )  # fmt: skip
    draws = np.column_stack([a[:, 0], g[:, 0]])
    np.testing.assert_allclose(draws.mean(0), mean, atol=5e-3)
    np.testing.assert_allclose(np.cov(draws.T), cov, atol=5e-3)


def test_beta_conditional_matches_closed_form():
    rng = np.random.default_rng(2)
    n = 30
    X = rng.normal(size=(n, 2))
    w = rng.uniform(0.5, 2, n)
    resid = rng.normal(size=n)
    sig2, sd = 0.7, 2.0
    prec = np.eye(2) / sd**2 + X.T @ (X * w[:, None]) / sig2
    cov = np.linalg.inv(prec)
    mean = cov @ (X.T @ (w * resid)) / sig2
    C = 100_000
    draws = h._draw_beta(
        np.random.default_rng(3), np.tile(resid, (C, 1)), X, w, np.full((C, 1), sig2), sd
    )
    np.testing.assert_allclose(draws.mean(0), mean, atol=5e-3)
    np.testing.assert_allclose(np.cov(draws.T), cov, atol=5e-3)


def test_group_mean_and_scale_conditionals():
    rng = np.random.default_rng(4)
    C = 200_000
    tau2, n, total, sd = 0.25, 10, 3.0, 1.0
    m = h._draw_normal_mean(rng, np.full(C, total), np.full(C, tau2), n, sd)
    prec = 1 / sd**2 + n / tau2
    assert m.mean() == pytest.approx(total / tau2 / prec, abs=3e-3)
    assert m.var() == pytest.approx(1 / prec, rel=0.02)

    # tau^2 | aux fixed: IG((df+n)/2, df/aux + ss/2); mean = scale / (shape - 1)
    df, ss, nn, aux = 3.0, 2.0, 8, 1.5
    t2, _ = h._draw_scale(rng, np.full(C, ss), nn, np.full(C, aux), df, 1.0)
    shape, scale = (df + nn) / 2, df / aux + ss / 2
    assert t2.mean() == pytest.approx(scale / (shape - 1), rel=0.02)


# --- diagnostics -------------------------------------------------------------------------


def test_diagnostics_iid_vs_stuck_chains():
    rng = np.random.default_rng(0)
    iid = rng.normal(size=(4, 1000))
    assert split_rhat(iid) == pytest.approx(1.0, abs=0.01)
    assert ess_bulk(iid) == pytest.approx(4000, rel=0.25)
    stuck = iid + np.arange(4)[:, None] * 3.0
    assert split_rhat(stuck) > 1.5
    assert split_rhat(stuck, rank=False) > 1.5


def test_ess_detects_autocorrelation():
    rng = np.random.default_rng(1)
    phi = 0.9
    x = np.zeros((4, 2000))
    e = rng.normal(size=x.shape)
    for t in range(1, x.shape[1]):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    theory = 8000 * (1 - phi) / (1 + phi)  # ~421
    assert ess_bulk(x) == pytest.approx(theory, rel=0.4)


# --- adapter -----------------------------------------------------------------------------


def test_panel_to_inputs_roundtrip():
    rows = []
    for fb, pid, n in [("a", "p1", 4), ("a", "p2", 3), ("b", "p1", 5)]:
        for t in range(n):
            rows.append(
                {"fb": fb, "pid": pid, "y": t * 1.0, "p": t, "s": t % 2, "x": t * 0.5, "w": 1.0 + t}
            )
    df = pd.DataFrame(rows)
    df.loc[0, "y"] = np.nan
    m = panel_to_inputs(
        df, y_next="y", perf="p", std="s", sent="x", fanbase="fb", player="pid", weight="w"
    )
    assert m.y_next.size == 11
    assert m.fanbase_names == ["a", "b"]
    assert m.fanbase_idx_of_unit.tolist() == [0, 0, 1]
    assert np.bincount(m.unit_idx).tolist() == [3, 3, 5]
    assert m.unit_names[2] == ("b", "p1")


def test_noncentered_conditional_matches_closed_form():
    rng = np.random.default_rng(6)
    J, n = 4, 6
    s = rng.normal(size=(J, n))
    w = rng.uniform(0.5, 2, (J, n))
    r = rng.normal(size=(J, n))
    eta = rng.normal(size=J)
    sig2, tau2, sd, df, scale = 0.9, 0.3, 1.0, 3.0, 0.7
    C = 200_000
    onehot = np.ones((J, 1))
    s1 = (w * s * r).sum(1)
    swss = (w * s * s).sum(1)
    mu, theta = h._draw_noncentered_group(
        np.random.default_rng(7), np.tile(s1, (C, 1)), swss[None], np.tile(eta, (C, 1)),
        onehot, np.full((C, 1), 1 / sig2), np.full((C, 1), tau2), sd, df, scale,
    )  # fmt: skip

    # Reference: for fixed xi2 the draw is Bayesian regression of r on [s, eta*s]; marginalize
    # xi2 ~ IG((df+1)/2, df*scale^2/2 + tau2/2) by averaging conditional moments over its draws.
    xi2 = 1 / np.random.default_rng(8).gamma((df + 1) / 2, 1 / (df * scale**2 / 2 + tau2 / 2), C)
    Z = np.stack([s.ravel(), (eta[:, None] * s).ravel()], axis=1)
    ztz = Z.T @ (Z * w.ravel()[:, None]) / sig2
    ztr = Z.T @ (w.ravel() * r.ravel()) / sig2
    means = []
    for x in xi2[:20000]:
        cov = np.linalg.inv(ztz + np.diag([1 / sd**2, 1 / x]))
        means.append(cov @ ztr)
    ref = np.mean(means, axis=0)
    np.testing.assert_allclose([mu.mean(), theta.mean()], ref, atol=0.01)


def test_posterior_interval_calibration():
    # Truth drawn near the prior's bulk; a correct sampler gives ~95% coverage of mu_k.
    rng = np.random.default_rng(11)
    hits = total = 0
    for rep in range(40):
        mu_true = rng.normal(0, 0.2, 3)
        d = simulate(mu_true=tuple(mu_true), units_per_fb=8, weeks=15, tau_g=0.1, seed=100 + rep)
        res = h.fit(
            d["y"], d["perf"], d["std"], d["sent"], d["u"], d["k"],
            seed=rep, n_iter=700, burn=300, n_chains=2, store_units=False,
        )  # fmt: skip
        s = res.fanbase_summary()
        truth = mu_true * res.scales["sent"][1] / res.scales["y_next"][1]
        hits += int(((s.mu_lo95 < truth) & (truth < s.mu_hi95)).sum())
        total += 3
    assert hits / total > 0.88
