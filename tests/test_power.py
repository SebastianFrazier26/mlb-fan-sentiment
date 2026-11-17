import numpy as np
import pandas as pd
import pytest

from fansent import power


def test_mde_interpolates_between_grid_points():
    assert power.mde([0, 0.1, 0.2], [0.05, 0.5, 0.95], 0.8) == pytest.approx(0.1 + 0.3 / 0.45 * 0.1)


def test_mde_edge_cases():
    assert np.isnan(power.mde([0, 0.1], [0.05, 0.5], 0.8))
    assert power.mde([0, 0.1, 0.2], [0.05, 0.8, 1.0], 0.8) == pytest.approx(0.1)
    assert power.mde([0.2, 0, 0.1], [1.0, 0.05, 0.5], 0.8) == pytest.approx(0.1 + 0.3 / 0.5 * 0.1)


def test_credible_detect_is_two_sided():
    rng = np.random.default_rng(0)
    d = np.column_stack([rng.normal(1, 0.1, 500), rng.normal(-1, 0.1, 500), rng.normal(0, 1, 500)])
    assert power.credible_detect(d, 0.05).tolist() == [True, True, False]


def test_simulate_shapes_and_effect_scale():
    scn = power.Scenario("t", 2, 40, 16.0)
    inp, true_mu = power.simulate(scn, 0.2, power.Dgp(tau=0.0), np.random.default_rng(1))
    assert inp.fanbase_idx_of_unit.tolist() == [0] * 40 + [1] * 40
    assert np.bincount(inp.unit_idx).min() >= power.MIN_OBS_PER_UNIT
    assert true_mu.tolist() == [0.2, 0.2]
    # within-unit slope of y on sent approximates mu * sd(y)/sd(sent)
    u = inp.unit_idx
    sent_c = inp.sent - (np.bincount(u, inp.sent) / np.bincount(u))[u]
    y_c = inp.y_next - (np.bincount(u, inp.y_next) / np.bincount(u))[u]
    slope_z = (sent_c @ y_c) / (sent_c @ sent_c) * inp.sent.std() / inp.y_next.std()
    assert slope_z == pytest.approx(0.2, abs=0.06)


def test_pooled_scenario_has_single_fanbase():
    scn = power.Scenario("p", 3, 10, 12.0, pooled=True)
    inp, true_mu = power.simulate(scn, 0.0, power.Dgp(), np.random.default_rng(2))
    assert set(inp.fanbase_idx_of_unit) == {0} and true_mu.size == 1


def test_simulate_rejects_impossible_effect():
    with pytest.raises(ValueError):
        power.simulate(power.Scenario("t", 1, 5, 10.0), 1.0, power.Dgp(), np.random.default_rng(0))


def test_power_null_near_nominal_and_increasing():
    scn = power.Scenario("t", 10, 12, 14.0, mus=(0.0, 0.1, 0.4))
    cfg = power.PowerConfig(reps=6, seed=3, fit_iter=300, fit_burn=150)
    c = power.run_power([scn], cfg)
    p = c["power"].to_numpy()
    assert p[0] <= 0.2  # nominal 0.05 over 60 tests; loose for Monte-Carlo noise
    assert p[0] < p[1] < p[2] or (p[0] <= p[1] <= p[2] and p[2] > 0.8)
    assert c["n_tests"].tolist() == [60, 60, 60]


def test_run_power_independent_of_worker_count():
    scn = power.Scenario("t", 2, 8, 12.0, mus=(0.0, 0.3))
    kw = {"reps": 3, "seed": 5, "fit_iter": 200, "fit_burn": 100}
    a = power.run_power([scn], power.PowerConfig(workers=1, **kw))
    b = power.run_power([scn], power.PowerConfig(workers=2, **kw))
    pd.testing.assert_frame_equal(a, b)


def test_build_scenarios_uses_real_sizes_and_validates_names():
    sizes = {"CHICubs": (20.0, 16.0), "redsox": (24.0, 14.0), "baseball": (50.0, 15.0)}
    team, base = power.build_scenarios(sizes, 25, 30, (0.1,), (0.01,), ["team", "baseball"])
    assert (team.units_per_fanbase, team.obs_per_unit) == (22, 15.0)
    assert base.units_per_fanbase == 50
    with pytest.raises(ValueError):
        power.build_scenarios({}, 25, 30, (0.1,), (0.01,), ["nope"])
