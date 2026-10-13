import numpy as np
import pandas as pd
import pytest

from fansent.correlate import analyze
from fansent.inference import (
    InferenceConfig,
    permutation_null,
    permutation_p,
    within_unit_permutation,
)
from fansent.model_data import ModelInputs
from fansent.panel import PanelConfig
from synth import make_scores, make_stats


def test_permutation_stays_within_units():
    rng = np.random.default_rng(0)
    unit = rng.integers(0, 5, 60)  # interleaved, unsorted units
    sent = rng.normal(size=60)
    p = within_unit_permutation(unit, np.random.default_rng(1))
    assert sorted(p) == list(range(60))
    assert (unit[p] == unit).all()
    for j in range(5):
        np.testing.assert_allclose(np.sort(sent[p][unit == j]), np.sort(sent[unit == j]))
    assert not np.array_equal(sent[p], sent)


def small_inputs(seed, n_units=6, weeks=12):
    rng = np.random.default_rng(seed)
    u = np.repeat(np.arange(n_units), weeks)
    n = u.size
    perf, std, sent = rng.normal(size=(3, n))
    y = 0.4 * perf + rng.normal(size=n)  # sentiment carries no information
    return ModelInputs(y, perf, std, sent, u, np.repeat([0, 1], n_units // 2), None, ["a", "b"], [])


FAST = InferenceConfig(perm_b=15, perm_chains=2, perm_iter=200, perm_burn=100, seed=3)


def test_null_pvalues_roughly_uniform_and_reproducible():
    ps = []
    for rep in range(10):
        obs, null = permutation_null(small_inputs(rep), FAST, rep)
        ps.extend(permutation_p(obs, null))
    ps = np.array(ps)
    assert ps.min() >= 1 / 16 and ps.max() <= 1.0
    assert 0.3 < ps.mean() < 0.7
    assert (ps <= 0.2).mean() < 0.4
    a = permutation_null(small_inputs(0), FAST, 0)
    b = permutation_null(small_inputs(0), FAST, 0)
    np.testing.assert_array_equal(a[1], b[1])


def test_permutation_p_formula():
    obs = np.array([0.5, 0.0])
    null = np.array([[0.6, 0.1], [-0.7, 0.2], [0.1, -0.3], [0.2, 0.0]])
    np.testing.assert_allclose(permutation_p(obs, null), [(1 + 2) / 5, 1.0])


@pytest.fixture(scope="module")
def e2e():
    stats = make_stats(n_players=8, weeks=24, seed=5)
    scores = make_scores(stats, {"CHICubs": 1.0, "redsox": 0.0}, seed=6)
    icfg = InferenceConfig(n_iter=800, burn=400, perm_b=39, perm_iter=300, perm_burn=150, seed=1)
    pcfg = PanelConfig(min_weeks=8, min_units_per_fanbase=3)
    return analyze(stats, scores, [], None, pcfg, icfg)


def test_planted_effect_detected_and_null_not(e2e):
    fb = e2e.fanbases.set_index(["metric", "fanbase"])
    planted = fb.loc[("OPS", "CHICubs")]
    null = fb.loc[("OPS", "redsox")]
    assert planted.mu_mean > 0.1 and planted.ci_low > 0 and planted.perm_p <= 0.05
    assert null.ci_low < 0 < null.ci_high and null.perm_p > 0.05
    assert fb.loc[("COMPOSITE", "CHICubs")].mu_mean > 0
    assert planted.n_units == 8 and planted.rhat_max < 1.1
    assert (e2e.fanbases[["q_bh", "p_holm"]] >= e2e.fanbases[["perm_p"]].to_numpy()).all().all()


def test_unit_table_and_columns(e2e):
    assert set(e2e.units["fanbase"]) == {"CHICubs", "redsox"}
    assert e2e.units["n_obs"].min() >= 8
    assert list(e2e.fanbases.columns)[:4] == ["family", "exploratory", "metric", "fanbase"]
    assert e2e.oaa_fanbases.empty


def test_sidecar_input_accepted():
    stats = make_stats(n_players=4, weeks=12, seed=2)
    scores = make_scores(stats, {"news": 0.5, "baseball": 0.0}, seed=3)
    side = pd.DataFrame(
        {
            "text_id": scores["text_id"].iloc[:20],
            "player_id": scores["player_id"].iloc[:20],
            "attribution": "reply_inherited",
        }
    )
    icfg = InferenceConfig(n_iter=300, burn=150, perm_b=3, perm_iter=150, perm_burn=75)
    out = analyze(
        stats, scores, [side], None, PanelConfig(min_weeks=6), icfg, include_composite=False
    )
    assert {"news", "baseball"} == set(out.fanbases["fanbase"])
