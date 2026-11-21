import numpy as np
import pandas as pd
import pytest

from fansent.correlate import analyze
from fansent.inference import (
    InferenceConfig,
    demean_by_unit,
    freedman_lane_parts,
    permutation_null,
    permutation_p,
    permutation_p_less,
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
    assert (ps <= 0.2).mean() < 0.5
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
    pcfg = PanelConfig(min_weeks=8, min_units_per_fanbase=3, fanbase_teams={})
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
        stats,
        scores,
        [side],
        None,
        PanelConfig(min_weeks=6, fanbase_teams={}),
        icfg,
        include_composite=False,
    )
    assert {"news", "baseball"} == set(out.fanbases["fanbase"])


def reactive_inputs(seed, n_units=14, weeks=24, react=1.5, ar=0.9):
    """Persistent latent form drives next-week outcome; perf is its noisy reading and sentiment
    reacts to perf. Sentiment has no effect of its own on the next outcome."""
    rng = np.random.default_rng(seed)
    cols = []
    for _ in range(n_units):
        level = 0.7 * rng.normal()
        form = np.empty(weeks + 1)
        form[0] = level + 0.5 * rng.normal()
        for t in range(weeks):
            form[t + 1] = level + ar * (form[t] - level) + 0.5 * rng.normal()
        perf = form[:weeks] + 0.7 * rng.normal(size=weeks)
        std = np.cumsum(perf) / np.arange(1, weeks + 1)
        sent = react * perf + rng.normal(size=weeks)
        cols.append((form[1:] + 0.3 * rng.normal(size=weeks), perf, std, sent))
    y, perf, std, sent = (np.concatenate([c[i] for c in cols]) for i in range(4))
    z = lambda x: (x - x.mean()) / x.std()
    u = np.repeat(np.arange(n_units), weeks)
    return ModelInputs(z(y), z(perf), z(std), z(sent), u, np.zeros(n_units, int), None, ["p"], [])


def test_freedman_lane_parts_reconstruct_and_orthogonal():
    inp = reactive_inputs(0)
    rng = np.random.default_rng(1)
    order = rng.permutation(inp.sent.size)  # interleaved units
    inp = ModelInputs(
        *(a[order] for a in (inp.y_next, inp.perf, inp.std, inp.sent, inp.unit_idx)),
        inp.fanbase_idx_of_unit, None, inp.fanbase_names, [],
    )  # fmt: skip
    fitted, resid = freedman_lane_parts(inp)
    np.testing.assert_allclose(fitted + resid, inp.sent)
    u = inp.unit_idx
    for j in np.unique(u):
        m = u == j
        assert abs(resid[m].mean()) < 1e-10
    for x in (inp.perf, inp.std):
        assert abs(np.dot(resid, demean_by_unit(x, u))) < 1e-8
    # fitted carries the reaction to performance, residuals do not
    assert np.corrcoef(fitted, inp.perf)[0, 1] > 0.5


def test_freedman_lane_null_permutes_residuals_within_unit():
    inp = reactive_inputs(2)
    fitted, resid = freedman_lane_parts(inp)
    perm = within_unit_permutation(inp.unit_idx, np.random.default_rng(5))
    null_sent = fitted + resid[perm]
    assert (inp.unit_idx[perm] == inp.unit_idx).all()
    for j in np.unique(inp.unit_idx):
        m = inp.unit_idx == j
        np.testing.assert_allclose(np.sort(null_sent[m] - fitted[m]), np.sort(resid[m]))


def test_permutation_p_is_centered_on_null_median():
    b = 99
    sym = np.linspace(-1, 1, b)[:, None]
    obs = np.array([0.5])
    shifted = sym + 10.0
    assert permutation_p(obs, sym) == permutation_p(obs + 10.0, shifted)
    # an uncentered |null| >= |obs| would call this p = 1/(1+B): obs sits at the null's centre
    np.testing.assert_allclose(permutation_p(np.array([10.0]), shifted), 1.0)
    # one-sided p are relative to the null as it is, not re-centered
    np.testing.assert_allclose(permutation_p_less(np.array([10.0]), shifted), 51 / 100)


def test_freedman_lane_valid_when_sentiment_reacts_but_shuffle_is_not():
    cfg = InferenceConfig(perm_b=39, perm_chains=2, perm_iter=300, perm_burn=150, seed=11)
    s_mean, f_mean, s_p, f_p = [], [], [], []
    for rep in range(6):
        inp = reactive_inputs(rep)
        _, null_s = permutation_null(inp, cfg, rep, "shuffle")
        obs, null_f = permutation_null(inp, cfg, rep, "freedman-lane")
        s_mean.append(null_s.mean())
        f_mean.append(null_f.mean())
        s_p.append(permutation_p_less(obs, null_s)[0])
        f_p.append(permutation_p_less(obs, null_f)[0])
    # shuffle null is shifted well off 0 (about +0.04 here); Freedman-Lane stays near 0
    assert min(s_mean) > 0.02
    assert abs(np.mean(f_mean)) < 0.02
    # so the shuffle null rejects spuriously more often than Freedman-Lane
    assert sum(p <= 0.05 for p in s_p) > sum(p <= 0.05 for p in f_p)
