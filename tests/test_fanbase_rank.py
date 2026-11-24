import numpy as np
import pandas as pd
import pytest

from fansent import fanbase_rank as fr


def test_het_stat_zero_when_equal_and_weighted():
    n = pd.Series({"a": 1, "b": 3})
    assert fr.het_stat(pd.Series({"a": 1.0, "b": 1.0}), n) == 0
    # weighted mean 1.75; 1*(0.75^2) + 3*(0.25^2)
    assert fr.het_stat(pd.Series({"a": 1.0, "b": 2.0}), n) == pytest.approx(0.75)


def test_permute_units_keeps_unit_counts_within_group():
    panel = pd.DataFrame(
        {
            "unit": ["a|1", "a|1", "a|2", "b|3", "b|4", "a|5"],
            "fanbase": ["a", "a", "a", "b", "b", "a"],
            "group": ["h", "h", "h", "h", "h", "p"],
        }
    )
    out = fr.permute_units(panel, np.random.default_rng(0))
    assert out.drop_duplicates("unit").groupby(["group", "fanbase"]).size().to_dict() == {
        ("h", "a"): 2,
        ("h", "b"): 2,
        ("p", "a"): 1,
    }
    assert out.groupby("unit")["fanbase"].nunique().max() == 1


def test_permutation_test_detects_and_ignores():
    rng = np.random.default_rng(0)
    rows = []
    for f in range(8):
        for u in range(15):
            rows.append(
                {"unit": f"{f}|{u}", "fanbase": str(f), "group": "h", "x": rng.normal(0.5 * f, 1)}
            )
    panel = pd.DataFrame(rows)

    def est(p):
        return p.groupby("fanbase")["x"].mean()

    _, p_sig, _ = fr.permutation_test(panel, est, 199, 1)
    assert p_sig < 0.01
    panel["x"] = rng.normal(0, 1, len(panel))
    _, p_null, _ = fr.permutation_test(panel, est, 199, 1)
    assert p_null > 0.05


def test_normal_normal_gibbs_recovers_spread():
    rng = np.random.default_rng(2)
    theta = rng.normal(0, 1.0, 40)
    se = np.full(40, 0.3)
    post = fr.normal_normal_gibbs(theta + rng.normal(0, se), se, n_iter=3000, burn=1000)
    assert np.median(post["omega"]) == pytest.approx(1.0, abs=0.35)
    shrunk = post["theta"].reshape(-1, 40).mean(axis=0)
    assert np.corrcoef(shrunk, theta)[0, 1] > 0.9
    flat = fr.normal_normal_gibbs(np.zeros(20) + rng.normal(0, 1, 20), np.ones(20), seed=1)
    assert np.median(flat["omega"]) < 0.6


def test_rank_summary_and_score():
    draws = np.tile(np.arange(12.0), (100, 1))
    names = [f"f{i}" for i in range(12)]
    r = fr.rank_summary(draws, names, higher_better=False).set_index("fanbase")
    assert r.loc["f0", "rank_median"] == 1 and r.loc["f0", "tier"] == "top"
    assert r.loc["f11", "tier"] == "bottom"
    dl = np.tile(np.array([0.0, 1.0, 2.0]), (5, 1))
    oi = np.tile(np.array([2.0, 1.0, 0.0]), (5, 1))
    c = fr.score_draws({"dL": dl, "OI": oi})
    assert c[0].argmax() == 0  # dL is primary: low dL wins against high OI
