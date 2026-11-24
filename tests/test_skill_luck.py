import numpy as np
import pandas as pd
import pytest

from fansent import skill_luck as sl


def test_statcast_daily_uses_estimate_on_contact_actual_otherwise():
    pa = pd.DataFrame(
        {
            "game_date": ["2026-05-01"] * 3,
            "batter": ["1", "1", "1"],
            "pitcher": ["9", "9", "8"],
            "woba_value": [0.9, 0.0, 0.7],
            "woba_denom": [1, 1, 1],
            "estimated_woba_using_speedangle": [0.4, np.nan, np.nan],
        }
    )
    d = sl.statcast_daily(pa).set_index(["group", "player_id"])
    assert d.loc[("hitting", "1"), "xw_num"] == pytest.approx(0.4 + 0.0 + 0.7)
    assert d.loc[("hitting", "1"), "w_num"] == pytest.approx(1.6)
    assert d.loc[("pitching", "9"), "w_den"] == 2


def test_weekly_components_hitter_by_hand():
    games = pd.DataFrame(
        {
            "player_id": ["1", "1"],
            "date": pd.to_datetime(["2026-05-04", "2026-05-05"]),
            "group": ["hitting", "hitting"],
            "team_id": [112, 112],
            "team_win": [True, False],
            "pa": [5, 5],
            "ab": [4, 4],
            "h": [2, 1],
            "double": [1, 0],
            "triple": [0, 0],
            "hr": [0, 1],
            "bb": [1, 1],
            "ibb": [0, 0],
            "hbp": [0, 0],
            "sf": [0, 0],
            "k": [1, 2],
        }
    )
    sc = pd.DataFrame(
        {
            "player_id": ["1"],
            "group": ["hitting"],
            "week": pd.to_datetime(["2026-05-04"]),
            "xw_num": [3.0],
            "w_num": [3.5],
            "w_den": [10],
        }
    )
    w = sl.weekly_components(games, sc).iloc[0]
    assert w["k_pct"] == pytest.approx(0.3)
    assert w["bb_pct"] == pytest.approx(0.2)
    assert w["iso"] == pytest.approx((1 + 3) / 8)
    assert w["babip"] == pytest.approx((3 - 1) / (8 - 3 - 1))
    assert w["xwoba"] == pytest.approx(0.3)
    assert w["woba_minus_xwoba"] == pytest.approx(0.05)
    assert w["exposure"] == 10


def test_rest_of_season_by_hand():
    v = pd.DataFrame(
        {
            "player_id": ["a"] * 3,
            "group": ["hitting"] * 3,
            "date": pd.to_datetime(["2026-05-04", "2026-05-11", "2026-05-18"]),
            "v": [1.0, 2.0, 4.0],
            "n": [10.0, 10.0, 20.0],
        }
    )
    r = sl.rest_of_season(v).reset_index(drop=True)
    assert np.isnan(r.loc[0, "std_rate"])
    assert r.loc[1, "std_rate"] == pytest.approx(0.1)
    assert r.loc[1, "ros_rate"] == pytest.approx(0.2)
    assert r.loc[0, "ros_n"] == 30


def test_luck_share():
    c = pd.Series(
        {
            "k_pct": 1.0,
            "bb_pct": -1.0,
            "iso": 0.0,
            "xwoba": 2.0,
            "babip": -1.0,
            "woba_minus_xwoba": 1.0,
        }
    )
    assert sl.luck_share(c, "hitting") == pytest.approx(2 / 6)


def _synthetic(fan_on_luck: float, seed: int = 0):
    """Components are pure noise draws; the benchmark (rest-of-season rate) depends on skill
    components only; fans follow the benchmark weights, plus `fan_on_luck` on BABIP."""
    rng = np.random.default_rng(seed)
    cols = sl.SKILL["hitting"] + sl.LUCK["hitting"]
    w_true = pd.Series([0.3, 0.2, 0.2, 0.5, 0.0, 0.0], index=cols)
    rows = []
    for p in range(300):
        level = rng.normal()
        for wk in range(20):
            z = rng.normal(size=len(cols))
            ros = 0.01 * (z @ w_true.to_numpy()) + rng.normal(0, 0.01)
            sent = level + 0.1 * (z @ w_true.to_numpy()) + fan_on_luck * z[4] + rng.normal(0, 0.2)
            rows.append(
                {
                    "player_id": str(p),
                    "unit": f"F|{p}",
                    "fanbase": "F",
                    "sent": sent,
                    "n_mentions": 4,
                    "ros_rate": ros,
                    "ros_n": 200.0,
                    "std_rate": 0.0,
                    **dict(zip(cols, z, strict=True)),
                }
            )
    df = pd.DataFrame(rows)
    return df, df


def test_fit_s2_rational_vs_luck_chasing():
    fan, bench = _synthetic(0.0)
    assert abs(sl.fit_s2(fan, bench, "hitting").dL) < 0.08
    fan, bench = _synthetic(0.1)
    fit = sl.fit_s2(fan, bench, "hitting")
    assert fit.dL > 0.2
    assert fit.b["babip"] == pytest.approx(0.1, abs=0.02)


def test_bootstrap_dl_shapes():
    fan, bench = _synthetic(0.1, seed=1)
    d = sl.bootstrap_dl(fan, bench, "hitting", n_boot=3, seed=0, by_fanbase=True)
    assert list(d.columns) == ["draw", "dL", "F"] and np.isfinite(d["dL"]).all()
