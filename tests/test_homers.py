import numpy as np
import pandas as pd
import pytest

from fansent import homers


def _wk(gap: float, slope_gap: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(200):
        for w in range(10):
            v = rng.normal()
            news = rng.normal()
            week = pd.Timestamp("2026-04-06") + pd.Timedelta(weeks=w)
            base = {
                "player_id": str(p),
                "week": week,
                "group": "hitting",
                "n_mentions": 5,
                "vz_l0": v,
                "vz_l1": 0.0,
                "vz_l2": 0.0,
                "vz_l3": 0.0,
                "n_l0": 20.0,
            }
            rows.append(
                {**base, "fanbase": "baseball", "sent": news + 0.1 * v + rng.normal(0, 0.1)}
            )
            rows.append(
                {
                    **base,
                    "fanbase": "Team",
                    "sent": news + gap + (0.1 + slope_gap) * v + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


def test_pairs_level_and_reaction_gap():
    p = homers.pairs(_wk(0.2, 0.05))
    assert len(p) == 2000
    g, se, n = homers.level_gap(p)
    assert g == pytest.approx(0.2, abs=0.02) and se > 0 and n == 2000
    rg, rse, _ = homers.reaction_gap(p)
    assert rg == pytest.approx(0.05, abs=0.02) and rse > 0


def test_pairs_min_mentions():
    wk = _wk(0.0, 0.0)
    wk.loc[wk["fanbase"] == "baseball", "n_mentions"] = 2
    assert homers.pairs(wk).empty
