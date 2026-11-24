import numpy as np
import pandas as pd
import pytest

from fansent import trades


def test_trade_events_first_midseason_move_only():
    pt = pd.DataFrame(
        {
            "player_id": ["1"] * 4 + ["2"] * 2 + ["3"] * 2,
            "team_id": [10, 10, 20, 30, 10, 10, 10, 20],
            "week_start": [
                "2026-03-23",
                "2026-03-30",
                "2026-04-06",
                "2026-04-13",
                "2026-03-23",
                "2026-03-30",
                "2026-03-23",
                "2026-03-23",
            ],
        }
    )
    ev = trades.trade_events(pt)
    assert ev.to_dict("records") == [
        {"player_id": "1", "old_team": 10, "new_team": 20, "week0": pd.Timestamp("2026-04-06")}
    ]


def _ev(honeymoon: float, early_extra: float, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(60):
        for e in range(-4, 10):
            v, news = rng.normal(), rng.normal(0, 0.1)
            week = pd.Timestamp("2026-06-01") + pd.Timedelta(weeks=e)
            base = {"player_id": str(p), "week": week, "event_week": e, "vz_l0": v, "n_mentions": 5}
            rows.append(
                {
                    **base,
                    "side": "league",
                    "fanbase": "baseball",
                    "sent": news + 0.1 * v + rng.normal(0, 0.05),
                }
            )
            if e < 0:
                rows.append(
                    {
                        **base,
                        "side": "old",
                        "fanbase": "Old",
                        "sent": news + 0.1 * v + rng.normal(0, 0.05),
                    }
                )
            else:
                slope = 0.1 + (early_extra if e < 4 else 0.0)
                bump = honeymoon if e < 4 else 0.0
                rows.append(
                    {
                        **base,
                        "side": "new",
                        "fanbase": "New",
                        "sent": news + bump + slope * v + rng.normal(0, 0.05),
                    }
                )
    return pd.DataFrame(rows)


def test_honeymoon_and_early_slope_recovered():
    ev = _ev(0.15, 0.05, seed=1)
    est, _, _, k = trades.honeymoon_gap(ev)
    assert est == pytest.approx(0.15, abs=0.03) and k == 60
    est, se, _, _ = trades.early_slope_gap(ev)
    assert est == pytest.approx(0.05, abs=0.02) and se > 0
