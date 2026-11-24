import json

import numpy as np
import pandas as pd
import pytest

from fansent import context as cx


def test_team_winpct_before_each_date():
    g = pd.DataFrame(
        {
            "team_id": [1, 1, 1, 1],
            "player_id": ["a", "b", "a", "a"],
            "date": pd.to_datetime(["2026-04-01", "2026-04-01", "2026-04-02", "2026-04-03"]),
            "team_win": [True, True, False, True],
        }
    )
    w = cx.team_winpct(g)
    assert w["winpct_pre"].tolist() == [0.5, 1.0, 0.5]


def test_debut_and_first_week(tmp_path):
    (tmp_path / "7.json").write_text(
        json.dumps({"people": [{"id": 7, "mlbDebutDate": "2026-04-02"}]})
    )
    (tmp_path / "8.json").write_text(json.dumps({"people": [{"id": 8}]}))
    d = cx.debut_years(tmp_path)
    assert d["7"] == 2026 and np.isnan(d["8"])
    pt = pd.DataFrame(
        {
            "player_id": ["7", "8", "8"],
            "team_id": [1, 1, 2],
            "week_start": ["2026-03-23", "2026-03-23", "2026-06-01"],
        }
    )
    assert cx.first_week_pairs(pt) == {("7", 1), ("8", 1)}


def test_fit_context_recovers_win_interaction():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(150):
        for d in range(50):
            v, won = rng.normal(), float(rng.random() < 0.5)
            winpct = rng.normal()
            row = {
                "player_id": str(p),
                "unit": f"F|{p}",
                "fan_day": pd.Timestamp("2026-04-01") + pd.Timedelta(days=d),
                "n_mentions": 3,
                "winpct_z": winpct,
                "new_to_team": 0.0,
                "rookie": float(p < 20),
            }
            for lag in range(2):
                row[f"vz_l{lag}"] = v if lag == 0 else 0.0
                row[f"played_l{lag}"] = 1.0 if lag == 0 else 0.0
                row[f"team_win_l{lag}"] = won if lag == 0 else 0.0
            row["sent"] = 0.1 * v + 0.05 * v * won + 0.2 * won + rng.normal(0, 0.2)
            rows.append(row)
    p = pd.DataFrame(rows)
    p.loc[p["player_id"] == "0", "new_to_team"] = 1.0
    r = cx.fit_context(p, 1)
    t = dict(zip(r.names, r.coef, strict=True))
    assert t["vz_l0_x_won"] == pytest.approx(0.05, abs=0.015)
    assert t["vz_l0_x_winpct_z"] == pytest.approx(0.0, abs=0.015)
