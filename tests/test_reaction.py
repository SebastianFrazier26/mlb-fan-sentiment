import numpy as np
import pandas as pd
import pytest

from fansent import reaction as rx


def test_within_wls_matches_dummy_regression():
    rng = np.random.default_rng(0)
    n, units = 400, 20
    unit = rng.integers(0, units, n)
    X = rng.normal(size=(n, 2))
    w = rng.uniform(0.5, 3, n)
    y = rng.normal(size=units)[unit] + X @ [0.7, -0.3] + rng.normal(0, 0.5, n)
    res = rx.within_wls(y, X, unit, w, unit, ["a", "b"])
    D = np.column_stack([X, np.eye(units)[unit]])
    sw = np.sqrt(w)
    full = np.linalg.lstsq(D * sw[:, None], y * sw, rcond=None)[0]
    np.testing.assert_allclose(res.coef, full[:2], atol=1e-10)
    assert np.all(res.se > 0)
    plain = rx.within_wls(y, X, unit, w, unit, ["a", "b"], absorb=False)
    assert plain.names[0] == "const" and len(plain.coef) == 3


def _synthetic_panel(kind: str, seed: int = 0, n_lags: int = 1) -> tuple[pd.DataFrame, float]:
    """Unit-days where sentiment responds to the lag-0 talent update (rational) or to the raw
    surprise (recency). k = 50; exposure varies over the season, which identifies the split."""
    rng = np.random.default_rng(seed)
    k = 50.0
    rows = []
    for p in range(150):
        n_pre = 0.0
        level = rng.normal()
        for d in range(60):
            s = rng.normal()
            n = 4.0
            u = s / (n_pre + n + k)
            scale_mid = 100 + k
            signal = u * scale_mid if kind == "rational" else s
            rows.append(
                {
                    "player_id": str(p),
                    "unit": f"F|{p}",
                    "fan_day": pd.Timestamp("2026-04-01") + pd.Timedelta(days=d),
                    "u_l0": u,
                    "s_l0": s,
                    "played_l0": 1.0,
                    "sent": level + 0.2 * signal + rng.normal(0, 0.3),
                    "n_mentions": 3,
                }
            )
            n_pre += n
    df = pd.DataFrame(rows)
    for col in ("u", "s", "played"):
        df[f"{col}_l1"] = df.groupby("player_id")[f"{col}_l0"].shift(1).fillna(0.0)
    return df, 100 + k


def test_overreaction_index_rational_vs_recency():
    rational, scale = _synthetic_panel("rational")
    oi, _, _ = rx.oi_from(rx.fit_overreaction(rational, 1), 1, scale)
    assert abs(oi) < 0.15
    recency, scale = _synthetic_panel("recency")
    oi, _, b_s = rx.oi_from(rx.fit_overreaction(recency, 1), 1, scale)
    assert oi == pytest.approx(1.0, abs=0.15)
    assert b_s == pytest.approx(0.2, abs=0.03)


def test_block_bootstrap_runs():
    panel, scale = _synthetic_panel("recency")
    draws = rx.block_bootstrap_oi(panel, 1, scale, n_boot=5, seed=1)
    assert draws.shape == (5,) and np.all(np.isfinite(draws))


def test_season_guard(tmp_path):
    rx.require_open_season(2026, None)
    with pytest.raises(SystemExit):
        rx.require_open_season(2025, None)
    draft = tmp_path / "pre.md"
    draft.write_text("Status: DRAFT\n")
    with pytest.raises(SystemExit):
        rx.require_open_season(2025, draft)
    draft.write_text("Status: FROZEN 2027-01-01T00:00Z\n")
    rx.require_open_season(2025, draft)
