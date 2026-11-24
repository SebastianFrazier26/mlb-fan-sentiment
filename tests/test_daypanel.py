import json

import numpy as np
import pandas as pd
import pytest

from fansent import daypanel as dp


def _utc(s):
    return pd.Series([int(pd.Timestamp(x, tz="UTC").timestamp()) for x in s])


def test_fan_day_local_cutoff_and_dst():
    ts = _utc(
        [
            "2026-07-01 07:30",  # 00:30 PDT, after a night game -> previous fan-day
            "2026-07-01 07:30",  # 03:30 EDT -> previous fan-day
            "2026-07-01 10:00",  # 06:00 EDT -> same day
            "2026-03-08 09:30",  # 05:30 EDT (DST began 02:00) -> Mar 8; EST would say Mar 7
            "2026-03-08 08:30",  # 04:30 EDT -> Mar 7
            "2026-07-01 11:30",  # 04:30 MST (Phoenix, no DST) -> previous fan-day
        ]
    )
    tz = pd.Series(
        [
            "America/Los_Angeles",
            "America/New_York",
            "America/New_York",
            "America/New_York",
            "America/New_York",
            "America/Phoenix",
        ]
    )
    got = dp.fan_day(ts, tz).dt.strftime("%Y-%m-%d").tolist()
    assert got == [
        "2026-06-30",
        "2026-06-30",
        "2026-07-01",
        "2026-03-08",
        "2026-03-07",
        "2026-06-30",
    ]
    assert dp.fan_day(ts[:1], pd.Series([None])).isna().all()


def _split(pid, date, stat, game_type="R", team=112, win=True):
    return {
        "player": {"id": pid},
        "date": date,
        "gameType": game_type,
        "team": {"id": team},
        "isWin": win,
        "stat": stat,
    }


def test_read_game_logs_sums_doubleheaders_and_drops_non_regular(tmp_path):
    hit = {"plateAppearances": 4, "atBats": 4, "hits": 1, "homeRuns": 1}
    splits = [
        _split(1, "2026-05-01", hit),
        _split(1, "2026-05-01", hit, win=False),
        _split(1, "2026-10-05", hit, game_type="P"),
    ]
    (tmp_path / "1_hitting.json").write_text(json.dumps({"stats": [{"splits": splits}]}))
    g = dp.read_game_logs(tmp_path)
    assert len(g) == 1
    row = g.iloc[0]
    assert (row["pa"], row["hr"], row["games"], bool(row["team_win"])) == (8, 2, 2, True)


def _games():
    return pd.DataFrame(
        {
            "player_id": ["a", "b", "p", "q"],
            "date": pd.to_datetime(["2026-05-01"] * 4),
            "group": ["hitting", "hitting", "pitching", "pitching"],
            "pa": [4, 4, 0, 0],
            "ab": [4, 4, 0, 0],
            "h": [1, 0, 0, 0],
            "double": [0, 0, 0, 0],
            "triple": [0, 0, 0, 0],
            "hr": [1, 0, 0, 0],
            "bb": [0, 0, 0, 0],
            "ibb": [0, 0, 0, 0],
            "hbp": [0, 0, 0, 0],
            "sf": [0, 0, 0, 0],
            "k": [0, 0, 0, 0],
            "outs": [0, 0, 18, 9],
            "bf": [0, 0, 22, 14],
            "er": [0, 0, 1, 5],
        }
    )


def test_game_values_runs_above_league():
    v = dp.game_values(_games()).set_index("player_id")
    lg = dp.WOBA_WEIGHTS["hr"] / 8
    assert v.loc["a", "v"] == pytest.approx((dp.WOBA_WEIGHTS["hr"] - lg * 4) / dp.WOBA_SCALE)
    assert v.loc[["a", "b"], "v"].sum() == pytest.approx(0)
    assert v.loc[["p", "q"], "v"].sum() == pytest.approx(0)
    assert v.loc["p", "v"] == pytest.approx(6 / 27 * 18 - 1)
    assert v.loc["p", "n"] == 22


def test_eb_hyper_recovers_moments():
    rng = np.random.default_rng(1)
    tau, sigma = 0.03, 0.5
    rows = []
    for i in range(600):
        theta = rng.normal(0, tau)
        for _ in range(rng.integers(40, 120)):
            n = int(rng.integers(3, 6))
            rows.append((str(i), n, n * theta + rng.normal(0, sigma * np.sqrt(n))))
    g = pd.DataFrame(rows, columns=["player_id", "n", "v"])
    h = dp.eb_hyper(g)
    assert h.sigma2 == pytest.approx(sigma**2, rel=0.05)
    assert h.tau2 == pytest.approx(tau**2, rel=0.25)


def test_add_surprise_by_hand():
    g = pd.DataFrame(
        {
            "player_id": ["a", "a"],
            "date": pd.to_datetime(["2026-05-02", "2026-05-01"]),
            "v": [2.0, 1.0],
            "n": [4.0, 4.0],
        }
    )
    h = dp.EBHyper(m=0.0, tau2=1.0, sigma2=10.0, min_exposure=0)  # k = 10
    out = dp.add_surprise(g, h).reset_index(drop=True)
    assert out["N_pre"].tolist() == [0, 4]
    assert out.loc[0, "s"] == pytest.approx(1.0)
    assert out.loc[0, "u"] == pytest.approx(1.0 / 14)
    theta = 1.0 / 14
    assert out.loc[1, "theta_pre"] == pytest.approx(theta)
    assert out.loc[1, "s"] == pytest.approx(2.0 - 4 * theta)
    assert out.loc[1, "u"] == pytest.approx((2.0 - 4 * theta) / 18)


def test_mention_days_own_team_trade_and_league_zone():
    sub_fb = {"chicubs": "CHICubs", "dodgers": "Dodgers"}
    fb_teams = {"chicubs": frozenset({112}), "dodgers": frozenset({119})}
    pt = pd.DataFrame(
        {"player_id": ["7", "7"], "team_id": [112, 119], "week_start": ["2026-06-22", "2026-06-29"]}
    )
    m = pd.DataFrame(
        {
            "text_id": list("abcdef"),
            "subreddit": ["CHICubs", "CHICubs", "Dodgers", "Dodgers", "baseball", "baseball"],
            "player_id": ["7"] * 6,
            "created_utc": _utc(
                [
                    "2026-06-24 23:00",  # Cubs week, Cubs sub: kept
                    "2026-07-01 20:00",  # after the trade, Cubs sub: dropped
                    "2026-06-24 23:00",  # before the trade, Dodgers sub: dropped
                    "2026-07-02 08:00",  # 01:00 PDT Jul 2 -> fan-day Jul 1, Dodgers: kept
                    "2026-07-02 08:00",  # r/baseball, player on LAD: PDT -> Jul 1
                    "2026-06-25 08:00",  # r/baseball, player on CHC: 03:00 CDT -> Jun 24
                ]
            ),
            "compound": [0.5, 0.9, 0.9, -0.5, 0.2, 0.4],
        }
    )
    out = dp.mention_days(m, pt, sub_fb, fb_teams)
    got = {(r.fanbase, r.fan_day.strftime("%m-%d")): (r.sent, r.team_id) for r in out.itertuples()}
    assert got == {
        ("CHICubs", "06-24"): (0.5, 112),
        ("Dodgers", "07-01"): (-0.5, 119),
        ("baseball", "07-01"): (0.2, 119),
        ("baseball", "06-24"): (0.4, 112),
    }


def test_daily_lags_places_games_and_zero_fills():
    games = pd.DataFrame(
        {
            "player_id": ["a", "a"],
            "date": pd.to_datetime(["2026-05-01", "2026-05-03"]),
            "v": [1.0, 2.0],
        }
    )
    days = pd.DataFrame({"player_id": ["a"], "fan_day": pd.to_datetime(["2026-05-03"])})
    out = dp.daily_lags(games, days, ["v"], 3).iloc[0]
    assert [out[f"v_l{k}"] for k in range(4)] == [2.0, 0.0, 1.0, 0.0]
    assert [out[f"played_l{k}"] for k in range(4)] == [1.0, 0.0, 1.0, 0.0]


def test_weekly_aggregates_and_lags():
    days = pd.DataFrame(
        {
            "fanbase": ["F", "F"],
            "player_id": ["a", "a"],
            "fan_day": pd.to_datetime(["2026-05-11", "2026-05-12"]),
            "sent": [1.0, 0.0],
            "n_mentions": [1, 3],
        }
    )
    games = pd.DataFrame(
        {
            "player_id": ["a", "a"],
            "date": pd.to_datetime(["2026-05-06", "2026-05-12"]),
            "v": [1.0, 2.0],
            "n": [4.0, 4.0],
        }
    )
    w = dp.weekly(days, games, ["v"], 1).iloc[0]
    assert w["sent"] == pytest.approx(0.25)
    assert (w["v_l0"], w["v_l1"], w["n_l1"]) == (2.0, 1.0, 4.0)


def test_first_pitch_and_pregame_drop():
    sched = pd.DataFrame(
        {
            "official_date": ["2026-06-24", "2026-06-24"],
            "game_utc": ["2026-06-24T23:05:00Z", "2026-06-24T17:05:00Z"],
            "home_id": [112, 112],
            "away_id": [138, 138],
        }
    )
    fp = dp.first_pitch(sched)
    assert len(fp) == 2
    assert fp["first_pitch_utc"].iloc[0] == int(
        pd.Timestamp("2026-06-24 17:05", tz="UTC").timestamp()
    )
    pt = pd.DataFrame({"player_id": ["7"], "team_id": [112], "week_start": ["2026-06-22"]})
    m = pd.DataFrame(
        {
            "text_id": ["a", "b"],
            "subreddit": ["CHICubs", "CHICubs"],
            "player_id": ["7", "7"],
            "created_utc": _utc(["2026-06-24 15:00", "2026-06-24 19:00"]),
            "compound": [1.0, -1.0],
        }
    )
    sub_fb, fb_teams = {"chicubs": "CHICubs"}, {"chicubs": frozenset({112})}
    assert dp.mention_days(m, pt, sub_fb, fb_teams)["n_mentions"].sum() == 2
    out = dp.mention_days(m, pt, sub_fb, fb_teams, pitch_times=fp)
    assert out["n_mentions"].tolist() == [1] and out["sent"].tolist() == [-1.0]
