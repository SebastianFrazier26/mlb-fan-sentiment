import numpy as np
import pandas as pd
import pytest

from fansent.composite import composite_index
from fansent.panel import (
    PanelConfig,
    build_weekly_panel,
    fanbase_of,
    oriented,
    sentiment_periods,
)


def scores(rows):
    df = pd.DataFrame(
        rows, columns=["text_id", "source", "subreddit", "date", "player_id", "compound"]
    )
    df["player_name"] = "x"
    return df


def stat_rows(pid, metric, weeks, values, weights=None, hib=True, group="hitting"):
    weights = weights or [20] * len(weeks)
    return [
        {
            "player_id": pid,
            "full_name": f"P{pid}",
            "team": "T",
            "season": 2026,
            "week_start": w,
            "group": group,
            "metric": metric,
            "value": v,
            "std_value": v,
            "weight": wt,
            "std_weight": 100,
            "higher_is_better": hib,
        }
        for w, v, wt in zip(weeks, values, weights, strict=True)
    ]


def test_fanbase_mapping():
    src = pd.Series(["reddit_arctic", "gdelt_headline", "reddit_arctic"])
    sub = pd.Series(["CHICubs", "", None])
    assert fanbase_of(src, sub).tolist() == ["CHICubs", "news", "reddit_arctic"]


def test_weekly_windows_mon_to_sun():
    s = scores(
        [
            ("a", "reddit_arctic", "redsox", "2026-04-06", "1", 0.2),  # Mon
            ("b", "reddit_arctic", "redsox", "2026-04-12", "1", 0.4),  # Sun, same week
            ("c", "reddit_arctic", "redsox", "2026-04-13", "1", -1.0),  # next Mon
        ]
    )
    weekly, monthly = sentiment_periods(s, [], PanelConfig())
    w = weekly.set_index("period")
    assert w.loc["2026-04-06", "sent"] == pytest.approx(0.3)
    assert w.loc["2026-04-06", "n_mentions"] == 2
    assert w.loc["2026-04-13", "sent"] == pytest.approx(-1.0)
    assert monthly["period"].nunique() == 1


def test_sidecar_missing_means_explicit_and_inherited_weighting():
    s = scores(
        [
            ("a", "reddit_arctic", "redsox", "2026-04-06", "1", 0.8),
            ("b", "reddit_arctic", "redsox", "2026-04-07", "1", 0.2),
        ]
    )
    side = pd.DataFrame(
        {
            "text_id": ["b"],
            "player_id": ["1"],
            "attribution": ["reply_inherited"],
            "parent_id": ["z"],
        }
    )
    base, _ = sentiment_periods(s, [], PanelConfig())
    assert base["sent"].iloc[0] == pytest.approx(0.5)
    half, _ = sentiment_periods(s, [side], PanelConfig(reply_inherited_weight=0.5))
    assert half["sent"].iloc[0] == pytest.approx((0.8 + 0.1) / 1.5)
    assert half["n_mentions"].iloc[0] == 2
    drop, _ = sentiment_periods(s, [side], PanelConfig(reply_inherited_weight=0))
    assert drop["sent"].iloc[0] == pytest.approx(0.8) and drop["n_mentions"].iloc[0] == 1


def weekly_sent(pid, weeks, fanbase="redsox", n=5):
    return pd.DataFrame(
        {
            "fanbase": fanbase,
            "player_id": pid,
            "period": pd.to_datetime(weeks),
            "sent": 0.1,
            "n_mentions": n,
        }
    )


CFG = PanelConfig(min_weeks=1, min_units_per_fanbase=1)


def test_consecutive_weeks_and_thresholds():
    weeks = ["2026-04-06", "2026-04-13", "2026-04-27", "2026-05-04", "2026-05-11"]
    # week 04-20 missing; 05-04 below the 10 PA floor
    stats = pd.DataFrame(
        stat_rows("1", "OPS", weeks, [0.7, 0.8, 0.9, 1.0, 1.1], [20, 20, 20, 5, 20])
    )
    sent = weekly_sent("1", weeks)
    p = build_weekly_panel(stats, sent, CFG, include_composite=False)
    # Only 04-06 -> 04-13 qualifies: 04-13 has no successor, 04-27's successor is below floor,
    # 05-04 is ineligible as t, 05-11 has no successor.
    assert p["period"].dt.strftime("%Y-%m-%d").tolist() == ["2026-04-06"]
    assert p["y_next"].tolist() == [0.8] and p["perf"].tolist() == [0.7]


def test_mention_and_week_minimums():
    weeks = ["2026-04-06", "2026-04-13", "2026-04-20"]
    stats = pd.DataFrame(stat_rows("1", "OPS", weeks, [0.7, 0.8, 0.9]))
    sent = weekly_sent("1", weeks)
    sent.loc[0, "n_mentions"] = 2
    cfg = PanelConfig(min_weeks=1, min_units_per_fanbase=1)
    p = build_weekly_panel(stats, sent, cfg, include_composite=False)
    assert len(p) == 1 and p["period"].iloc[0] == pd.Timestamp("2026-04-13")
    none = build_weekly_panel(
        stats, sent, PanelConfig(min_weeks=2, min_units_per_fanbase=1), include_composite=False
    )
    assert none.empty
    sparse = build_weekly_panel(
        stats, sent, PanelConfig(min_weeks=1, min_units_per_fanbase=2), include_composite=False
    )
    assert sparse.empty


def test_weight_is_next_week_and_orientation():
    weeks = ["2026-04-06", "2026-04-13"]
    stats = pd.DataFrame(
        stat_rows("2", "ERA", weeks, [4.0, 2.0], [6.0, 9.0], hib=False, group="pitching")
    )
    p = build_weekly_panel(stats, weekly_sent("2", weeks), CFG, include_composite=False)
    assert p["weight"].iloc[0] == 9.0
    o = oriented(p)
    assert o["y_next"].iloc[0] == -2.0 and o["perf"].iloc[0] == -4.0
    assert p["y_next"].iloc[0] == 2.0  # raw panel untouched


def _two_weeks_stats():
    wk = ["2026-04-06", "2026-04-13"]
    rows = []
    # hitters: A good OPS/AVG, few errors; B bad OPS/AVG, many errors
    for pid, ops, avg, err in (
        ("A", [0.9, 0.95], [0.30, 0.30], [0, 0]),
        ("B", [0.5, 0.5], [0.2, 0.2], [3, 3]),
    ):
        rows += stat_rows(pid, "OPS", wk, ops)
        rows += stat_rows(pid, "AVG", wk, avg)
        rows += stat_rows(pid, "ERRORS", wk, err, hib=False, group="fielding")
    # pitchers: C low ERA/WHIP (good), D high (bad)
    for pid, era, whip in (("C", [2.0, 2.0], [1.0, 1.0]), ("D", [6.0, 6.0], [1.6, 1.6])):
        rows += stat_rows(pid, "ERA", wk, era, hib=False, group="pitching")
        rows += stat_rows(pid, "WHIP", wk, whip, hib=False, group="pitching")
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["week_start"])
    # hitters have PA, pitchers none
    df["std_weight"] = np.where(df["player_id"].isin(["A", "B"]), 100, 0)
    return df


def test_composite_orientation():
    st = _two_weeks_stats()
    weights = {
        "hitter": {"OPS": 1.0, "AVG": 1.0, "ERRORS": 1.0},
        "pitcher": {"ERA": 1.0, "WHIP": 1.0},
    }
    comp = composite_index(st, weights, 20.0, st)
    v = comp.groupby("player_id")["value"].mean()
    assert v["A"] > 0 > v["B"]  # fewer errors and higher OPS both read as better
    assert v["C"] > 0 > v["D"]  # lower ERA/WHIP reads as better
    assert (comp["metric"] == "COMPOSITE").all() and comp["higher_is_better"].all()


def test_composite_needs_anchor_and_role():
    st = _two_weeks_stats()
    only_field = st[~st["metric"].isin(["OPS", "AVG"]) | ~st["player_id"].isin(["B"])]
    weights = {"hitter": {"OPS": 1.0, "ERRORS": 1.0}, "pitcher": {"ERA": 1.0}}
    comp = composite_index(only_field, weights, 20.0, st)
    assert "B" not in set(comp["player_id"]) and "A" in set(comp["player_id"])
    only = st[(st["player_id"] == "B") & (st["metric"] == "ERRORS")]
    assert composite_index(only, weights, 20.0, st).empty


def test_oaa_panel_consecutive_months_and_cumulative_std():
    from fansent.panel import build_oaa_panel

    oaa = pd.DataFrame(
        {
            "player_id": ["1"] * 5,
            "full_name": "P",
            "month": ["2026-04-01", "2026-05-01", "2026-06-01", "2026-08-01", "2026-09-01"],
            "oaa": [2.0, -1.0, 3.0, 1.0, 4.0],
            "attempts": [40, 35, 5, 30, 30],  # June below the 10-attempt floor
        }
    )
    sent = pd.DataFrame(
        {
            "fanbase": "redsox",
            "player_id": "1",
            "period": pd.to_datetime(oaa["month"]),
            "sent": 0.1,
            "n_mentions": 20,
        }
    )
    cfg = PanelConfig(min_months=1, min_units_per_fanbase=1)
    p = build_oaa_panel(oaa, sent, cfg)
    got = p.sort_values("period")[["period", "perf", "y_next", "std"]]
    assert got["period"].dt.strftime("%m").tolist() == ["04", "08"]
    assert got["perf"].tolist() == [2.0, 1.0] and got["y_next"].tolist() == [-1.0, 4.0]
    assert got["std"].tolist() == [2.0, 1.0 + 2.0 - 1.0]  # cumulative over eligible months


def test_default_composite_weights():
    from fansent.composite import DEFAULT_COMPOSITE_WEIGHTS as W

    assert W["hitter"] == {"OPS": 0.375, "AVG": 0.375, "FLD_PCT": 0.125, "ERRORS": 0.125}
    assert W["pitcher"] == {"ERA": 0.5, "WHIP": 0.5}
    assert PanelConfig().composite_weights == W
    assert PanelConfig().reply_inherited_weight == 1.0
