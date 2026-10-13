import datetime as dt
from pathlib import Path

import pytest

from fansent import stats
from fansent.oaa import load_oaa
from fansent.stats import cumulative, weekly_counts, weekly_metrics
from fansent.util import ip_to_outs

MON1, MON2, MON3 = "2026-04-06", "2026-04-13", "2026-04-20"


def _hit(date, **stat):
    base = {
        "atBats": 4,
        "hits": 1,
        "baseOnBalls": 0,
        "hitByPitch": 0,
        "sacFlies": 0,
        "totalBases": 1,
        "plateAppearances": 4,
    }
    base.update(stat)
    return {"date": date, "gameType": "R", "stat": base}


def _pit(date, ip, er=0, h=0, bb=0):
    return {
        "date": date,
        "gameType": "R",
        "stat": {"inningsPitched": ip, "earnedRuns": er, "hits": h, "baseOnBalls": bb},
    }


def _fld(date, pk, **stat):
    base = {"chances": 0, "putOuts": 0, "assists": 0, "errors": 0}
    base.update(stat)
    return {"date": date, "gameType": "R", "game": {"gamePk": pk}, "stat": base}


def _by_metric(rows):
    return {(r["week_start"], r["metric"]): r for r in rows}


def test_ip_to_outs_exact():
    assert ip_to_outs("5.2") == 17
    assert ip_to_outs("0.1") == 1
    assert ip_to_outs("7") == 21
    assert ip_to_outs("") == 0


def test_hitting_ops_and_avg():
    splits = [
        _hit("2026-04-06", hits=2, totalBases=5),
        _hit("2026-04-07", hits=1, totalBases=1, baseOnBalls=1, plateAppearances=5),
    ]
    rows, counts = weekly_metrics(splits, "hitting")
    m = _by_metric(rows)
    ops, avg = m[(MON1, "OPS")], m[(MON1, "AVG")]
    # h=3 ab=8 bb=1 tb=6 -> OBP 4/9, SLG 6/8
    assert ops["value"] == pytest.approx(4 / 9 + 6 / 8, abs=1e-4)
    assert avg["value"] == pytest.approx(3 / 8, abs=1e-4)
    assert ops["weight"] == 9 and avg["weight"] == 8
    assert ops["higher_is_better"] is True
    assert counts[0]["games"] == 2 and counts[0]["std_ab"] == 8


def test_pitching_era_whip_use_outs():
    splits = [_pit("2026-04-07", "5.2", er=2, h=5, bb=2), _pit("2026-04-12", "1.1", er=0, h=1)]
    rows, counts = weekly_metrics(splits, "pitching")
    m = _by_metric(rows)
    # 7.0 IP exactly (17 + 4 = 21 outs)
    assert counts[0]["outs"] == 21
    assert m[(MON1, "ERA")]["value"] == pytest.approx(9 * 2 / 7, abs=1e-4)
    assert m[(MON1, "WHIP")]["value"] == pytest.approx((6 + 2) / 7, abs=1e-4)
    assert m[(MON1, "ERA")]["weight"] == pytest.approx(7.0)
    assert m[(MON1, "ERA")]["higher_is_better"] is False
    assert m[(MON1, "WHIP")]["higher_is_better"] is False


def test_fielding_aggregates_positions_and_catcher_steals():
    splits = [
        # two positions in one game: one game, summed chances
        _fld("2026-04-07", 1, chances=3, putOuts=2, assists=1),
        _fld("2026-04-07", 1, chances=2, putOuts=1, assists=0, errors=1),
        _fld("2026-04-08", 2, chances=4, putOuts=4, stolenBases=2, caughtStealing=1),
    ]
    rows, counts = weekly_metrics(splits, "fielding")
    m = _by_metric(rows)
    assert counts[0]["games"] == 2
    assert counts[0]["chances"] == 9 and counts[0]["errors"] == 1
    assert m[(MON1, "FLD_PCT")]["value"] == pytest.approx(8 / 9, abs=1e-4)
    assert m[(MON1, "ERRORS")]["value"] == 1 and m[(MON1, "ERRORS")]["higher_is_better"] is False
    assert m[(MON1, "CS_PCT")]["value"] == pytest.approx(1 / 3, abs=1e-4)


def test_fielding_no_steal_attempts_means_no_cs_row():
    rows, _ = weekly_metrics([_fld("2026-04-07", 1, chances=2, putOuts=2)], "fielding")
    assert "CS_PCT" not in {r["metric"] for r in rows}


def test_non_regular_season_dropped():
    spring = {**_hit("2026-03-10"), "gameType": "S"}
    assert weekly_counts([spring], "hitting") == {}


def test_std_recomputed_from_counts_not_mean_of_weekly_rates():
    splits = [
        _hit("2026-04-06", atBats=1, hits=1, plateAppearances=1, totalBases=1),
        _hit(MON2, atBats=9, hits=0, plateAppearances=9, totalBases=0),
    ]
    m = _by_metric(weekly_metrics(splits, "hitting")[0])
    assert m[(MON1, "AVG")]["value"] == 1.0
    assert m[(MON2, "AVG")]["std_value"] == pytest.approx(0.1)  # 1/10, not (1.0 + 0)/2
    assert m[(MON2, "AVG")]["std_weight"] == 10


def test_std_has_no_leakage():
    early = [_hit("2026-04-06", hits=1), _hit(MON2, hits=2, totalBases=3)]
    late = early + [_hit(MON3, hits=4, totalBases=16), _hit("2026-04-28", hits=4, totalBases=16)]
    a = _by_metric(weekly_metrics(early, "hitting")[0])
    b = _by_metric(weekly_metrics(late, "hitting")[0])
    for key, row in a.items():
        assert b[key]["std_value"] == row["std_value"]
        assert b[key]["std_weight"] == row["std_weight"]


def test_cumulative_spans_gap_weeks():
    weeks = {dt.date(2026, 4, 6): dict.fromkeys(stats.COUNT_FIELDS, 1)}
    weeks[dt.date(2026, 4, 20)] = dict.fromkeys(stats.COUNT_FIELDS, 2)
    std = cumulative(weeks)
    assert std[dt.date(2026, 4, 6)]["ab"] == 1 and std[dt.date(2026, 4, 20)]["ab"] == 3


def test_run_with_mocked_http(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, "STATS_CSV", tmp_path / "stats.csv")
    monkeypatch.setattr(stats, "STATS_COUNTS_CSV", tmp_path / "counts.csv")
    monkeypatch.setattr(
        stats,
        "read_csv",
        lambda _p: [
            {"player_id": "1", "full_name": "A B", "team": "CHC", "is_pitcher": "False"},
            {"player_id": "2", "full_name": "C D", "team": "BOS", "is_pitcher": "True"},
        ],
    )
    logs = {
        "hitting": [_hit("2026-04-06", hits=2, totalBases=2)],
        "pitching": [_pit("2026-04-07", "6.0", er=2, h=4, bb=1)],
        "fielding": [_fld("2026-04-06", 1, chances=2, putOuts=2)],
    }
    monkeypatch.setattr(stats, "fetch_game_log", lambda pid, season, group: logs[group])
    stats.run(2026, pause=0)
    text = (tmp_path / "stats.csv").read_text()
    for metric in ("OPS", "AVG", "ERA", "WHIP", "FLD_PCT", "ERRORS"):
        assert f",{metric}," in text
    assert (tmp_path / "counts.csv").exists()


def test_load_oaa_monthly_fixture():
    rows = {
        (r["player_id"], r["month"]): r
        for r in load_oaa(Path(__file__).parent / "fixtures" / "oaa_monthly.csv")
    }
    assert len(rows) == 3  # null OAA row dropped
    crawford_apr = rows[("999001", "2026-04-01")]
    assert crawford_apr["oaa"] == 2 and crawford_apr["full_name"] == "Pete Crawford"
    assert rows[("999001", "2026-05-01")]["oaa"] == -1
    # two positions in the same month are summed
    hoerner = rows[("999002", "2026-04-01")]
    assert hoerner["oaa"] == 4 and hoerner["attempts"] == 55


def test_load_oaa_missing_column(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("player_id,month\n1,2026-04\n")
    with pytest.raises(ValueError, match="outs_above_average"):
        load_oaa(bad)
