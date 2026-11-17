import datetime as dt

import pandas as pd
import pytest

from fansent import rosters
from fansent.correlate import read_weekly_teams
from fansent.panel import PanelConfig, fanbase_of, filter_own_team, load_team_subreddits
from fansent.util import read_csv

MONDAYS = [dt.date(2026, 4, 6), dt.date(2026, 4, 13), dt.date(2026, 4, 20)]
PEOPLE = {
    1: {"id": 1, "fullName": "Traded Guy", "firstName": "Traded", "lastName": "Guy"},
    2: {
        "id": 2,
        "fullName": "Ronald Acuña Jr.",
        "firstName": "Ronald",
        "lastName": "Acuña",
        "useLastName": "Acuña Jr.",
        "nameSuffix": "Jr.",
    },
    3: {"id": 3, "fullName": "Old Timer III", "useLastName": "Timer III"},
}


def _entry(pid, pos="P"):
    return {
        "person": {"id": pid, "fullName": PEOPLE[pid]["fullName"]},
        "position": {"abbreviation": pos},
    }


def fake_api(url, params, cache, pause=0.0):
    if url.endswith("/teams"):
        return {
            "teams": [
                {"id": 112, "abbreviation": "CHC", "name": "Chicago Cubs"},
                {"id": 111, "abbreviation": "BOS", "name": "Boston Red Sox"},
            ]
        }
    if url.endswith("/people"):
        return {"people": [PEOPLE[int(i)] for i in params["personIds"].split(",")]}
    team = int(url.split("/teams/")[1].split("/")[0])
    snap = dt.date.fromisoformat(params["date"])
    assert snap.weekday() == 3 and params["rosterType"] == "40Man"
    on_bos = snap >= dt.date(2026, 4, 23)
    if team == 112:
        return {"roster": [] if on_bos else [_entry(1)]}
    return {"roster": [_entry(2, "RF")] + ([_entry(1)] if on_bos else [])}


def _run(tmp_path, **kw):
    rosters.run(2026, tmp_path, first_week=MONDAYS[0], last_week=MONDAYS[-1], fetch=fake_api, **kw)


def test_split_name_uses_api_fields_not_display_name():
    assert rosters.split_name(PEOPLE[2]) == ("Ronald", "Acuña", "Jr.")
    assert rosters.split_name(PEOPLE[1]) == ("Traded", "Guy", "")
    # No lastName: strip the suffix token from useLastName instead of keeping "III" as the surname
    assert rosters.split_name(PEOPLE[3]) == ("", "Timer", "III")


def test_weekly_rosters_and_traded_player(tmp_path):
    _run(tmp_path)
    pt = read_csv(tmp_path / "player_teams.csv")
    traded = {(r["week_start"], r["team_id"]) for r in pt if r["player_id"] == "1"}
    assert traded == {("2026-04-06", "112"), ("2026-04-13", "112"), ("2026-04-20", "111")}
    assert len([r for r in pt if r["player_id"] == "2"]) == 3
    players = {r["player_id"]: r for r in read_csv(tmp_path / "players.csv")}
    assert len(players) == 2
    assert players["1"]["team"] == "BOS" and players["1"]["team_id"] == "111"
    assert players["2"]["last_name"] == "Acuña" and players["2"]["suffix"] == "Jr."
    assert players["1"]["is_pitcher"] == "True" and players["2"]["is_pitcher"] == "False"
    assert set(players["1"]) >= {"first_name", "suffix", "subreddit", "position", "full_name"}


def test_teams_option_restricts(tmp_path):
    _run(tmp_path, teams=["CHC"])
    assert {r["team_id"] for r in read_csv(tmp_path / "player_teams.csv")} == {"112"}


def _scores(rows):
    return pd.DataFrame(
        [
            {"text_id": t, "source": "reddit_arctic", "subreddit": sub, "date": d, "player_id": "1"}
            for t, sub, d in rows
        ]
    )


WEEKLY = frozenset({("1", "2026-04-06", 112), ("1", "2026-04-13", 112), ("1", "2026-04-20", 111)})


def test_own_team_filter_is_by_week_of_mention():
    rows = _scores(
        [
            ("pre_cubs", "CHICubs", "2026-04-15"),
            ("post_cubs", "CHICubs", "2026-04-22"),
            ("pre_sox", "redsox", "2026-04-19"),
            ("post_sox", "redsox", "2026-04-20"),
            ("sunday_sox", "redsox", "2026-04-26"),
        ]
    )
    kept, dropped = filter_own_team(rows, PanelConfig(), None, WEEKLY)
    assert kept["text_id"].tolist() == ["pre_cubs", "post_sox", "sunday_sox"]
    assert dropped == {"CHICubs": 1, "redsox": 1}


def test_multi_sub_fanbase_mapping_and_filter():
    m = load_team_subreddits()
    assert m["mets"] == m["newyorkmets"] == "NewYorkMets"
    assert m["sacramentoathletics"] == m["oaklandathletics"] == "Athletics"
    assert "letsgofish" not in m and m["miamimarlins"] == "miamimarlins"
    cfg = PanelConfig()
    df = _scores([("a", "Mets", "2026-04-06"), ("b", "NewYorkMets", "2026-04-06")])
    assert set(fanbase_of(df["source"], df["subreddit"], cfg.subreddit_fanbase)) == {"NewYorkMets"}
    assert cfg.fanbase_teams["newyorkmets"] == frozenset({121})
    mets = {("1", "2026-04-06", 121)}
    kept, dropped = filter_own_team(df, cfg, None, frozenset(mets))
    assert len(kept) == 2 and dropped == {}


def test_missing_player_teams_warns_and_falls_back(tmp_path):
    with pytest.warns(UserWarning, match="falls back to players.csv"):
        assert read_weekly_teams(tmp_path / "player_teams.csv") is None
    rows = _scores([("a", "redsox", "2026-04-06"), ("b", "CHICubs", "2026-04-06")])
    kept, dropped = filter_own_team(rows, PanelConfig(), {"1": 111}, None)
    assert kept["text_id"].tolist() == ["a"] and dropped == {"CHICubs": 1}


def test_read_weekly_teams_roundtrip(tmp_path):
    _run(tmp_path)
    got = read_weekly_teams(tmp_path / "player_teams.csv")
    assert ("1", "2026-04-20", 111) in got and ("1", "2026-04-20", 112) not in got
