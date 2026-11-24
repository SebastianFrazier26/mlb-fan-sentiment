from fansent import schedule


def _game(pk, date, utc, state="F"):
    return {
        "gamePk": pk,
        "officialDate": date,
        "gameDate": utc,
        "doubleHeader": "N",
        "gameNumber": 1,
        "status": {"codedGameState": state},
        "teams": {"home": {"team": {"id": 112}}, "away": {"team": {"id": 138}}},
    }


def test_parse_keeps_final_games_sorted():
    data = {
        "dates": [
            {
                "games": [
                    _game(2, "2026-05-02", "2026-05-02T18:00:00Z"),
                    _game(3, "2026-05-02", "2026-05-02T17:00:00Z", state="D"),
                    _game(1, "2026-05-01", "2026-05-01T23:00:00Z"),
                ]
            }
        ]
    }
    rows = schedule.parse(data)
    assert [r["game_pk"] for r in rows] == [1, 2]
    assert rows[0]["home_id"] == 112
