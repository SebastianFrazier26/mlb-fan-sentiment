import numpy as np
import pandas as pd
import pytest

from fansent import cli, correlate
from fansent.correlate import Scope, apply_scope, correct, family_of, format_table, resolve_teams
from fansent.inference import InferenceConfig
from fansent.multitest import benjamini_hochberg, holm
from fansent.panel import PanelConfig


def _fb(rows):
    return pd.DataFrame(rows, columns=["metric", "fanbase", "perm_p"])


def test_family_assignment():
    assert [family_of(m) for m in ("OPS", "AVG", "ERA", "WHIP", "FLD_PCT", "ERRORS")] == [
        "metrics"
    ] * 6
    assert family_of("COMPOSITE") == "COMPOSITE"
    assert family_of("OAA") == "OAA"
    assert family_of("CS_PCT") == "exploratory"


def test_q_values_computed_within_family_only():
    rows = (
        [("OPS", "a", 0.01), ("ERA", "a", 0.04), ("AVG", "b", 0.03)]
        + [("COMPOSITE", "a", 0.02), ("COMPOSITE", "b", 0.5)]
        + [("OAA", "a", 0.03)]
    )
    out = correct(_fb(rows))
    m = out[out["family"] == "metrics"]
    np.testing.assert_allclose(m["q_bh"], benjamini_hochberg([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(m["p_holm"], holm([0.01, 0.04, 0.03]))
    c = out[out["family"] == "COMPOSITE"]
    np.testing.assert_allclose(c["q_bh"], benjamini_hochberg([0.02, 0.5]))
    # a lone-test family is uncorrected, unaffected by the other families
    oaa = out[out["family"] == "OAA"]
    assert oaa["q_bh"].iloc[0] == 0.03 and oaa["p_holm"].iloc[0] == 0.03


def test_exploratory_excluded_from_correction():
    rows = [("OPS", "a", 0.01), ("CS_PCT", "a", 0.001), ("CS_PCT", "b", 0.002)]
    out = correct(_fb(rows))
    cs = out[out["metric"] == "CS_PCT"]
    assert cs["exploratory"].all() and cs[["q_bh", "p_holm"]].isna().all().all()
    assert (cs["perm_p"] == [0.001, 0.002]).all()
    ops = out[out["metric"] == "OPS"].iloc[0]
    assert not ops["exploratory"] and ops["q_bh"] == 0.01


def test_table_grouped_by_family():
    fb = correct(_fb([("OPS", "a", 0.01), ("CS_PCT", "a", 0.2), ("COMPOSITE", "a", 0.03)]))
    for c in ("mu_mean", "ci_low", "ci_high", "p_pos", "rhat_max"):
        fb[c] = 0.1
    text = format_table(fb, "bh", 0.05)
    assert text.index("family: COMPOSITE") < text.index("family: metrics")
    assert text.index("family: metrics") < text.index("family: exploratory")
    assert "NA" in text


PCFG = PanelConfig(fanbase_teams={})


def _scores():
    # 2026-04-06 is a Monday: 04-12 (Sunday) is in the same week, 04-13 starts the next.
    rows = [
        ("1", "reddit_arctic", "NYYankees", "2026-04-12", "p1"),
        ("2", "reddit_arctic", "redsox", "2026-04-07", "p2"),
        ("3", "gdelt_headline", "", "2026-04-08", "p1"),
        ("4", "reddit_arctic", "NYYankees", "2026-04-13", "p1"),
    ]
    df = pd.DataFrame(rows, columns=["text_id", "source", "subreddit", "date", "player_id"])
    return df.assign(compound=0.1)


def test_include_and_exclude_fanbases():
    s = _scores()
    only = apply_scope(s, Scope(fanbases=frozenset({"nyyankees"})), PCFG, None)
    assert set(only["text_id"]) == {"1", "4"}
    no_news = apply_scope(s, Scope(exclude_fanbases=frozenset({"News"})), PCFG, None)
    assert set(no_news["text_id"]) == {"1", "2", "4"}
    both = apply_scope(
        s,
        Scope(fanbases=frozenset({"NYYankees", "news"}), exclude_fanbases=frozenset({"news"})),
        PCFG,
        None,
    )
    assert set(both["text_id"]) == {"1", "4"}


def test_teams_filter_is_by_week_of_text_date():
    weekly = frozenset(
        {("p1", "2026-04-06", 147), ("p1", "2026-04-13", 111), ("p2", "2026-04-06", 111)}
    )
    out = apply_scope(_scores(), Scope(teams=frozenset({147})), PCFG, weekly)
    assert set(out["text_id"]) == {"1", "3"}
    both = apply_scope(_scores(), Scope(teams=frozenset({147, 111})), PCFG, weekly)
    assert set(both["text_id"]) == {"1", "2", "3", "4"}


def test_teams_needs_weekly_rosters_and_empty_scope_errors():
    with pytest.raises(SystemExit, match="player_teams"):
        apply_scope(_scores(), Scope(teams=frozenset({147})), PCFG, None)
    with pytest.raises(SystemExit, match="no sentiment rows"):
        apply_scope(_scores(), Scope(fanbases=frozenset({"nope"})), PCFG, None)


def test_resolve_teams_accepts_ids_and_abbreviations(tmp_path):
    path = tmp_path / "teams.csv"
    pd.DataFrame({"team_id": [147, 111], "team": ["NYY", "BOS"], "name": ["a", "b"]}).to_csv(
        path, index=False
    )
    assert resolve_teams(["nyy", "111", "147"], path) == {147, 111}
    with pytest.raises(SystemExit, match="unknown team"):
        resolve_teams(["XXX"], path)
    assert resolve_teams(["147"], tmp_path / "missing.csv") == {147}


@pytest.mark.parametrize("pool,expected", [("league", {"p1", "p2", "p3"}), ("teams", {"p1"})])
def test_pool_stats_restricts_composite_pool(monkeypatch, pool, expected):
    seen = {}

    def fake(stats, *a, **k):
        seen["players"] = set(stats["player_id"].astype(str))
        raise SystemExit("stop")

    monkeypatch.setattr(correlate, "build_weekly_panel", fake)
    stats = pd.DataFrame({"player_id": ["p1", "p2", "p3"]})
    weekly = frozenset({("p1", "2026-04-06", 147)})
    scope = Scope(teams=frozenset({147}), pool_stats=pool)
    with pytest.raises(SystemExit, match="stop"):
        correlate.analyze(
            stats, _scores(), [], None, PCFG, InferenceConfig(), None, True, None, weekly, scope
        )
    assert seen["players"] == expected


def test_cli_fit_defaults_and_overrides(monkeypatch, tmp_path):
    got = {}
    monkeypatch.setattr(correlate, "run", lambda *a: got.update(args=a))
    cli.main(["--data-dir", str(tmp_path), "correlate"])
    icfg, scope = got["args"][1], got["args"][-1]
    assert (icfg.n_chains, icfg.n_iter, icfg.burn) == (4, 6000, 3000)
    assert (icfg.perm_iter, icfg.perm_burn) == (600, 300)
    assert scope == Scope()

    pd.DataFrame({"team_id": [147], "team": ["NYY"], "name": ["x"]}).to_csv(
        tmp_path / "teams.csv", index=False
    )
    cli.main(
        ["--data-dir", str(tmp_path), "correlate", "--chains", "2", "--iter", "100", "--burn", "50",
         "--perm-iter", "40", "--perm-burn", "20", "--teams", "NYY", "--pool-stats", "teams",
         "--exclude-fanbases", "news"]
    )  # fmt: skip
    icfg, scope = got["args"][1], got["args"][-1]
    assert (icfg.n_chains, icfg.n_iter, icfg.burn) == (2, 100, 50)
    assert (icfg.perm_iter, icfg.perm_burn) == (40, 20)
    assert scope == Scope(
        teams=frozenset({147}), exclude_fanbases=frozenset({"news"}), pool_stats="teams"
    )
    with pytest.raises(SystemExit):
        cli.main(["--data-dir", str(tmp_path), "correlate", "--iter", "10", "--burn", "10"])
