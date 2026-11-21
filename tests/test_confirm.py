import json

import numpy as np
import pandas as pd
import pytest

from fansent import cli, confirm, correlate
from fansent.composite import fielding_index
from fansent.correlate import Scope, apply_scope, correct
from fansent.inference import (
    InferenceConfig,
    permutation_p,
    permutation_p_greater,
    permutation_p_less,
)
from fansent.model_data import panel_to_inputs
from fansent.panel import PanelConfig, build_weekly_panel
from synth import make_scores, make_stats

PCFG = PanelConfig(fanbase_teams={})


def fld_rows(pid, week, pct, errors, chances, std_pct=None, std_errors=None, w_err=None):
    base = {"player_id": pid, "full_name": pid, "period": pd.Timestamp(week)}
    return [
        {**base, "metric": "FLD_PCT", "value": pct, "std_value": std_pct or pct,
         "weight": chances, "higher_is_better": True},
        {**base, "metric": "ERRORS", "value": errors, "std_value": std_errors or errors,
         "weight": w_err or chances, "higher_is_better": False},
    ]  # fmt: skip


def three_players():
    rows = []
    for pid, pct, err in (("good", 1.0, 0.0), ("mid", 0.97, 1.0), ("bad", 0.90, 4.0)):
        rows += fld_rows(pid, "2026-04-06", pct, err, 20)
    return pd.DataFrame(rows)


def test_fld_z_orientation_and_shape():
    out = fielding_index(three_players()).set_index("player_id")
    assert out["value"]["good"] > out["value"]["mid"] > out["value"]["bad"]
    assert out["std_value"]["good"] > out["std_value"]["bad"]
    assert (out["metric"] == "FLD_Z").all() and out["higher_is_better"].all()
    assert (out["weight"] == 20).all()
    # pooled z over a symmetric-ish set: components average to ~0 overall
    assert abs(out["value"].mean()) < 1e-9


def test_fld_z_is_mean_of_component_z_and_weighted_by_chances():
    df = three_players()
    z = {}
    for m, sign in (("FLD_PCT", 1.0), ("ERRORS", -1.0)):
        v = df[df["metric"] == m].set_index("player_id")["value"] * sign
        z[m] = (v - v.mean()) / v.std()
    equal = fielding_index(df).set_index("player_id")["value"].reindex(z["FLD_PCT"].index)
    np.testing.assert_allclose(equal, (z["FLD_PCT"] + z["ERRORS"]) / 2)

    # unequal component weights (cannot happen in real data; pins the formula)
    rows = []
    for pid, pct, err in (("good", 1.0, 0.0), ("mid", 0.97, 1.0), ("bad", 0.90, 4.0)):
        rows += fld_rows(pid, "2026-04-06", pct, err, 30, w_err=10)
    skew = pd.DataFrame(rows)
    out = fielding_index(skew).set_index("player_id").reindex(z["FLD_PCT"].index)
    np.testing.assert_allclose(out["value"], (30 * z["FLD_PCT"] + 10 * z["ERRORS"]) / 40)
    np.testing.assert_allclose(out["weight"], 20.0)


def test_fld_z_std_value_is_separate_pooled_z_of_std_columns():
    rows = []
    # week-t lines rank c > b > a; season-to-date lines rank a > b > c
    for pid, pct, err, spct, serr in (
        ("a", 0.90, 4.0, 0.99, 2.0),
        ("b", 0.95, 1.0, 0.95, 9.0),
        ("c", 0.99, 0.0, 0.93, 20.0),
    ):
        rows += fld_rows(pid, "2026-04-06", pct, err, 10, std_pct=spct, std_errors=serr)
    out = fielding_index(pd.DataFrame(rows)).set_index("player_id")
    assert out["std_value"]["a"] > out["std_value"]["b"] > out["std_value"]["c"]
    assert out["value"]["c"] > out["value"]["b"] > out["value"]["a"]


def test_fld_z_single_component_week_and_missing_dropped():
    df = three_players()
    df = df[~((df["player_id"] == "bad") & (df["metric"] == "ERRORS"))]
    out = fielding_index(df)
    assert set(out["player_id"]) == {"good", "mid", "bad"}
    assert fielding_index(df[df["metric"] == "OPS"]).empty


def _stats_with_fielding():
    st = make_stats(n_players=4, weeks=6, seed=1)
    rows = []
    for i, (_, r) in enumerate(st[st["metric"] == "OPS"].iterrows()):
        for m, v, hib in (("FLD_PCT", 0.9 + 0.001 * i, True), ("ERRORS", float(i % 4), False)):
            rows.append({**r, "group": "fielding", "metric": m, "value": v,
                         "std_value": v, "weight": 10.0, "higher_is_better": hib})  # fmt: skip
    return pd.concat([st, pd.DataFrame(rows)], ignore_index=True)


def test_fld_z_is_opt_in_in_the_panel():
    stats = _stats_with_fielding()
    sent = pd.DataFrame(
        [("redsox", p, pd.Timestamp(w), 0.1, 5) for p in stats["player_id"].unique()
         for w in stats["week_start"].unique()],
        columns=["fanbase", "player_id", "period", "sent", "n_mentions"],
    )  # fmt: skip
    cfg = PanelConfig(min_weeks=2, min_units_per_fanbase=1, fanbase_teams={})
    default = build_weekly_panel(stats, sent, cfg)
    assert "FLD_Z" not in set(default["metric"])
    named = build_weekly_panel(stats, sent, cfg, ["FLD_Z"], include_composite=False)
    assert set(named["metric"]) == {"FLD_Z"}


# --- --exclude-teams ---------------------------------------------------------------------

WEEKLY = frozenset(
    {
        ("p1", "2026-04-06", 112),  # Cubs in week 1, traded to NYY for week 2
        ("p1", "2026-04-13", 147),
        ("p2", "2026-04-06", 147),
        ("p3", "2026-04-06", 111),
    }
)


def _rows():
    df = pd.DataFrame(
        [
            ("1", "reddit_arctic", "baseball", "2026-04-12", "p1"),  # Sunday of Cubs week
            ("2", "reddit_arctic", "baseball", "2026-04-13", "p1"),  # Monday: now NYY
            ("3", "reddit_arctic", "baseball", "2026-04-07", "p2"),
            ("4", "reddit_arctic", "baseball", "2026-04-08", "p3"),
            ("5", "reddit_arctic", "baseball", "2026-04-08", "p9"),  # on no roster: kept
        ],
        columns=["text_id", "source", "subreddit", "date", "player_id"],
    )
    return df.assign(compound=0.1)


def test_exclude_teams_is_by_week_of_text_date():
    out = apply_scope(_rows(), Scope(exclude_teams=frozenset({111, 112})), PCFG, WEEKLY)
    assert set(out["text_id"]) == {"2", "3", "5"}


def test_exclude_teams_is_inverse_of_teams_and_composes():
    keep = apply_scope(_rows(), Scope(teams=frozenset({111, 112})), PCFG, WEEKLY)
    drop = apply_scope(_rows(), Scope(exclude_teams=frozenset({111, 112})), PCFG, WEEKLY)
    assert set(keep["text_id"]) == {"1", "4"}
    assert set(drop["text_id"]) == {"2", "3", "5"}
    both = apply_scope(
        _rows(), Scope(teams=frozenset({112, 147}), exclude_teams=frozenset({112})), PCFG, WEEKLY
    )
    assert set(both["text_id"]) == {"2", "3"}


def test_exclude_teams_needs_rosters():
    with pytest.raises(SystemExit, match="player_teams"):
        apply_scope(_rows(), Scope(exclude_teams=frozenset({112})), PCFG, None)


# --- pooled grouping ---------------------------------------------------------------------


def _two_fanbase_panel():
    rng = np.random.default_rng(0)
    rows = []
    for fb, players in (("baseball", ["a", "b", "c"]), ("Mets", ["a", "d", "e"])):
        for p in players:
            for w in range(10):
                rows.append((fb, p, w, *rng.normal(size=4), 10.0))
    df = pd.DataFrame(rows, columns=["fanbase", "player_id", "w", "y", "perf", "std", "sent", "wt"])
    return df


def test_pool_makes_one_group_but_keeps_fanbase_player_units():
    df = _two_fanbase_panel()
    kw = {"y_next": "y", "perf": "perf", "std": "std", "sent": "sent", "weight": "wt"}
    per = panel_to_inputs(df, **kw)
    pooled = panel_to_inputs(df, pool=True, **kw)
    assert per.fanbase_names == ["Mets", "baseball"]
    assert pooled.fanbase_names == ["pooled"]
    assert (pooled.fanbase_idx_of_unit == 0).all()
    assert len(pooled.fanbase_idx_of_unit) == 6  # player "a" is two units
    assert pooled.unit_names == per.unit_names
    assert [p for _, p in pooled.unit_names].count("a") == 2
    np.testing.assert_array_equal(pooled.unit_idx, per.unit_idx)


def test_pooled_analyze_metric_reports_single_row_and_real_unit_fanbases():
    stats = make_stats(n_players=6, weeks=14, seed=2)
    scores = make_scores(stats, {"baseball": 0.0, "CHICubs": 0.0}, seed=3)
    icfg = InferenceConfig(
        n_iter=300, burn=150, perm_b=9, perm_iter=150, perm_burn=75, pool_fanbases=True
    )
    out = correlate.analyze(
        stats, scores, [], None, PanelConfig(min_weeks=8, fanbase_teams={}), icfg, ["OPS"], False
    )
    assert list(out.fanbases["fanbase"]) == ["pooled"]
    assert out.fanbases["n_units"].iloc[0] == 12
    assert set(out.units["fanbase"]) == {"baseball", "CHICubs"}
    assert len(out.units) == 12


# --- one-sided p -------------------------------------------------------------------------


def test_one_sided_permutation_p_formula():
    obs = np.array([-0.5, 0.0, 0.5])
    null = np.array([[-0.6, 0.1, 0.6], [-0.7, 0.2, 0.1], [0.1, -0.3, 0.5], [0.2, 0.0, 0.2]])
    # less counts null <= obs per column: 2, 2, 3; greater counts null >= obs: 2, 3, 2
    np.testing.assert_allclose(permutation_p_less(obs, null), [3 / 5, 3 / 5, 4 / 5])
    np.testing.assert_allclose(permutation_p_greater(obs, null), [3 / 5, 4 / 5, 3 / 5])
    assert permutation_p_less(obs, null).min() >= 1 / 5
    # two-sided is centered on the null median (-0.25 here), not on 0
    np.testing.assert_allclose(permutation_p(obs, null)[0], 1.0)


def test_alternative_selects_correction_column():
    df = pd.DataFrame(
        {
            "metric": ["OPS", "AVG"],
            "fanbase": "x",
            "perm_p": [0.9, 0.9],
            "perm_p_less": [0.01, 0.04],
            "perm_p_greater": [0.99, 0.97],
        }
    )
    two = correct(df)
    less = correct(df, "less")
    assert (two["q_bh"] == 0.9).all()
    assert less["q_bh"].max() == pytest.approx(0.04)
    assert less["p_holm"].min() == pytest.approx(0.02)
    assert correct(df, "greater")["q_bh"].min() > 0.9


def test_analyze_metric_emits_all_three_p_columns():
    stats = make_stats(n_players=6, weeks=14, seed=2)
    scores = make_scores(stats, {"baseball": 0.0}, seed=3)
    icfg = InferenceConfig(n_iter=200, burn=100, perm_b=9, perm_iter=100, perm_burn=50)
    out = correlate.analyze(
        stats, scores, [], None, PanelConfig(min_weeks=8, fanbase_teams={}), icfg, ["OPS"], False
    )
    r = out.fanbases.iloc[0]
    assert {"perm_p", "perm_p_less", "perm_p_greater"} <= set(out.fanbases.columns)
    assert 1 / 10 <= r.perm_p_less <= 1.0 and 1 / 10 <= r.perm_p_greater <= 1.0


# --- decision rule and R-hat gate --------------------------------------------------------


@pytest.mark.parametrize(
    "mu,p,rhat,expected",
    [
        (-0.05, 0.01, 1.01, confirm.CONFIRMED),
        (-0.05, 0.05, 1.01, confirm.NOT_CONFIRMED),  # p must be strictly below 0.05
        (-0.05, 0.20, 1.01, confirm.NOT_CONFIRMED),
        (0.05, 0.001, 1.01, confirm.NOT_CONFIRMED),  # right sign matters
        (-0.05, 0.01, 1.05, confirm.INVALID),  # R-hat must be strictly below 1.05
        (-0.05, 0.01, float("nan"), confirm.INVALID),
        (0.05, 0.9, 1.2, confirm.INVALID),
    ],
)
def test_decision_rule_and_rhat_gate(mu, p, rhat, expected):
    v = confirm.decide(mu, mu - 0.1, mu + 0.1, p, rhat)
    assert v.decision == expected
    assert v.rhat_ok == (rhat < 1.05)


def _fake_analyze(captured, mu, p, rhat):
    def fake(stats, scores, sidecars, oaa, pcfg, icfg, metrics, composite, pt, wt, scope):
        captured.update(icfg=icfg, metrics=metrics, composite=composite, scope=scope, oaa=oaa)
        fb = pd.DataFrame(
            {
                "metric": ["FLD_Z"], "fanbase": ["pooled"], "mu_mean": [mu], "ci_low": [mu - 0.1],
                "ci_high": [mu + 0.1], "perm_p": [2 * p], "perm_p_less": [p], "n_units": [40],
                "n_obs": [900], "rhat_max": [rhat], "null_mean": [0.03],
                "supp_perm_p_less": [0.4], "supp_perm_p": [0.7], "supp_null_mean": [0.0],
            }
        )  # fmt: skip
        return correlate.Outputs(fb, fb, fb.iloc[0:0], fb.iloc[0:0], [], {}, 0.0)

    return fake


def _data_dir(tmp_path):
    pd.DataFrame({"player_id": ["1"], "full_name": ["x"], "team_id": [147]}).to_csv(
        tmp_path / "players.csv", index=False
    )
    pd.DataFrame({"player_id": ["1"], "week_start": ["2026-04-06"], "team_id": [147]}).to_csv(
        tmp_path / "player_teams.csv", index=False
    )
    pd.DataFrame({"player_id": ["1"], "metric": ["OPS"]}).to_csv(
        tmp_path / "stats.csv", index=False
    )
    pd.DataFrame(
        {"text_id": ["t"], "source": ["r"], "subreddit": ["s"], "date": ["2026-04-06"],
         "player_id": ["1"], "compound": [0.1]}
    ).to_csv(tmp_path / "sentiment.csv", index=False)  # fmt: skip


@pytest.mark.parametrize(
    "mu,p,rhat,expected",
    [(-0.04, 0.01, 1.01, "CONFIRMED"), (-0.04, 0.3, 1.01, "NOT CONFIRMED"),
     (-0.04, 0.01, 1.2, "INVALID")],
)  # fmt: skip
def test_confirm_run_hardcodes_spec_and_writes_result(monkeypatch, tmp_path, mu, p, rhat, expected):
    _data_dir(tmp_path)
    got = {}
    monkeypatch.setattr(confirm, "analyze", _fake_analyze(got, mu, p, rhat))
    v = confirm.run(tmp_path, workers=3, seed=7)
    icfg = got["icfg"]
    assert (icfg.n_chains, icfg.n_iter, icfg.burn, icfg.perm_b) == (4, 6000, 3000, 1000)
    assert icfg.pool_fanbases and icfg.alternative == "less" and icfg.workers == 3
    assert icfg.null == "shuffle" and icfg.supplementary_null == "freedman-lane"
    assert got["metrics"] == ["FLD_Z"] and got["composite"] is False and got["oaa"] is None
    assert got["scope"].exclude_teams == frozenset({111, 112})
    assert got["scope"].exclude_fanbases == frozenset({"news"})
    assert got["scope"].teams is None
    res = json.loads((tmp_path / "confirm_result.json").read_text())
    assert v.decision.startswith(expected) and res["decision"] == v.decision
    assert res["perm_p_less"] == p and res["mu_mean"] == mu and res["seed"] == 7
    assert res["preregistration"] == "docs/preregistration.md"
    assert res["null"] == "shuffle"
    supp = res["supplementary_post_hoc_null_fix"]
    assert supp["perm_p_less"] == 0.4 and supp["perm_p_two_sided_centered"] == 0.7


def test_confirm_requires_weekly_rosters(monkeypatch, tmp_path):
    _data_dir(tmp_path)
    (tmp_path / "player_teams.csv").unlink()
    with pytest.warns(UserWarning), pytest.raises(SystemExit, match="player_teams"):
        confirm.run(tmp_path)


# --- CLI wiring --------------------------------------------------------------------------


def test_cli_new_flags_and_confirm_dispatch(monkeypatch, tmp_path):
    got = {}
    monkeypatch.setattr(correlate, "run", lambda *a: got.update(args=a))
    pd.DataFrame({"team_id": [111, 112], "team": ["BOS", "CHC"], "name": ["a", "b"]}).to_csv(
        tmp_path / "teams.csv", index=False
    )
    cli.main(["--data-dir", str(tmp_path), "correlate", "--exclude-teams", "bos", "112",
              "--pool-fanbases", "--alternative", "less", "--metrics", "FLD_Z"])  # fmt: skip
    icfg, scope = got["args"][1], got["args"][-1]
    assert scope.exclude_teams == frozenset({111, 112})
    assert icfg.pool_fanbases and icfg.alternative == "less"
    assert got["args"][2] == ["FLD_Z"]
    cli.main(["--data-dir", str(tmp_path), "correlate"])
    assert got["args"][-1].exclude_teams == frozenset()
    assert not got["args"][1].pool_fanbases and got["args"][1].alternative == "two-sided"

    seen = {}
    monkeypatch.setattr(confirm, "run", lambda d, **k: seen.update(d=d, **k))
    cli.main(["--data-dir", str(tmp_path), "confirm", "--workers", "2", "--seed", "5"])
    assert seen == {"d": tmp_path, "workers": 2, "seed": 5}
    with pytest.raises(SystemExit):
        cli.main(["confirm", "--iter", "10"])
