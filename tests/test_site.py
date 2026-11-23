import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from fansent import cli, site
from fansent.correlate import FANBASE_COLS
from fansent.site import SnapshotError, validate_snapshot

PUBLISHED = Path(__file__).resolve().parents[1] / "results" / "published"


def _run_dir(tmp_path):
    rows = [
        ["COMPOSITE", False, "COMPOSITE", "CHICubs", -0.137, -0.245, -0.029, 0.007, 14 / 1001],
        ["COMPOSITE", False, "COMPOSITE", "redsox", 0.063, -0.069, 0.197, 0.829, 634 / 1001],
        ["metrics", False, "OPS", "CHICubs", -0.118, -0.252, 0.017, 0.044, 26 / 1001],
        ["exploratory", True, "CS_PCT", "redsox", 0.01, -0.3, 0.3, 0.5, 501 / 1001],
    ]
    df = pd.DataFrame(rows, columns=FANBASE_COLS[:9])
    df["perm_p_less"] = df["perm_p"] / 2
    df["perm_p_greater"] = 1 - df["perm_p"] / 2
    df["q_bh"] = [0.028, 0.633, 0.156, float("nan")]
    df["p_holm"] = [0.028, 0.633, 0.3, float("nan")]
    df["n_units"] = [23, 23, 13, 4]
    df["n_obs"] = [383, 348, 253, 40]
    df["rhat_max"] = [1.03, 1.03, 1.008, 1.01]
    d = tmp_path / "run"
    d.mkdir()
    df[FANBASE_COLS].to_csv(d / "analysis_fanbases.csv", index=False)
    return d


@pytest.fixture
def snap(tmp_path):
    return site.snapshot_from_run(_run_dir(tmp_path), "2026-10-13-test", 2026, 1000)


def _strings(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)
    elif isinstance(x, str):
        yield x


def test_committed_snapshots_are_valid_and_aggregate_only():
    paths = sorted(PUBLISHED.glob("*.json"))
    assert paths
    allowed = set(site.KNOWN_METRICS) | set(site.KNOWN_FAMILIES) | site.TOP_KEYS | site.ROW_KEYS
    allowed |= {"bh", "holm", *site.STATUSES, *site.NULLS, *site.SCOPES, site.CONFIRM_METRIC}
    allowed |= site.TOP_KEYS_V2 | site.SECTION_KEYS | site.CONFIRM_KEYS
    for path in paths:
        snap = site.load_snapshot(path)
        assert path.stem == snap["run_id"]
        for s in _strings(snap):
            assert (
                s in allowed or site.FANBASE_RE.match(s) or s in (snap["run_id"], snap["created"])
            ), s
            assert " " not in s


def test_snapshot_from_run_keeps_only_summary_columns(snap, tmp_path):
    assert set(snap) == site.TOP_KEYS
    assert all(set(r) == site.ROW_KEYS for r in snap["rows"])
    cs = next(r for r in snap["rows"] if r["metric"] == "CS_PCT")
    assert cs["q_bh"] is None and cs["exploratory"] is True
    path = site.write_snapshot(snap, tmp_path / "pub")
    assert site.load_snapshot(path) == snap


def test_snapshot_ignores_extra_run_columns(tmp_path):
    d = _run_dir(tmp_path)
    df = pd.read_csv(d / "analysis_fanbases.csv")
    df["text"] = "some comment"
    df.to_csv(d / "analysis_fanbases.csv", index=False)
    snap = site.snapshot_from_run(d, "x", 2026, 1000)
    assert "some comment" not in json.dumps(snap)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["rows"][0].update(text="great game by him"),
        lambda s: s.update(note="free text"),
        lambda s: s["rows"][0].update(fanbase="u/some user"),
        lambda s: s["rows"][0].update(metric="Seiya Suzuki"),
        lambda s: s["rows"][0].update(family="anything"),
        lambda s: s["rows"][0].update(mu_mean=float("nan")),
        lambda s: s["rows"][0].update(mu_mean=5.0),
        lambda s: s["rows"][0].update(perm_p=0.0123),
        lambda s: s["rows"][0].update(q_bh=None),
        lambda s: s["rows"][0].update(n_obs=True),
        lambda s: s.update(run_id="a run with spaces"),
        lambda s: s.update(created="yesterday"),
        lambda s: s.update(perm_b=999),
        lambda s: s.update(rows=[]),
        lambda s: s.update(status="draft"),
        lambda s: s.update(status=None),
        lambda s: s.pop("status"),
    ],
)
def test_validate_rejects(snap, mutate):
    bad = copy.deepcopy(snap)
    mutate(bad)
    with pytest.raises(SnapshotError):
        validate_snapshot(bad)


def test_render_html_sections_and_no_script(snap):
    page = site.render_html(snap)
    for section in ("h-headline", "h-forest", "h-perm", "h-method", "h-caveats"):
        assert f'id="{section}"' in page
    assert page.count('role="img"') == 2
    assert "<desc " in page and "<table>" in page
    assert "<script" not in page.lower()
    assert "frame-ancestors" not in page and "x-frame-options" not in page.lower()
    assert "prefers-color-scheme:dark" in page
    assert site.REPO_URL in page
    assert "r/CHICubs" in page and "Upbeat weeks preceded <strong>worse</strong> play" in page
    assert "Caught stealing %" in page


def test_status_defaults_to_preliminary_and_drives_banner(snap):
    assert snap["status"] == "preliminary"
    page = site.render_html(snap)
    assert 'class="prelim"' in page and "<strong>Preliminary results</strong>" in page
    assert (
        "the 2026-10-13 run" in page
        and "baseline sentiment scorer" in page
        and "will change" in page
    )
    assert "<title>Are the Fans Right? (preliminary)</title>" in page
    assert 'id="h-caveats"' in page
    final = site.render_html({**snap, "status": "final"})
    assert 'class="prelim"' not in final and "(preliminary)" not in final


def test_banner_without_dated_run_id_names_the_run(snap):
    page = site.render_html({**snap, "run_id": "teams-clean"})
    assert "run teams-clean" in page


def test_render_escapes_labels(snap):
    snap["run_id"] = "a<b"
    with pytest.raises(SnapshotError):
        site.render_html(snap)


def test_cli_site_builds_from_snapshot(snap, tmp_path, capsys):
    path = site.write_snapshot(snap, tmp_path / "pub")
    out = tmp_path / "site"
    cli.main(["site", "--snapshot", str(path), "--out", str(out)])
    assert (out / "index.html").read_text().startswith("<!doctype html>")
    assert json.loads((out / "data.json").read_text()) == snap


def test_cli_snapshot(tmp_path):
    d = _run_dir(tmp_path)
    cli.main(["snapshot", str(d), "--out-dir", str(tmp_path / "pub")])
    assert site.load_snapshot(tmp_path / "pub" / "run.json")["status"] == "preliminary"
    cli.main(["snapshot", str(d), "--status", "final", "--out-dir", str(tmp_path / "fin")])
    assert site.load_snapshot(tmp_path / "fin" / "run.json")["status"] == "final"


def test_latest_snapshot_picks_newest_run(snap, tmp_path):
    for rid in ("2026-09-01-a", "2026-10-13-b"):
        site.write_snapshot({**snap, "run_id": rid}, tmp_path)
    assert site.latest_snapshot(tmp_path).name == "2026-10-13-b.json"


def _confirm_json(tmp_path, **over):
    c = {
        "preregistration": "docs/preregistration.md",
        "metric": "FLD_Z",
        "decision": "NOT CONFIRMED",
        "mu_mean": -0.0023,
        "ci_low": -0.022,
        "ci_high": 0.017,
        "perm_p_less": 183 / 1001,
        "rhat_max": 1.005,
        "n_units": 651,
        "n_obs": 11818,
        "excluded_fanbases": ["news"],
        **over,
    }
    path = tmp_path / "confirm_result.json"
    path.write_text(json.dumps(c))
    return path


@pytest.fixture
def snap2(tmp_path):
    d = _run_dir(tmp_path)
    return site.snapshot_from_runs(
        {"league": d, "demo": d},
        "2026-11-22-test",
        2026,
        1000,
        "freedman-lane",
        _confirm_json(tmp_path),
        status="final",
    )


def test_v2_snapshot_holds_only_closed_vocabulary(snap2, tmp_path):
    assert set(snap2) == site.TOP_KEYS_V2
    assert [s["scope"] for s in snap2["sections"]] == ["league", "demo"]
    assert all(set(r) == site.ROW_KEYS for s in snap2["sections"] for r in s["rows"])
    assert set(snap2["confirmatory"]) == site.CONFIRM_KEYS
    assert snap2["confirmatory"]["confirmed"] is False
    dumped = json.dumps(snap2)
    assert "preregistration.md" not in dumped and "excluded_fanbases" not in dumped
    path = site.write_snapshot(snap2, tmp_path / "pub")
    assert site.load_snapshot(path) == snap2


def test_v2_confirm_must_be_a_decided_fld_z_test(tmp_path):
    d = _run_dir(tmp_path)
    for over in ({"metric": "OPS"}, {"decision": "INVALID"}):
        with pytest.raises(SnapshotError):
            site.snapshot_from_runs(
                {"league": d}, "x", 2026, 1000, "shuffle", _confirm_json(tmp_path, **over)
            )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(rows=[]),
        lambda s: s.update(null="made up"),
        lambda s: s.update(sections=[]),
        lambda s: s["sections"][0].update(scope="some other scope"),
        lambda s: s["sections"][1].update(scope="league"),
        lambda s: s["sections"][0].update(title="free text"),
        lambda s: s["sections"][0]["rows"][0].update(text="great game by him"),
        lambda s: s["sections"][0].update(rows=[]),
        lambda s: s["confirmatory"].update(note="free text"),
        lambda s: s["confirmatory"].update(metric="Seiya Suzuki"),
        lambda s: s["confirmatory"].update(confirmed="no"),
        lambda s: s["confirmatory"].update(perm_p_less=0.0123),
        lambda s: s["confirmatory"].update(mu_mean=5.0),
        lambda s: s["confirmatory"].update(n_obs=True),
        lambda s: s.pop("confirmatory"),
    ],
)
def test_v2_validate_rejects(snap2, mutate):
    bad = copy.deepcopy(snap2)
    mutate(bad)
    with pytest.raises(SnapshotError):
        validate_snapshot(bad)


def test_v2_render(snap2):
    page = site.render_html(snap2)
    for section in ("h-confirm", "h-league", "h-demo", "h-method", "h-caveats"):
        assert f'id="{section}"' in page
    assert page.count('role="img"') == 4
    for key in ("forest-league", "perm-league", "forest-demo", "perm-demo"):
        assert page.count(f'id="{key}-t"') == 1
    assert "<script" not in page.lower()
    assert "Pre-registered test: not confirmed" in page and "no evidence" in page
    assert "Freedman-Lane permutation" in page
    assert 'class="prelim"' not in page and "(preliminary)" not in page
    assert 'class="prelim"' in site.render_html({**snap2, "status": "preliminary"})
    no_confirm = site.render_html({**snap2, "confirmatory": None})
    assert 'id="h-confirm"' not in no_confirm and "no evidence" not in no_confirm


def test_pooled_fanbase_is_not_labelled_as_a_subreddit():
    assert site.fanbase_label("pooled") == "All fanbases, pooled"


def test_cli_snapshot_sections(tmp_path):
    d = _run_dir(tmp_path)
    pub = tmp_path / "pub"
    args = ["snapshot", "--section", f"league={d}", "--confirm", str(_confirm_json(tmp_path))]
    cli.main([*args, "--run-id", "2026-11-22-x", "--status", "final", "--out-dir", str(pub)])
    snap = site.load_snapshot(pub / "2026-11-22-x.json")
    assert snap["schema_version"] == 2 and snap["null"] == "freedman-lane"
    with pytest.raises(SystemExit):
        cli.main([*args, "--out-dir", str(pub)])
