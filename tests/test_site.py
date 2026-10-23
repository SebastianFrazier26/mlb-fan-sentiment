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
    allowed |= {"bh", "holm", *site.STATUSES}
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
