import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from fansent import cli, figures, site

PUBLISHED = Path(__file__).resolve().parents[1] / "results" / "published"
FINAL = PUBLISHED / "2026-11-22-final.json"
IMG = Path(__file__).resolve().parents[1] / "docs" / "img"


def test_figures_are_standalone_svg_without_script(tmp_path):
    paths = figures.write_figures(FINAL, tmp_path)
    assert [p.name for p in paths] == list(figures.FIGURES)
    for path in paths:
        text = path.read_text()
        root = ET.fromstring(text)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert "<script" not in text and "<style" not in text and "href" not in text
        assert 'fill="#ffffff"' in text


def test_confirmatory_figure_states_the_registered_result():
    snap = site.load_snapshot(FINAL)
    text = figures.confirmatory_figure(snap)
    assert "not confirmed (one-sided p = 0.183)" in text
    assert "Rest of league (pre-registered)" in text and "-0.002" in text
    assert text.count("Two teams:") == 4


def test_league_figure_has_one_row_per_corrected_metric():
    text = figures.league_figure(site.load_snapshot(FINAL))
    assert text.count("<circle") == 7 and "Caught stealing" not in text


def test_committed_figures_match_the_snapshot():
    for name, build in figures.FIGURES.items():
        assert (IMG / name).read_text() == build(site.load_snapshot(FINAL)), name


def test_figures_need_a_schema_2_snapshot(tmp_path):
    with pytest.raises(site.SnapshotError):
        figures.write_figures(PUBLISHED / "2026-10-13-teams-clean.json", tmp_path)


def test_cli_figures(tmp_path, capsys):
    cli.main(["figures", "--snapshot", str(FINAL), "--out", str(tmp_path)])
    assert (tmp_path / "league.svg").exists()
    assert "wrote" in capsys.readouterr().out
