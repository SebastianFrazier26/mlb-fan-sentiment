from pathlib import Path

import pytest

from fansent import cli, rosters
from fansent.util import DEFAULT_DATA_DIR, ENV_VAR, resolve_data_dir

FIXTURE = Path(__file__).parent / "fixtures" / "oaa_monthly.csv"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)


def test_default_is_relative_data():
    assert resolve_data_dir() == DEFAULT_DATA_DIR == Path("data")


def test_env_beats_default(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "/env/dir")
    assert resolve_data_dir() == Path("/env/dir")


def test_flag_beats_env(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "/env/dir")
    assert resolve_data_dir("/flag/dir") == Path("/flag/dir")


def test_empty_values_fall_through(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "")
    assert resolve_data_dir("") == DEFAULT_DATA_DIR


def test_tilde_expanded():
    assert "~" not in str(resolve_data_dir("~/x"))


def test_oaa_writes_into_absolute_flag_dir(tmp_path):
    out = tmp_path / "abs"
    cli.main(["--data-dir", str(out), "oaa", str(FIXTURE)])
    assert (out / "oaa_monthly.csv").exists()


def test_oaa_relative_dir_resolves_against_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["--data-dir", "rel/sub", "oaa", str(FIXTURE)])
    assert (tmp_path / "rel" / "sub" / "oaa_monthly.csv").exists()


def test_env_var_used_by_cli(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "fromenv"))
    cli.main(["oaa", str(FIXTURE)])
    assert (tmp_path / "fromenv" / "oaa_monthly.csv").exists()


def test_default_dir_is_cwd_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["oaa", str(FIXTURE)])
    assert (tmp_path / "data" / "oaa_monthly.csv").exists()


def test_rosters_writes_into_dir_with_mocked_network(tmp_path, monkeypatch):
    row = {
        "player_id": "1",
        "full_name": "A B",
        "last_name": "B",
        "team": "Cubs",
        "team_id": 112,
        "subreddit": "CHICubs",
        "position": "P",
        "is_pitcher": True,
    }
    monkeypatch.setattr(rosters, "fetch_rosters", lambda season, roster_type: [row])
    cli.main(["--data-dir", str(tmp_path), "rosters", "--season", "2026"])
    assert "A B" in (tmp_path / "players.csv").read_text()


def test_score_reads_and_writes_in_dir(tmp_path):
    (tmp_path / "texts.csv").write_text(
        "text_id,source,subreddit,date,player_id,player_name,text\n"
        "t1,reddit,CHICubs,2026-04-06,1,A B,This guy is great\n"
    )
    cli.main(["--data-dir", str(tmp_path), "score"])
    assert "t1" in (tmp_path / "sentiment.csv").read_text()
