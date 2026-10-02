# Changelog

## 2026-10-01

- `stats` rewritten: tidy long `stats.csv` (`player_id, full_name, team, season, week_start,
  group, metric, value, std_value, weight, std_weight, higher_is_better`) with OPS, AVG, ERA,
  WHIP and fielding metrics (FLD_PCT, ERRORS, CS_PCT), plus season-to-date `std_value`
  recomputed from counts. New wide `stats_counts.csv` of raw weekly and `std_` counts.
  Replaces the old `games, metric, value, sample` schema; `analyze.py` reads `weight`.
- Added `fansent oaa PATH` loader for Baseball Savant monthly Outs Above Average exports.
- CLI `--season` defaults now 2026.

- Split data collectors (`reddit-dump`, `reddit-live`, `news`) into the private
  `fansent-collect` repo; `fansent` keeps `rosters`, `stats`, `score`, `correlate`
  and consumes the collectors' `texts.csv` / `news_tone.csv`.
- Dropped `zstandard` dependency and the `live` (praw) extra; declared `numpy` explicitly.
- Licensed under Apache-2.0.
- Committed `uv.lock`; bumped urllib3 2.7.0 -> 2.8.0 (pip-audit: PYSEC-2026-4175/4176/4177).
- Added GitHub Actions CI: ruff check, ruff format check, pytest, pip-audit.
- Applied `ruff format` across the codebase.
- Added `CLAUDE.md` agent guide.
