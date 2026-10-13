# Changelog

## 2026-10-13

- First-name-collision guard no longer fires when the following word is ALL CAPS (2+ letters):
  `NICO WTF`, `THE SEIYA SPECIAL` are credited. An all-caps word still blocks when it is a
  tracked surname (`ANTHONY VOLPE` stays uncredited to Roman Anthony); an untracked all-caps
  surname (`ANTHONY HISCHIER`) is credited, accepted tradeoff. `name_pattern` takes an optional
  `surnames` set for this. Title-case headlines (`Seiya Mash Against Skenes`) remain a limitation.

## 2026-10-12

- Name matching is accent-insensitive in both directions (`Alcantara` = `Alcántara`);
  folding is length-preserving so match offsets stay valid.
- First-name-collision guard: a bare surname that is also a common first name
  (`Anthony` for Roman Anthony) is not credited when directly followed by a capitalized
  word outside the player's own name (`Anthony Volpe`). Known limitation: a sentence
  start after the name (`Anthony Then homered`) is dropped too.
- New `fansent.names.name_pattern(...)` (accent folding + optional guard) for alias layers;
  `build_matchers` now returns pattern-like objects with `.search()` instead of `re.Pattern`.
- New `fansent.first_names.COMMON_FIRST_NAMES`; roster first names are added at build time.
- Multiple-comparison correction now runs within separate families: `metrics`
  (OPS/AVG/ERA/WHIP/FLD_PCT/ERRORS), `COMPOSITE`, `OAA`. `CS_PCT` is exploratory (raw
  `perm_p`, NA `q_bh`/`p_holm`). `analysis_fanbases.csv` gains `family` and `exploratory`
  columns; the printed table is grouped by family.
- Default composite weights: hitters OPS 0.375, AVG 0.375, FLD_PCT 0.125, -ERRORS 0.125;
  pitchers ERA 0.5, WHIP 0.5. `reply_inherited` weight stays 1.0 (`--reply-weight`).

## 2026-10-02

- `fansent correlate` now fits the hierarchical model per metric (OPS, AVG, ERA, WHIP,
  FLD_PCT, ERRORS, CS_PCT, COMPOSITE) with a within-unit permutation null and Benjamini-Hochberg
  / Holm corrections (`fansent.panel`, `composite`, `inference`, `multitest`, `correlate`).
  Writes `analysis_fanbases.csv` (new schema), `analysis_units.csv`, `analysis_oaa.csv`.
- Monthly OAA check: sentiment vs next-month OAA through the same model.
- Reads optional collector sidecars `reddit_attribution.csv` / `news_headlines.csv`.
- Legacy partial-correlation path kept as `fansent correlate --method partial`; its fanbase
  output moved to `analysis_fanbases_partial.csv`.

## 2026-10-01

- Added hierarchical Bayesian sentiment model (`fansent.hierarchical`), hand-built in
  numpy/scipy and fit by vectorized multi-chain Gibbs sampling with an ASIS step; own
  split-R-hat / bulk-ESS in `fansent.diagnostics`; `fansent.model_data` adapts a panel
  DataFrame to the model's arrays. Wired into `fansent correlate` on 2026-10-02.
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
