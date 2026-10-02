# Changelog

## 2026-10-01

- Added hierarchical Bayesian sentiment model (`fansent.hierarchical`), hand-built in
  numpy/scipy and fit by vectorized multi-chain Gibbs sampling with an ASIS step; own
  split-R-hat / bulk-ESS in `fansent.diagnostics`; `fansent.model_data` adapts a panel
  DataFrame to the model's arrays. Not yet wired into the CLI; `analyze.py` unchanged.
- Split data collectors (`reddit-dump`, `reddit-live`, `news`) into the private
  `fansent-collect` repo; `fansent` keeps `rosters`, `stats`, `score`, `correlate`
  and consumes the collectors' `texts.csv` / `news_tone.csv`.
- Dropped `zstandard` dependency and the `live` (praw) extra; declared `numpy` explicitly.
- Licensed under Apache-2.0.
- Committed `uv.lock`; bumped urllib3 2.7.0 -> 2.8.0 (pip-audit: PYSEC-2026-4175/4176/4177).
- Added GitHub Actions CI: ruff check, ruff format check, pytest, pip-audit.
- Applied `ruff format` across the codebase.
- Added `CLAUDE.md` agent guide.
