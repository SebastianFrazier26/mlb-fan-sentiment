# fansent — agent guide

Public repo (Apache-2.0). Fan sentiment vs. MLB player performance — "are the fans
always right?" Read `AGENTS.md` (repo rules) and `.agent/CONTINUITY.md` (decisions,
progress; read at session start, update on meaningful deltas).

## Commands

- Install: `uv sync` (or `make setup`)
- Test: `uv run pytest` (or `make test`)
- Lint/format: `uv run ruff check .` and `uv run ruff format --check .` (`make check` = lint + test)
- Run: `uv run fansent {rosters,stats,score,correlate} ...` — see README for the pipeline
- Audit (as CI does): `uv export --locked --format requirements.txt --no-dev --no-emit-project -o requirements.txt && uvx pip-audit@2.10.1 -r requirements.txt --disable-pip --require-hashes`

## Layout

- `src/fansent/cli.py` — argparse entry point `fansent`
- `src/fansent/rosters.py`, `stats.py` — MLB Stats API -> `players.csv`, weekly `stats.csv`
- `src/fansent/sentiment.py` — VADER over `texts.csv` -> `sentiment.csv`
- `src/fansent/analyze.py` — legacy lagged partial correlations (`correlate --method partial`)
- `src/fansent/panel.py`, `composite.py`, `inference.py`, `multitest.py`, `correlate.py` — panel builder, composite index, fit + permutation null, BH/Holm, `correlate` orchestration
- `src/fansent/hierarchical.py` — Gibbs-sampled hierarchical model (`fit` -> `FitResult`); `diagnostics.py` (split-R-hat, bulk ESS); `model_data.py` (panel -> arrays)
- `src/fansent/names.py`, `util.py` — name matching, paths, CSV I/O (also imported by `fansent-collect`)
- `tests/` — pytest; `.github/workflows/ci.yml` — lint, format, test, pip-audit
- `data/` — generated CSVs, gitignored

## Conventions

- Collectors (Reddit dumps/API, GDELT) live in the PRIVATE `fansent-collect` repo
  (`../fansent-collect`), which depends on this package via a uv editable path source.
  Never add scrapers/collectors here. Its CSV columns (`texts.csv`, `news_tone.csv`)
  are a contract — keep `names.py` / `util.py` APIs stable or update both repos together.
- `uv.lock` is committed; CI runs `uv sync --locked`.
- Ruff, line length 100. Comments load-bearing only. User-visible changes get a dated
  `CHANGELOG.md` entry.
- No secrets in the repo; remote APIs are read-only GETs.
