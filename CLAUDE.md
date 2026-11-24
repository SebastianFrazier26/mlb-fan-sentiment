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
- `src/fansent/classifier/` — from-scratch target-aware sentiment classifier, lexicon overlay, eval harness (`clf-eval`, `clf-train`, `score --scorer`)
- `src/fansent/analyze.py` — legacy lagged partial correlations (`correlate --method partial`)
- `src/fansent/panel.py`, `composite.py`, `inference.py`, `multitest.py`, `correlate.py` — panel builder, composite index, fit + permutation null (Freedman-Lane default, `--null shuffle` legacy; centered two-sided p; `--estimator mundlak` = opt-in within-player estimand, adds `beta_between`; default `pooled` is the registered one and `confirm` never uses mundlak without a new pre-registration), BH/Holm, `correlate` orchestration
- `src/fansent/confirm.py` — `fansent confirm`: the pre-registered confirmatory test (`docs/preregistration.md`), spec hard-coded (registered shuffle null; Freedman-Lane only as supplementary fields); never run on league data for exploration
- `src/fansent/hierarchical.py` — Gibbs-sampled hierarchical model (`fit` -> `FitResult`); `diagnostics.py` (split-R-hat, bulk ESS); `model_data.py` (panel -> arrays)
- `src/fansent/power.py` — `fansent power`: simulated power curves and MDE of the fanbase test
- `src/fansent/daypanel.py`, `reaction.py` — reverse-direction study (spec `docs/superpowers/specs/2026-11-23-reverse-direction-design.md`): fan-day panel with local-time alignment (`team_timezones.csv`), per-game runs-above-average, EB talent update/surprise; `fansent react` = S1 distributed lags + overreaction index. 2026 only unless `--prereg` is a FROZEN file; backfilled seasons (`data/seasons/<YYYY>/`) must not be analysed before that
- `src/fansent/names.py`, `util.py` — name matching, paths, CSV I/O (also imported by `fansent-collect`)
- `src/fansent/label.py` — `fansent label`: localhost-only labeling page for eval CSVs (data stays local, gitignored)
- `src/fansent/site.py` — `fansent snapshot` (run -> aggregate-only `results/published/<run-id>.json`, schema-validated; `--section SCOPE=RUN_DIR` + `--confirm` = schema 2, several runs and the pre-registered result on one page) and `fansent site` (snapshot -> static `site/`, gitignored); only aggregate numbers may ever be published
- `src/fansent/figures.py` — `fansent figures`: standalone static SVGs for the README (`docs/img/`) from a schema-2 snapshot; a test fails if the committed figures drift from the snapshot
- `tests/` — pytest; `.github/workflows/ci.yml` — lint, format, test, pip-audit; `pages.yml` — build site from the newest snapshot, deploy to GitHub Pages
- `data/` — generated CSVs, gitignored; location overridable via `fansent --data-dir DIR` / `$FANSENT_DATA_DIR` (`util.resolve_data_dir`, resolved once in `cli.main` and passed to each `run(..., data_dir)`; modules hold filename constants only, never full paths)

## Conventions

- Collectors (Reddit dumps/API, GDELT) live in the PRIVATE `fansent-collect` repo
  (`../fansent-collect`), which depends on this package via a uv editable path source.
  Never add scrapers/collectors here. Its CSV columns (`texts.csv`, `news_tone.csv`)
  are a contract — keep `names.py` / `util.py` APIs stable or update both repos together.
- `uv.lock` is committed; CI runs `uv sync --locked`.
- Ruff, line length 100. Comments load-bearing only. User-visible changes get a dated
  `CHANGELOG.md` entry.
- No secrets in the repo; remote APIs are read-only GETs.
