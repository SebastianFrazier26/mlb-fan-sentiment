# CONTINUITY — fansent

## [PLANS]
- 2026-08-18T00:00Z [USER] Goal: correlate fan sentiment (Reddit + news) with MLB weekly player stats; compare fanbases (r/CHICubs vs r/redsox vs r/baseball vs news) on predictive accuracy ("are the fans always right"). Personal research, never commercial.
- 2026-10-01T00:00Z [USER] Scope: Cubs + Red Sox for the demo; whole sport later.
- 2026-10-01T00:00Z [USER] Data window (supersedes 2026-08-18 Academic Torrents plan): Reddit Mar–Aug 2026 from Arctic Shift full-site monthly dumps, filtered per subreddit one month at a time (download -> filter -> delete); Sept 1–27 2026 via Arctic Shift API. GDELT DOC 2.0 has ~1y lookback -> pull 2026 headlines early.

## [DECISIONS]
- 2026-08-18T00:00Z [USER] Measure FAN sentiment about players (not player-expressed mood).
- 2026-10-01T00:00Z [USER] Model: hierarchical — per-player random sentiment slopes, fanbase-level means — hand-built in numpy/scipy via Gibbs sampling (no PyMC). Validate by synthetic recovery. Replaces the per-player partial-correlation + Fisher-z design (current analyze.py) once built.
- 2026-10-01T00:00Z [USER] Controls: week-t performance + season-to-date performance.
- 2026-10-01T00:00Z [USER] Metrics: OPS, AVG (hitters); ERA, WHIP (pitchers); weekly fielding from Stats API game logs; plus monthly Statcast OAA check. Each metric fit separately with multiple-comparison correction, plus a composite per-player performance index.
- 2026-10-01T00:00Z [USER] Inference: within-player permutation null (shuffle sentiment across weeks).
- 2026-08-18T00:00Z [ASSUMPTION] Still unconfirmed from the original design: weekly Mon–Sun windows; min 10 PA / 3 IP per week; min 3 mentions/week; min 8 weeks per (fanbase, player).
- 2026-10-01T00:00Z [USER] News = GDELT artlist headlines scored with VADER (replaces GDELT timelinetone).
- 2026-08-18T00:00Z [TOOL] Docker daemon not running on host -> dev via uv venv; Dockerfile + Make targets provided for the container path.
- 2026-10-01T00:00Z [USER] Season default = most recent (2026). (CLI defaults changed to 2026 on feat/data-v2.)
- 2026-10-01T00:00Z [USER] Repo public (Apache-2.0); collectors (reddit_dump, reddit_live, news_gdelt) moved to private repo `fansent-collect` (~/Developer/fansent-collect), which depends on `fansent` via uv editable path source and writes texts.csv / news_tone.csv into ../mlb-fan-sentiment/data (override: --data-dir / FANSENT_DATA_DIR).
- 2026-10-12T00:00Z [USER] reply_inherited weight default 1.0 (equal to explicit); keep `--reply-weight`.
- 2026-10-12T00:00Z [USER] Composite weights: hitters OPS 0.375, AVG 0.375, FLD_PCT 0.125, -ERRORS 0.125; pitchers ERA 0.5, WHIP 0.5, no fielding.
- 2026-10-12T00:00Z [USER] Multiple-comparison families, separate: `metrics` (OPS, AVG, ERA, WHIP, FLD_PCT, ERRORS; BH default, Holm also reported), `COMPOSITE` (headline), `OAA` (secondary); `CS_PCT` exploratory (raw perm_p, no correction). Output has `family`, `exploratory` columns.
- 2026-10-12T00:00Z [USER] Weekly/mention/week thresholds stay as-is; still unconfirmed assumptions in README.
- 2026-10-13T00:00Z [USER] Sentiment scorer: own target-aware classifier built from scratch in numpy (masked target/other names, word+char+target-window TF-IDF, L2 multinomial logistic regression, L-BFGS), replacing raw VADER (~39% on draft labels). Baselines kept in the same harness: VADER and VADER + baseball lexicon overlay. Training/eval on real labels deferred until the user adds improvements.
- 2026-10-13T00:00Z [ASSUMPTION] `na` labels are excluded from training and eval (relevance, not sentiment) rather than a 4th class; revisit if na rows are common in real data.
- 2026-10-13T00:00Z [USER] A team subreddit fanbase counts ONLY its own team's players (r/CHICubs -> Cubs 112, r/redsox -> Red Sox 111; roster `team_id` from players.csv, so traded players leave the old team's sub). r/baseball and news stay league-wide. `--all-teams` disables for sensitivity checks.
- 2026-10-13T00:00Z [CODE] Unmapped subreddits are treated as league-wide (not an error) so new/non-team sources work; 28 non-demo entries in team_subreddits.csv are unverified recall.

- 2026-11-16T00:00Z [USER] Expand to all 30 MLB teams. Track player->team BY WEEK from dated 40-man rosters (`player_teams.csv`); own-team filter uses the team in the week of the mention (supersedes the 2026-10-13 players.csv current-roster rule, which stays as the warned fallback). Fix suffix bug (last_name from API `lastName`, `suffix` column). Corrected/multi-subreddit fanbases (Mets, Athletics); team_subreddits.csv schema `fanbase, subreddit, team_id, status`.
- 2026-11-16T00:00Z [CODE] Roster snapshot = Thursday of each Mon-Sun week (mid-week) so mid-week trades split evenly rather than lagging a week. Stats `team` column = team on that week's roster.
- 2026-11-18T00:00Z [USER] Permutation-null fix approved: Freedman-Lane residual permutation is the `correlate` default (`--null {freedman-lane,shuffle}`); two-sided perm_p centered on null median; `confirm` keeps the registered shuffle/uncentered one-sided p and adds FL one-sided + centered p as supplementary post-hoc fields.

## [PROGRESS]
- 2026-08-18T00:00Z [CODE] Scaffolded ~/Developer/mlb-fan-sentiment: fansent CLI (rosters, stats, reddit-dump, reddit-live, news, score, correlate), tests, Dockerfile, Makefile, README.
- 2026-08-18T00:00Z [TOOL] Verified: ruff clean, 9/9 pytest pass; live smokes OK (rosters=108 players 2025 fullSeason, stats gameLog->weekly OPS, GDELT timelinetone real data); synthetic end-to-end recovered planted lagged signal (r_partial=0.99), control pitcher correctly excluded.
- 2026-08-18T00:00Z [TOOL] Full 2025 pulls launched: stats (all 108) and GDELT news tone 2025-03-01..2025-11-15. Docker image NOT built (daemon down) — `make docker-build` untested.
- 2026-10-01T00:00Z [CODE] Split collectors into private fansent-collect; public initial commit has no collector code in history. Branch chore/repo-norms: Apache-2.0 LICENSE, uv.lock committed, numpy explicit, CI (ruff check/format, pytest, pip-audit), CLAUDE.md, CHANGELOG.md, ruff format applied.
- 2026-10-01T00:00Z [CODE] Branch feat/hierarchical: hierarchical.py (Gibbs + ASIS, half-Cauchy tau via parameter expansion), diagnostics.py, model_data.py, tests/test_hierarchical.py. Not wired to CLI.
- 2026-10-01T00:00Z [CODE] feat/data-v2: stats.csv now tidy long (OPS/AVG/ERA/WHIP/fielding + std_value, weight), stats_counts.csv, OAA monthly loader; schema documented in README.
- 2026-10-02T00:00Z [CODE] Branch feat/inference (merged feat/data-v2 + feat/hierarchical): panel.py, composite.py, inference.py, multitest.py, correlate.py; CLI `correlate` runs model + permutation null + BH/Holm; `--method partial` keeps legacy path. Thresholds, composite weights, reply_inherited weight=1 are unconfirmed assumptions (README Method).

- 2026-10-01T00:00Z [TOOL] pip-audit flagged urllib3 2.7.0 (PYSEC-2026-4175/4176/4177) -> lock bumped to 2.8.0.
- 2026-10-01T00:00Z [USER] Known bug to fix: fansent-collect reddit-dump --since/--until parse dates as local time while rows are UTC.
- 2026-08-18T00:00Z [TOOL] GDELT rate-limits ~1 req/5s per IP (429 otherwise) — collector uses 5.5s pause + backoff retry.
- 2026-08-18T00:00Z [TOOL] Academic Torrents subreddit dumps cover 2005-06..2025-12, per-subreddit files (top 40k subs) — solves Reddit historical backfill.
- 2026-08-18T00:00Z [TOOL] Reddit API free tier (100 QPM/OAuth client, non-commercial) confirmed still current as of 2026-08.
- 2026-10-13T00:00Z [CODE] Branch feat/classifier: fansent.classifier (text, features, softmax, pipeline, overlay, scorers, evaluate), `clf-eval`/`clf-train`/`score --scorer`, synthetic-only tests. Not run on real labels.
- 2026-10-22T00:00Z [CODE] Branch feat/demo-site: `fansent snapshot`/`fansent site` (site.py), results/published/2026-10-13-teams-clean.json (aggregate-only), pages.yml (Pages via Actions; owner must set Source = GitHub Actions). Site is iframed as the project's hero portal on the portfolio.

## [OUTCOMES]
- 2026-08-18T00:00Z [CODE] MVP pipeline built and verified (unit + synthetic e2e + live smokes). Remaining: user torrents Reddit .zst dumps (baseball/CHICubs/redsox) and runs reddit-dump -> score -> correlate; optional reddit-live creds; top-100 player list; Docker build unverified.
- 2026-11-15T00:00Z [USER] Classifier gets optional feature blocks (fastText subword vectors, lexicon scores, parent-context TF-IDF), keeping our own logistic regression; selection by training-data CV only, never test_300.
- 2026-11-16T00:00Z [USER] Relevance filter approved: own second classifier (same numpy LR stack) predicts whether a matched comment is about the matched player vs `na`; `score` drops predicted-na rows and writes `relevance.csv`; threshold chosen on training CV only (default precision>=0.8 operating point), never on test_300.
- 2026-11-15T00:00Z [CODE] Branch feat/clf-features: own fastText .bin reader (no `fasttext` dependency: archived upstream, C++ sdist), `clf-fetch-embeddings`, `--emb/--lex/--parent`, block weights in Grid.
- 2026-11-18T00:00Z [USER] Within-unit estimator approved as opt-in (`correlate --estimator mundlak`, default pooled; `confirm` stays pooled). Reason: ~26% of sentiment variance is between units and unit-mean sentiment correlates +0.27 with unit-mean next-week outcome, so pooled mu mixes between and within; any confirmatory use needs a new pre-registration.
- 2026-11-18T00:00Z [CODE] Branch fix/permutation-null: shuffle null was centered ~+0.03 (sentiment reacts to perf; synthetic react panel rejected 10/10). Implemented FL null in inference.py, centered perm_p, confirm supplementary fields, tests, README/CHANGELOG. Not pushed.
- 2026-11-18T00:00Z [CODE] Branch feat/within-estimator (stacked on fix/permutation-null, unpushed): `--estimator mundlak` (model_data.py splits sent/perf/std into unit mean + deviation, unweighted means; unit means are `extras` into `hierarchical.fit(extra=...)`; output adds `beta_between`; FL null unchanged). tests/test_within.py. Exploratory runs in data/league/runs/2026-11-18-league-exploratory/*_mundlak and data/runs/2026-11-18-demo-mundlak.
