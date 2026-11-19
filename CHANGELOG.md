# Changelog

## 2026-11-18

- Fix: relevance features `own_team_comment` / `own_team_parent` were always 0 since a29c147
  (2026-11-16), when players.csv `team` became an abbreviation ("CHC") while the feature compares
  against full names ("Chicago Cubs"). `load_players` now resolves the team through the sibling
  `teams.csv` (`team_name`), and the relevance prep canonicalizes it, so abbreviation and
  full-name forms both work. Relevance models trained or scored between a29c147 and this fix saw
  these features at 0; retrain them (`clf-train --task relevance`).
- Relevance own-team features use the player's team in the Mon-Sun week of the text's date
  (`player_teams.csv`, next to players.csv), not his current players.csv team, so traded players
  are right before and after the deal. In the week he moved he counts for both the new and the
  previous week's team: `own_team_comment`, `own_team_parent` and `sub_own` (subreddit of any
  team in the set, via `team_subreddits.csv`) accept either, and the foreign-team features
  exclude both. Falls back to the players.csv team (and its `subreddit`) when the date is
  missing or the player-week has no row. The weekly map is an explicit `weekly_teams` argument
  (`load_weekly_roster(players_path)`) on `RelevanceClassifier` / `load` / `relevance_filter`;
  `None` keeps the current-team behaviour. `Example` gained an optional `date`; labeled CSVs'
  `date` column and `score` texts' date feed it. Models trained before this change still load
  and score, but were fit on current-team features: retrain for consistency.

## 2026-11-17

- `fansent confirm`: runs the pre-registered league-wide fielding test with the specification
  hard-coded (FLD_Z only; teams 111/112 and news excluded; pooled fanbases; 4 x 6000 / 3000
  chains; B = 1000; one-sided p), prints CONFIRMED / NOT CONFIRMED with the pooled `mu`, 95% CrI
  and p, enforces max R-hat < 1.05 (otherwise INVALID), and writes `confirm_result.json`. See
  README "Confirmatory test".
- `correlate`: new metric `FLD_Z` (chances-weighted mean of pooled z(FLD_PCT) and z(-ERRORS)
  with week-t and season-to-date controls); `--exclude-teams`; `--pool-fanbases`;
  `--alternative {two-sided,less,greater}`; `analysis_fanbases.csv` gains `perm_p_less` and
  `perm_p_greater` after `perm_p`.
- Pre-registered the confirmatory league-wide fielding test (`docs/preregistration.md`), written before any non-Cubs/Red Sox data was analyzed.
- `fansent correlate`: `--chains/--iter/--burn` for the main fit (defaults raised to 4 x 6000,
  burn 3000, `InferenceConfig` too; max R-hat 1.015 on real data vs 1.084 at 2000 iterations),
  `--perm-iter/--perm-burn` for the permutation fits (defaults unchanged).
- `--teams TEAM_ID|ABBR ...` restricts to rows where the player was on one of those teams in the
  week of the text date; `--pool-stats {league,teams}` chooses the COMPOSITE z-score pool
  (default league); `--fanbases` / `--exclude-fanbases` include-only / drop fanbases (defaults
  neutral: pass `--exclude-fanbases news` for Reddit only). See README Method.

- `fansent power`: Monte-Carlo power curves and minimum detectable effect (MDE, 80% power) of
  the fanbase test, so a null can be stated as "no effect larger than X". Panels are simulated
  from the hierarchical model's own structure at the real run's sizes (read from
  `analysis_fanbases.csv` COMPOSITE rows), refit with the same sampler, and detection is the
  95% CrI excluding 0. Scenarios: team sub, r/baseball, 30-fanbase league (per-fanbase), and
  the pooled league-wide effect. `--perm-b` re-checks a few replicates with the permutation
  test. Writes `power_curves.csv` / `power_mde.csv`. Results are in the README "Power" section.

- OAA check fixed for the real Savant export, which has monthly independent OAA and no attempts
  column (Mar+Apr arrive as one bucket labeled 2026-04). The OAA family is now unweighted and the
  10-attempt floor (`min_oaa_attempts`) is removed. A unit needs >= 3 consecutive-month pairs and
  months need >= 10 mentions (`min_month_mentions`); cumulative season OAA is the running sum of
  monthly values. `fansent oaa` still reads an `attempts` column if present.

## 2026-11-16

- League-wide weekly rosters. `fansent rosters --season 2026` now pulls the dated 40-man roster
  of all 30 teams for every Mon-Sun week of the season (snapshot taken on the Thursday of the
  week, so a mid-week trade lands about evenly on either side; responses cached under
  `<data-dir>/raw/mlb/rosters/`). New outputs: `player_teams.csv` (`player_id, team_id,
  week_start`, one row per player-week) and `teams.csv`. `players.csv` is one row per player
  with the team at the last week seen and gains `first_name` and `suffix`; `last_name` now comes
  from the API's `lastName` (suffix excluded: "Acuña", suffix "Jr."), fixing `Jr.`/`III` surnames
  that used to break name matching. `team` is now the abbreviation. `--roster-type` is removed
  (always 40-man by date); `--teams` restricts, `--first-week/--last-week` for other seasons.
- `team_subreddits.csv` schema is now `fanbase, subreddit, team_id, status`. Several subreddits
  may share one fanbase (Mets = NewYorkMets + Mets; Athletics = OaklandAthletics +
  SacramentoAthletics, low volume), mapped case-insensitively in the panel. All 30 verified;
  corrected names: miamimarlins (not letsgofish), TexasRangers (not r/Rangers).
- Own-team filter is by week: a team-subreddit mention counts only if the player was on that
  fanbase's team in the Mon-Sun week of the text date (`player_teams.csv`). Absent file: falls
  back to `players.csv` `team_id` with a warning. Dropped counts per fanbase are printed.
- `fansent stats` runs for the league list; game logs are cached under
  `<data-dir>/raw/mlb/gamelogs/<season>/` (`--refresh` to refetch) and the `team` column is the
  team on that week's roster when known. `load_team_subreddits()` now returns subreddit ->
  fanbase; `load_fanbase_teams()` gives fanbase -> team ids.
- `fansent score --scorer clf` with an `--emb` model now loads
  `<data-dir>/embeddings/crawl-300d-2M-subword.bin` for tokens not cached in the model file.
  Before, those tokens silently got no embedding at scoring time (unlike `clf-eval`). Missing
  `.bin`: stderr warning, cache-only vectors.
- Relevance filter: `clf-train` / `clf-eval --task relevance` train a second classifier that
  predicts whether a matched comment is about the matched player (`na`: surname collision with
  a non-roster player, ordinary word, pun). `fansent score` loads
  `<data-dir>/models/relevance.npz` if present (`--relevance-model`, `--no-relevance`), drops
  rows predicted `na` before scoring, prints kept/dropped per subreddit, and writes the audit
  sidecar `relevance.csv` (`text_id, player_id, p_relevant, kept`). `sentiment.csv` unchanged.
  `Example` gains `subreddit`; `load_labeled(keep_na=True)`.
- `fansent score --scorer clf` with a `--parent` model now scores `reply_inherited` rows
  (per `reddit_attribution.csv`) with their parent comment, via the new optional sidecar
  `parent_texts.csv` (`parent_id, text`), truncated to 300 characters. Explicit rows stay
  non-replies. If the sidecar is missing or lacks parents, those rows fall back to non-reply
  scoring with a stderr warning giving the count. `texts.csv` and `sentiment.csv` unchanged.

## 2026-11-15

- Classifier: three optional, toggleable feature blocks (`clf-eval` / `clf-train` flags
  `--emb`, `--lex`, `--parent`), stacked with the TF-IDF blocks via `scipy.sparse.hstack` with
  per-block weights that join the CV grid (`--emb-weights`, `--lex-weights`, `--parent-weights`).
  `--emb`: mean fastText vectors of the masked comment and of the +-4 target window
  (placeholders excluded, unit-normalised). `--lex`: baseball-lexicon VADER compound of the
  comment and of the target window plus log counts of positive/negative lexicon hits.
  `--parent`: TF-IDF of the parent comment (masked for the same target) plus an `is_reply` flag.
  Labeled CSVs may now carry `parent_text`. Saved models include the token vectors they used,
  so scoring does not need the fastText file for tokens seen in training. Old `clf.npz` files load.
- New `fansent clf-fetch-embeddings`: downloads (resumable) and extracts the official fastText
  `crawl-300d-2M-subword` model into `<data-dir>/embeddings/` (5.8 GB download, 7.2 GB `.bin`).
  The `.bin` is read by our own reader (`classifier/embeddings.py`; subword hashing included),
  so there is no `fasttext` package dependency.

## 2026-10-22

- Results site: snapshots carry a required `status` (`preliminary` | `final`, validated;
  `fansent snapshot --status`, default `preliminary`). Preliminary snapshots render a
  "Preliminary results" banner at the top of the page (run date, baseline sentiment scorer,
  target-aware classifier not yet trained, numbers will change) and a "(preliminary)" title
  suffix; `final` drops both. `2026-10-13-teams-clean` is marked preliminary.
- Results site: `fansent snapshot RUN_DIR [--run-id ID] [--perm-b 1000]` writes an
  aggregate-only, schema-validated `results/published/<run-id>.json` from a run's
  `analysis_fanbases.csv`; `fansent site [--snapshot PATH] [--out site]` renders a static,
  JavaScript-free page (headline tiles, forest plot with 95% credible intervals, permutation
  p vs corrected q, data table, method, caveats) with light/dark themes and no frame
  restrictions. First snapshot: `2026-10-13-teams-clean`. New `.github/workflows/pages.yml`
  deploys it to GitHub Pages on push to `main` (needs Pages Source = GitHub Actions).
  `site/` is gitignored build output.

## 2026-10-13

- New `fansent label --file <csv> [--port 8765] [--open]`: a local labeling page (stdlib
  `http.server`, bound to 127.0.0.1, no external assets) for sentiment-eval CSVs. Shows the
  comment with the player name highlighted, the parent comment for replies, and the draft
  label preselected; keys `1/2/3/0` = pos/neu/neg/na, `s`/`u` toggle sarcasm/uncertain,
  Enter/→ accept & next, ← back, "only unreviewed" filter. Each decision is written
  immediately (`final_label`, `sarcasm`, `uncertain`, `reviewed=1`, `reviewed_at`) via temp
  file + atomic rename. Requires `id`, `player`, `text` columns; missing output columns are added.
- Target-aware sentiment classifier (`fansent.classifier`, numpy + scipy.sparse, no ML
  libraries): target/other player masking (`__TGT__`/`__OTH__`), word + char + target-window
  TF-IDF, L2 multinomial logistic regression (own loss/gradient, L-BFGS driver), class weights.
  New `fansent clf-eval` (VADER vs VADER+baseball lexicon vs classifier: accuracy, macro-F1,
  per-class P/R, confusion, Spearman, ECE, grouped k-fold CV tuning) and `fansent clf-train`.
  `fansent score --scorer vader|lexicon|clf [--model PATH]`; `sentiment.csv` format unchanged
  (default stays `vader`). Baseball lexicon overlay is data: `classifier/baseball_lexicon.json`.

- Own-team filter: a team subreddit fanbase counts only its own team's players (roster
  `team_id` from `players.csv`); `r/baseball`, `news` and unmapped subreddits stay league-wide.
  Data-driven map in `fansent/team_subreddits.csv` (30 teams, only CHICubs/redsox verified),
  `PanelConfig.fanbase_teams`, `panel.filter_own_team`, dropped-row counts printed by
  `correlate`. `fansent correlate --all-teams` restores the old behaviour. This changes
  results relative to earlier runs.
- Global `--data-dir DIR` option (before the subcommand) with `FANSENT_DATA_DIR` fallback, then
  `./data` as before. Every subcommand honors it. Breaking for Python importers: `util.DATA_DIR`
  and the `*_CSV` path constants are replaced by `DEFAULT_DATA_DIR`, `resolve_data_dir()` and
  `*_FILE` filenames; `run()` functions take a trailing `data_dir` argument.
- Surnames that are ordinary English words (`fansent.common_words`, ~600 curated entries) match
  only when Capitalized: `a little rough` / `early innings` / `what a story` no longer credit
  Luke Little / Connelly Early / Trevor Story, while `Little blew the save` does. Full-name
  matches stay case-insensitive. ALL-CAPS surname counts unless the whole text has no
  lowercase letters. Known limitation: a sentence-initial `Early innings` still matches.
  `name_pattern` takes `capitalized=True` for alias layers.
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
