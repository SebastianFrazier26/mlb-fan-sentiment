# fansent — are the fans always right?

Personal research CLI correlating fan sentiment (Reddit + news) with MLB player
performance, and comparing fanbases on how "right" they are about their players.

**Question:** does fan sentiment in week *t* predict a player's performance in
week *t+1*, beyond what week *t* performance (and season-to-date form) already predicts?
Fanbases compared: all 30 team subreddits, r/baseball (neutral baseline), and `news` (GDELT
headlines) — each judged only on the tracked players it talks about.

**Results site:** <https://sebastianfrazier26.github.io/mlb-fan-sentiment/> (live once GitHub
Pages is enabled; see [Results site](#results-site)).

Fanbases compared: r/CHICubs, r/redsox, r/baseball (neutral baseline), and
`news` (GDELT media tone) — each judged only on the tracked players it talks about.

## Setup

```sh
make setup          # uv sync
make check          # ruff + pytest
```

Docker alternative: `make docker-build`, then `make docker-run ARGS="rosters"`.

## Pipeline

```sh
uv run fansent rosters --season 2026          # all 30 teams, weekly 40-man -> players.csv, player_teams.csv
uv run fansent stats --season 2026            # weekly stats -> data/stats.csv
# texts.csv (+ sidecars) come from the collectors (see below)
uv run fansent score                          # VADER -> data/sentiment.csv
uv run fansent correlate --perm-b 1000 --workers 8   # -> data/analysis_{fanbases,units,oaa}.csv
```

## Sentiment scoring

`fansent score --scorer vader|lexicon|clf [--model PATH]` writes
`sentiment.csv`; `compound` is the score in [-1, 1] whichever scorer ran, so `correlate`
is unchanged.

- `vader`: raw VADER (default). Misreads baseball slang (nasty, filthy, killing it) and
  sentiment aimed at someone other than the named player.
- `lexicon`: VADER plus a baseball overlay (`src/fansent/classifier/baseball_lexicon.json`,
  edit freely: words, phrases, emoji, VADER-scale valences).
- `clf`: own classifier (`src/fansent/classifier/`). The row's player is masked as
  `__TGT__` and other rostered players as `__OTH__`; features are word 1-2-gram and char
  3-5-gram TF-IDF plus a block of n-grams within +-k tokens of `__TGT__`; model is L2
  multinomial logistic regression with balanced class weights, fit by L-BFGS on our own
  loss/gradient. Score = P(pos) - P(neg).

Optional feature blocks (each off by default, combinable, weights tuned by CV on the
training data only): `--lex` adds baseball-lexicon scores, `--parent` adds the parent
comment's TF-IDF plus an `is_reply` flag (needs a `parent_text` column in the labeled CSV),
and `--emb` adds mean fastText vectors of the comment and the target window. `--emb` needs the
pretrained model once: `uv run fansent clf-fetch-embeddings` (5.8 GB download, 7.2 GB on disk
in `<data-dir>/embeddings/`, gitignored; `--emb-path` points elsewhere). Out-of-vocabulary
words get vectors from their character n-grams, via our own reader of the `.bin` format.
`fansent score` feeds `--parent` models the parent text of `reply_inherited` rows (see
"Where the text data comes from"); explicit rows are non-replies, as in training.

### Relevance filter

A matched surname is often someone else: with a Cubs + Red Sox roster, "Martinez" or
"Turner" in r/baseball usually means a player we do not track (~39% of Sonnet-labeled
r/baseball matches are `na`, ~3% in the team subs). A second from-scratch classifier (same
L2 logistic regression and L-BFGS) predicts whether a matched comment is about the matched
player. Its features keep the name context that sentiment masking hides: match type (full
name, alias, surname only), the words next to the mention, whether the preceding word is
another first name or capitalized word, mention casing, own-team versus other-team terms
(all 30 clubs, `classifier/teams.py`) in the comment and its parent, whether the parent
names the player, and the subreddit (dense flags are duplicated for r/baseball so it learns
its own rules), plus word TF-IDF. Labels: `na` vs pos/neu/neg from the labeled CSVs.

```sh
uv run fansent clf-eval  --task relevance --train a.csv b.csv --test test.csv  # holdout + test report
uv run fansent clf-train --task relevance --train a.csv b.csv                  # -> <data-dir>/models/relevance.npz
uv run fansent score --scorer clf          # uses relevance.npz automatically if present
```

The drop threshold is chosen on out-of-fold CV scores: `--operating-point precision`
(default; lowest threshold with CV precision >= `--min-precision`, 0.8) or `f1`. `fansent score`
drops rows with P(na) >= threshold before scoring, prints kept/dropped counts per subreddit,
and writes `relevance.csv` (`text_id, player_id, p_relevant, kept`) for audits and sensitivity
checks. `sentiment.csv` keeps its schema. `--relevance-model PATH` overrides the model,
`--no-relevance` disables the filter. A stale `relevance.csv` from an earlier run is not removed.

Labeled CSV schema (`text, player, player_id, final_label` or `draft_label`, optional
`aliases` as `;`-separated nicknames; `na` rows are excluded from training and evaluation):

```sh
uv run fansent clf-eval --train a.csv b.csv --test test.csv   # CV-tunes, then VADER vs lexicon vs clf
uv run fansent clf-train --train a.csv b.csv
uv run fansent score --scorer clf
```

`--players` (default `<data-dir>/players.csv`) supplies the roster used for `__OTH__` masking;
`clf-train --out` and `score --model` default to `<data-dir>/models/clf.npz`. Explicit path
arguments are taken as given (relative to the current directory, not `--data-dir`).
All commands read and write one data directory, chosen by `--data-dir DIR` (given before the
subcommand, e.g. `fansent --data-dir ~/fansent-data stats`), else `$FANSENT_DATA_DIR`, else
`./data` relative to the current directory. Note `fansent-collect` defaults differently (an
absolute path next to this checkout), so set `FANSENT_DATA_DIR` in your shell if you run both
tools from arbitrary directories.

## Where the text data comes from

Reddit and news collection lives in a separate, private companion repo
(`fansent-collect`), kept out of this public repo by design. It reads
`data/players.csv` and writes into `data/`:

- `texts.csv` — `text_id, source, subreddit, date, created_utc, player_id, player_name, text`
  (one row per text x mentioned player; `date` is the UTC day; `source` is e.g.
  `reddit_arctic` or `gdelt_headline`; `subreddit` is empty for headlines)
- `reddit_attribution.csv` (optional) — `text_id, player_id, attribution, parent_id`;
  `attribution` is `explicit` or `reply_inherited` (player named in the parent comment)
- `news_headlines.csv` (optional) — `text_id, player_id, url, domain, attribution, seendate`
- `parent_texts.csv` (optional) — `parent_id, text`; the comment each `reply_inherited` row
  replies to (`parent_id` as in `reddit_attribution.csv`). Only `fansent score --scorer clf`
  with a `--parent` model reads it: matching rows get `is_reply=1` and the parent-text block,
  truncated to the first 300 characters (the `parent_text` length of the training samples).
  Without it, those rows are scored as non-replies and a warning gives how many.

A missing sidecar means every mention counts as explicit. `fansent score` turns `texts.csv`
into `sentiment.csv` (VADER compound per text x player).

## Data sources

- **MLB Stats API** (statsapi.mlb.com) — rosters + game logs; free, no key, read-only.
- **Reddit** (via `fansent-collect`) — fan comments; personal, non-commercial research.
- **GDELT DOC 2.0** (via `fansent-collect`) — news per player; free, no key.

## Method (`fansent correlate`)

Per metric (OPS, AVG, ERA, WHIP, FLD_PCT, ERRORS, CS_PCT, and a COMPOSITE index):

1. **Panel.** Fanbase = the subreddit's fanbase for Reddit (`src/fansent/team_subreddits.csv`:
   `fanbase, subreddit, team_id, status`, all 30 teams verified 2026-11-16; Mets = NewYorkMets +
   Mets and Athletics = OaklandAthletics + SacramentoAthletics each pool into one fanbase, the
   A's are low-volume), `news` for `gdelt_headline`. A team fanbase counts only mentions of its
   own team's players (user decision 2026-10-13), judged by week (2026-11-16): the player must
   be on that team's 40-man in the Mon-Sun week of the text date (`player_teams.csv` from
   `fansent rosters`), so traded players leave the old team's fanbase and count for the new one
   from arrival. Without `player_teams.csv` it falls back, with a warning, to the single
   `team_id` in `players.csv`. `r/baseball`, `news`, and any subreddit not in the map are league-wide
   (no filter); rows dropped per fanbase are printed. `--all-teams` (or
   `PanelConfig(fanbase_teams={})`) disables the filter for sensitivity checks. Per (fanbase,
   player, Mon-Sun week): weighted mean VADER compound and mention count. Each row at week *t*
   carries `perf` = metric at *t*, `std` = season-to-date metric through *t*, `sent` at *t*,
   `y_next` = metric at *t+1* (consecutive weeks required) and `weight` = the sample size at
   *t+1*. Raw values are kept with `higher_is_better`; the sign is flipped only when fitting, so
   every effect reads "positive = better performance".
2. **Model.** `fansent.hierarchical`: `y_next = alpha_unit + b1*perf + b2*std + gamma_unit*sent`,
   unit = (fanbase, player), `gamma_unit ~ N(mu_fanbase, tau_fanbase^2)`; Gibbs sampler written
   from scratch. `mu_k` is the fanbase effect, reported in sd of next-week performance per sd of
   sentiment (variables are z-scored globally). `p_pos` = posterior P(mu_k > 0).
3. **Permutation null.** Within each unit sentiment is shuffled across that unit's weeks
   (y_next, perf, std fixed), the model is refit (light, `store_units=False`), and the posterior
   mean of each `mu_k` collected; `perm_p = (1 + #{|null| >= |obs|}) / (1 + B)`, two-sided,
   seeded and independent of worker count. This breaks the sentiment-to-outcome link while
   preserving each unit's marginals; it also breaks the within-unit sentiment-performance link,
   so it is a null for "sentiment carries no information", not for "sentiment merely reacts".
4. **Multiple comparisons**, corrected within separate families (Benjamini-Hochberg `q_bh` and
   Holm `p_holm`, both hand-implemented; `--correction` picks which one the `*` marker in the
   printed table uses, default BH). Families: `metrics` (OPS, AVG, ERA, WHIP, FLD_PCT, ERRORS x
   fanbase); `COMPOSITE` (the headline test, corrected only among its own fanbases); `OAA`
   (secondary). `CS_PCT` is `exploratory`: raw `perm_p` only, `q_bh`/`p_holm` are NA and it is
   excluded from every family. The `family` and `exploratory` columns record this, and the
   printed table is grouped by family.
5. **COMPOSITE.** Per metric, oriented values are z-scored pooled over eligible player-weeks;
   a player-week's composite is the weighted mean of the z-scores available that week
   (hitters: OPS 0.375, AVG 0.375, FLD_PCT 0.125, -ERRORS 0.125; pitchers: ERA 0.5, WHIP 0.5,
   no fielding; configurable in `PanelConfig.composite_weights`). Its season-to-date control is built the same way from
   `std_value`. Role = hitter if the player ever reached 20 PA, else pitcher. A week needs a
   hitting/pitching metric, so fielding alone never forms a composite. The composite has no
   sample-size weight.
6. **OAA check.** Monthly mean sentiment vs next-month Outs Above Average (consecutive months),
   controlling current-month OAA and cumulative season OAA, through the same model and
   permutation, weighted by next-month attempts. Monthly data gives ~5 points per unit, too few
   for per-unit partial correlations, so the hierarchical pooling is used instead.

**Fit settings and scope flags (2026-11-17).**

- `--chains/--iter/--burn` configure the main fit (defaults 4 x 6000, burn 3000; on real data
  this gave max R-hat 1.015 where 2000 iterations gave 1.084). The permutation fits keep their
  own light settings, `--perm-iter/--perm-burn` (defaults 600/300).
- `--teams TEAM_ID|ABBR ...` keeps only sentiment rows where (player, Monday-week of the text
  date) is in `player_teams.csv` with one of these team ids; abbreviations resolve through
  `teams.csv`. It needs `player_teams.csv`. It applies before and independently of the own-team
  filter.
- `--pool-stats {league,teams}` (default `league`) sets who the COMPOSITE z-scores are pooled
  over. `league`: every player in `stats.csv`, so composite values are comparable across runs.
  `teams`: only the players still present after `--teams/--fanbases/--exclude-fanbases`, so
  "good" means good relative to that group, and composites are not comparable with league runs.
  Only the COMPOSITE changes; single metrics are never pooled.
- `--fanbases NAME ...` (include only) and `--exclude-fanbases NAME ...` (drop), matched
  case-insensitively against fanbase names (`news`, `baseball`, a team fanbase such as
  `Yankees`). Defaults are neutral, so news is included unless dropped:
  `fansent correlate --exclude-fanbases news` for Reddit only.

Thresholds below are unconfirmed assumptions rather than tuned values (composite weights, the
reply weight and the correction families are user decisions of 2026-10-12): weekly minimum sample
OPS/AVG 10 (PA/AB), ERA/WHIP 3 IP, FLD_PCT/ERRORS 3 chances, CS_PCT 2 attempts; >= 3 mentions
per player-week; >= 8 weeks per unit; >= 3 units per (metric, fanbase); `reply_inherited`
mentions weigh the same as explicit ones (`--reply-weight`, user decision 2026-10-12); OAA months need >= 10 mentions,
>= 10 attempts, >= 3 months per unit.

Outputs: `analysis_fanbases.csv` (`family, exploratory, metric, fanbase, mu_mean, ci_low, ci_high, p_pos, perm_p,
q_bh, p_holm, n_units, n_obs, rhat_max`), `analysis_units.csv` (per-unit shrunk effects),
`analysis_oaa.csv` / `analysis_oaa_units.csv`. `--method partial` runs the legacy per-pair
partial correlation (OPS/ERA only, `analysis_players.csv`, `analysis_fanbases_partial.csv`) for
comparison; it does not use the model, permutation test or corrections.

## Results site

A static page (forest plot of fanbase effects, permutation-null comparison, method, caveats)
rendered from an aggregate-only snapshot, so it builds without `data/`:

```sh
uv run fansent snapshot data/runs/<run-id>   # analysis_fanbases.csv -> results/published/<run-id>.json
uv run fansent site                          # newest snapshot -> site/index.html (+ data.json)
python -m http.server -d site                # preview
```

The snapshot holds only per-(metric, fanbase) summary numbers from `analysis_fanbases.csv`;
`fansent.site.validate_snapshot` rejects unknown keys, any string outside the metric/family
vocabulary or a short identifier pattern, and non-finite numbers, so free text cannot be
published. Per-player files (`analysis_units.csv`) are never read. Commit the snapshot to
publish a new run; `.github/workflows/pages.yml` rebuilds and deploys on every push to `main`
(and on manual dispatch). Charts are inline SVG generated in Python (no JavaScript), themed by
`prefers-color-scheme`, and the page sets no frame restrictions so it can be embedded in an
iframe.

One-time setup (repo owner): Settings -> Pages -> Build and deployment -> Source: **GitHub
Actions**. Until then the deploy job fails and the URL above returns 404.

## Notes & caveats

- Name matching: full name always; bare last name only when unique among
  tracked players (≥4 chars). Common-word last names can still false-positive.
- Weekly windows are Mon–Sun (regular season only).
- Weekly OPS/ERA is extremely noisy — expect small effects; the pooled fanbase effects
  `mu_k` are the robust readout, not any single unit's `gamma`.

## Data schema (model contract)

`data/stats.csv` — tidy long, one row per player-week-metric; rows exist only for weeks
where the metric's weekly denominator is positive.

| column | meaning |
|---|---|
| `player_id, full_name, team, season` | identity (`player_id` = MLBAM id) |
| `week_start` | Monday (ISO date) of the Mon-Sun week |
| `group` | `hitting`, `pitching` or `fielding` |
| `metric` | `OPS`, `AVG` (hitting); `ERA`, `WHIP` (pitching); `FLD_PCT`, `ERRORS`, `CS_PCT` (fielding) |
| `value` | metric for that week alone |
| `std_value` | season-to-date metric through the END of that week (weeks <= t only), recomputed from summed counts |
| `weight` | weekly sample size: PA (OPS), AB (AVG), IP (ERA, WHIP), chances (FLD_PCT, ERRORS), SB+CS attempts (CS_PCT) |
| `std_weight` | same, season-to-date |
| `higher_is_better` | False for ERA, WHIP, ERRORS |

Fielding aggregates every position a player played that week (the log has one split per
game per position): counts are summed across positions and `games` counts distinct games.
`CS_PCT` = caught stealing / (SB + CS) allowed, so only catchers get rows. `ERRORS` is a
weekly/cumulative count, not a rate.

`data/stats_counts.csv` — wide companion: `player_id, full_name, team, season, week_start,
group`, then raw weekly counts `games, pa, ab, h, bb, hbp, sf, tb, outs, er, chances,
putouts, assists, errors, sb, cs` and the same prefixed `std_`. Columns not used by a
group are 0. For pitchers `h`/`bb` are hits/walks allowed and `outs` = innings x 3.

`data/oaa_monthly.csv` — from `uv run fansent oaa PATH` (reads a Baseball Savant Outs Above
Average CSV export with monthly splits; the download itself lives in the private
collector): `player_id, full_name, month (first of month), oaa, attempts,
fielding_runs_prevented`. Rows for several positions in a month are summed.

## License

Apache-2.0 — see [LICENSE](LICENSE).
