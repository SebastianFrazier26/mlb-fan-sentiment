# fansent — are the fans always right?

Personal research CLI correlating fan sentiment (Reddit + news) with MLB player
performance, and comparing fanbases on how "right" they are about their players.

**Question:** does fan sentiment in week *t* predict a player's performance in
week *t+1*, beyond what week *t* performance (and season-to-date form) already predicts?
Fanbases compared: r/CHICubs, r/redsox, r/baseball (neutral baseline), and `news` (GDELT
headlines) — each judged only on the tracked players it talks about.

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
uv run fansent rosters --season 2026          # Cubs + Red Sox -> data/players.csv
uv run fansent stats --season 2026            # weekly stats -> data/stats.csv
# texts.csv (+ sidecars) come from the collectors (see below)
uv run fansent score                          # VADER -> data/sentiment.csv
uv run fansent correlate --perm-b 1000 --workers 8   # -> data/analysis_{fanbases,units,oaa}.csv
```

## Sentiment scoring

`fansent score --scorer vader|lexicon|clf [--model data/models/clf.npz]` writes
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

Labeled CSV schema (`text, player, player_id, final_label` or `draft_label`, optional
`aliases` as `;`-separated nicknames; `na` rows are excluded from training and evaluation):

```sh
uv run fansent clf-eval --train a.csv b.csv --test test.csv   # CV-tunes, then VADER vs lexicon vs clf
uv run fansent clf-train --train a.csv b.csv --out data/models/clf.npz
uv run fansent score --scorer clf --model data/models/clf.npz
```

`--players` (default `data/players.csv`) supplies the roster used for `__OTH__` masking.

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

A missing sidecar means every mention counts as explicit. `fansent score` turns `texts.csv`
into `sentiment.csv` (VADER compound per text x player).

## Data sources

- **MLB Stats API** (statsapi.mlb.com) — rosters + game logs; free, no key, read-only.
- **Reddit** (via `fansent-collect`) — fan comments; personal, non-commercial research.
- **GDELT DOC 2.0** (via `fansent-collect`) — news per player; free, no key.

## Method (`fansent correlate`)

Per metric (OPS, AVG, ERA, WHIP, FLD_PCT, ERRORS, CS_PCT, and a COMPOSITE index):

1. **Panel.** Fanbase = subreddit for Reddit, `news` for `gdelt_headline`. Per (fanbase,
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
