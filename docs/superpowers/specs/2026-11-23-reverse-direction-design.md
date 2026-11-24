# Reverse-direction study: how fans react to performance

Status: DRAFT design, 2026-11-23 (revised the same day after owner answers to Q1-Q5).
Not a pre-registration. Owner decisions are recorded at the end.

## Why

The 2026 study asked whether fan sentiment *predicts* next-week performance. The
pre-registered test did not confirm it (`docs/results-2026-11.md`). Along the way we saw the
other direction clearly: sentiment *reacts* to performance (that reaction is why the shuffle
null was biased). This study measures that reaction directly and asks which fans react
*well*: in proportion to what a game says about a player's true talent, and more to skill
than to luck.

## Integrity protocol (read first)

The 2026 season has been looked at many times. It is **exploratory only** here. Seasons
2023, 2024 and 2025 are backfilled and are **confirmatory**.

1. **Analysis list first.** Every 2026 analysis is listed in "Exploratory analysis list"
   below, with its family, before it is run. Anything run that is not on the list is marked
   post hoc in the results note. Multiple comparisons: Benjamini-Hochberg within each family
   (`fansent.multitest.benjamini_hochberg`), Holm also reported.
2. **Backfilled seasons stay closed.** Until the confirmatory pre-registration is frozen,
   nobody opens backfilled-season sentiment. Allowed: collection QA counts (rows per
   subreddit-day, roster sizes, matched-mention counts, relevance-kept counts, PA counts),
   because collection cannot be checked without them. Not allowed: reading the sentiment
   score column, joining sentiment to performance, plotting or summarizing sentiment, or
   tuning anything (classifier, relevance threshold, aliases, thresholds) on backfilled text.
   One disclosed exception, the alias sample (item 3a). The analysis code enforces this: `fansent.reaction` refuses any season other than 2026
   unless it is given a pre-registration file whose status line reads `FROZEN`.
3. **Frozen models.** The sentiment classifier and relevance model used on 2026
   (`models/clf_big.npz`, `models/relevance_league_weekly2.npz`) score the backfilled
   seasons unchanged.

3a. **Disclosed deviation: alias-discovery sample (owner decision 2026-11-23, Q3).** To
   recover nicknames for players not on the curated 2026 alias list, a small uniform random
   sample of raw backfilled comment bodies may be read for name/nickname discovery only.
   Containment: (i) the sample is drawn by the private collector's `alias-sample` command
   with a fixed seed, at most 2,000 comments per season; (ii) the sample file holds the
   subreddit and the body only: no Reddit score, author, id or timestamp, and no sentiment
   score exists for these seasons yet (scoring happens after the review), and no stats are
   joined or shown; (iii) the review happens before the pre-registration is frozen and
   before any backfilled season is scored; (iv) every draw is appended to
   `data/seasons/alias_review_log.jsonl` (UTC time, season, seed, requested and returned
   counts, number of source files and comments in the frame) and the pre-registration
   copies that log; (v) the only output is new rows in `aliases.csv` with
   `source=backfill_sample_<season>`. This breaks strict closed-season purity: whoever
   reads the sample sees how fans in those seasons talk, though not how sentiment lines up
   with performance. It is reported as a deviation in the pre-registration and the results.
4. **Pre-registration.** After the 2026 exploration, `docs/preregistration-reverse.md` is
   written and frozen: hypotheses, exact estimands and models, data filters and thresholds,
   families, alpha, decision rules, which claims are publishable under which outcome, the
   code commit SHA, and a manifest (row counts + sha256) of each backfilled data dir. It is
   committed with `Status: FROZEN` and a UTC timestamp before the first confirmatory run.
   Pushing that commit gives it a public timestamp; that is the owner's call.
5. **Confirmatory run: every season replicates on its own (owner decision 2026-11-23,
   Q1).** A confirmatory claim holds only if it passes its pre-registered test separately
   in 2023, in 2024 and in 2025 (BH within its family inside each season, same alpha). There
   is no pooled test. A pooled three-season estimate may be shown as a description, never
   as the test. Results are reported whichever way they come out. Power under this rule is
   in "Power under per-season replication" below; for small effects it is much lower than a
   pooled test, and that is accepted.

## Data

### What exists (2026)

30 team fanbases + r/baseball, ~3.56M scored, relevance-kept mentions (`data/league_v2`),
weekly rosters (`player_teams.csv`), per-game MLB Stats API logs for hitting, pitching and
fielding (`raw/mlb/gamelogs/2026/<id>_<group>.json`), monthly Savant OAA. About 157k
player-days with own-team mentions.

### New data

- **Statcast expected stats** (Savant search, one request per game date, private
  collector `xstats-download`): one row per plate appearance with `woba_value`,
  `woba_denom`, `estimated_woba_using_speedangle` (xwOBA on contact), xBA, xSLG, launch
  speed/angle, batter and pitcher ids, `game_pk`. Gives per-game xwOBA for hitters and
  xwOBA-against for pitchers. Pulled for 2026 too.
- **Backfill seasons 2023-2025**: weekly 40-man rosters, game logs, OAA, Statcast PA rows,
  Arctic Shift comments + posts for the same 33 subreddits (30 teams, both Mets subs, both
  Athletics subs, r/baseball). One data dir per season, `data/seasons/<YYYY>/`.
- Not collected (owner decision): IL transactions.

### Coverage checks before use (QA counts only)

- Reddit API protest, June 2023: Arctic Shift daily counts for r/CHICubs are 0 for
  2023-06-13..06-17 (checked 2026-11-23 via the aggregate endpoint). Dark days are a
  property of the forum, not a collection gap. The pre-registration will list the excluded
  dark days per subreddit, fixed from counts alone.
- Subreddit names over time: r/OaklandAthletics and r/SacramentoAthletics both map to the
  Athletics; r/Mets and r/NewYorkMets both to the Mets (existing `team_subreddits.csv`).
- Season windows (MLB `seasons` endpoint): 2023-03-30..10-02, 2024-03-20..09-30 (Seoul
  Series), 2025-03-18..09-28 (Tokyo Series), 2026-03-25..09-27.

## Building blocks (all built here, no stats libraries beyond numpy/scipy/pandas)

### Fan-day and local-time alignment

Comments carry `created_utc`. A comment belongs to the **fan-day** of the fanbase's home
time zone with a 05:00 local cutoff: a comment at 01:30 Pacific after a 19:10 game is about
that game, not the next day's. Team time zones are a fixed table
(`src/fansent/team_timezones.csv`, IANA names, so DST is handled by `zoneinfo`). r/baseball
comments use the mentioned player's team zone that week. Game dates from the Stats API are
the local official date, so a game and its post-game discussion share a fan-day.
Fan-weeks are Monday-Sunday over fan-days. Doubleheaders: the day's games are summed.

Comments before first pitch on a game day land in the same fan-day as the game. They
cannot react to it, so they attenuate the lag-0 response rather than bias it. Schedule start
times (`fansent schedule`, one Stats API request per season, final games only) let the
robustness run R7 drop comments made before the player's team's first pitch that fan-day
(`fansent react --drop-pregame`).

### Game value

One scale for hitters and pitchers: **runs above average in the game**, from the game log.

- Hitters: game wOBA from linear weights (uBB, HBP, 1B, 2B, 3B, HR), `wRAA = (wOBA -
  lg_wOBA) / wOBA_scale x PA`, with the season's league wOBA and scale computed from the
  season's own game logs (FanGraphs-style weights held fixed across seasons; documented).
- Pitchers: `runs_saved = (lg_RA9 - 9 x ER / IP) x IP / 9`, plus a FIP-based version
  `(lg_FIP - FIP_game) x IP / 9` for S2.
- Expected versions from Statcast: `xwRAA` from per-PA xwOBA (contact: estimated wOBA;
  K/BB/HBP: actual wOBA value), hitters and pitchers (against).

Exposure `n` = PA (hitters) or batters faced (pitchers).

### Empirical-Bayes talent and surprise

Per group (hitters, pitchers), player per-exposure talent theta_i has a normal prior
N(m, tau^2); game value per exposure has noise variance sigma^2 / n. `m`, `tau^2`,
`sigma^2` are estimated by method of moments from the season's game logs (between-player
variance of season rates minus the expected sampling variance). With `k = sigma^2 / tau^2`,
the posterior mean after N exposures with total value V is `theta_hat = (V + k m) / (N + k)`.

For game g of player i with pre-game exposure N and value v on n exposures:

- surprise `s = v - n theta_hat_pre` (runs beyond what the fan should have expected),
- talent update `u = theta_hat_post - theta_hat_pre = s / (N + n + k)` (runs per exposure).

A rational fan whose sentiment tracks expected future value responds to `u`; the part of
`s` not carried by `u` is noise about the player.

### Inference

- Within-unit (player x fanbase) least squares with unit fixed effects absorbed by
  demeaning, precision weights = mention count, cluster-robust (CR1) standard errors by
  player and a player-block bootstrap (B = 999) for ratio statistics.
- For fanbase-level slopes: the existing Gibbs hierarchical model
  (`fansent.hierarchical.fit`: unit random slopes around fanbase means), with the
  regressor of interest in the `sent` slot and the outcome being sentiment, and the
  existing Freedman-Lane residual permutation (`fansent.inference`) for nulls.
- Cross-fanbase partial pooling (S3): a second-stage normal-normal model,
  `est_k ~ N(theta_k, se_k^2)`, `theta_k ~ N(m, omega^2)`, half-Cauchy prior on omega,
  Gibbs-sampled (same parameter-expansion trick as `hierarchical.py`).

## Studies, questions, estimands

Notation: `y_{j,d}` = mean sentiment of unit j (player i in fanbase f) on fan-day d, only
days with >= 1 relevance-kept mention; weights = mention count.

### S1: reaction and overreaction

Q: How much, and how fast, does sentiment move with results, and do fans react to the
talent signal or to the raw result?

- **S1a daily distributed lag**: `y_{j,d} = a_j + sum_{l=0..7} b_l v_{i,d-l} + sum_l c_l
  G_{i,d-l} + dow_d + e`, where `G` marks a game played (no-game days carry v = 0).
  Estimands: `b_0`, cumulative `B_7 = sum b_l`, and the share of B_7 reached by lag 1.
  Separate fits for hitters and pitchers; v standardized within group.
- **S1b weekly distributed lag**: fan-week mean sentiment on weekly value at lags 0..3,
  same structure. Estimand `B_3`.
- **S1c rational-fan overreaction index (primary)**: the S1a model with the game value
  replaced by its two parts, the talent update `u` and the raw surprise `s`, at lags 0..7:
  `y_{j,d} = a_j + sum_l (alpha_l u_{i,d-l} + beta_l s_{i,d-l} + gamma_l W_{i,d-l}) +
  played/dow + e`, where `W` = the player's team won that game (owner decision
  2026-11-23, Q4: the primary OI is the reaction to the player's own surprise net of the
  team result). With
  `B_u = sum alpha_l`, `B_s = sum beta_l`, a rational evaluative fan has `B_s = 0`.
  **OI = B_s / (B_s + B_u / (N_med + k))**, the share of the response to a one-run surprise
  at median pre-game exposure that the talent update does not justify: 0 = rational, 1 =
  pure recency. Identification comes from how N (and so the weight of a game) varies over
  the season and between players. The test of OI = 0 is the CR1 test of B_s = 0; the OI
  interval is a player-block bootstrap. (An earlier draft used a pre/post change score;
  the fixed-effects form was chosen so S1a and S1c share one model.)
  Secondary (own BH family F1s): the same OI without the team-won terms. The gap between
  the two is the part of apparent overreaction that is enjoyment of the team winning.
  Caveat stated with every result: some hedonic response to the player's own good game
  remains after the win control, and beta > 0 then is not irrational. S2 helps separate
  the two.

### S2: skill vs luck

Q: Do fans respond to the parts of performance that persist (skill) or to the parts that
don't (luck)?

Components per fan-week, per player (raw weekly rates; weeks under 10 PA / 12 batters
faced are dropped and the unit fixed effect absorbs the player's level. An earlier draft
shrank week rates by EB; dropped as unnecessary with the FE and the exposure floor):

- Hitters, skill: K%, BB%, ISO (power), xwOBA. Luck: BABIP, wOBA - xwOBA.
- Pitchers, skill: K%, BB%, HR/9 is mixed so FIP is the skill summary, xwOBA-against.
  Luck: BABIP-against, LOB%, ERA - FIP.

Two slope vectors on the standardized components (all entered jointly):

- **fan slopes b**: fan-week sentiment on this week's components (unit FE),
- **persistence benchmark w**: rest-of-season value per exposure (from the next week to
  season end) on this week's components (player FE removed by using the season-to-date
  rate as a control). w is what the components are actually worth as talent signals.

Estimand: **luck overweight `dL = L(b) - L(w)`** with `L(x) = sum_luck |x_c| / sum_all
|x_c|`. Rational = 0; positive = fans weigh luck more than its predictive value warrants.
Each component slope is also reported (sub-family). Persistence check reported alongside:
split-half (odd/even game) reliability of each component, which should rank skill above
luck; if it doesn't in our data the component labels are revisited before the
pre-registration, not after.

### S3: which fanbase "knows ball"

Q: Do fanbases differ in discrimination (S2) and in overreaction (S1c), and if so which are
best?

- Per fanbase k: `OI_k` and `dL_k` (same models restricted to fanbase k's own-team units),
  with standard errors from the block bootstrap.
- Partial pooling: second-stage normal-normal model per index, giving shrunk `theta_k`,
  95% credible intervals and the between-fanbase SD `omega`.
- **Heterogeneity test** per index: Cochran's Q of the per-fanbase estimates, null by
  permuting fanbase labels across player-units within player group (hitter/pitcher) and
  team-season strata where possible, B = 999. One family of two tests (OI, dL), BH.
- **Knows-ball score: skill vs luck is primary, overreaction secondary (owner decision
  2026-11-23, Q2).** `C_k = -(2 z(dL_k) + z(OI_k)) / 3` per posterior draw (z over
  fanbases within the draw). The 2:1 weight was confirmed by the owner on 2026-11-23. Report posterior median rank, 80% rank interval, and tiers: top tier =
  P(rank in top 10) >= 0.8, bottom tier = P(rank in bottom 10) >= 0.8, everyone else middle.
- **Classifier reliability**: sentiment measurement error enters in two ways. Classical
  noise in the outcome adds variance but no bias; it is in the SEs. A classifier that
  compresses scores differently per fanbase scales every slope by a calibration factor
  `lambda_k`. Both S3 indices are ratios of slopes, so `lambda_k` cancels; that is a design
  choice, not luck. Absolute slopes (S1a/b per fanbase, S4 level gaps) are disattenuated by
  `lambda_k`, estimated as the slope of classifier score on hand label in the labeled eval
  sets, partially pooled across fanbases (few labels per fanbase); split-half within-week
  reliability (the inventory's `sb_rel`) is reported as the precision check and used to
  flag fanbases too noisy to rank (sb_rel < 0.3, set before looking at the indices).

**Publishable fanbase claims.** Rankings, tiers, "fanbase X knows ball / overreacts most",
and any named per-fanbase index value are publishable **only if** the confirmatory
heterogeneity test for that index rejects (BH q < 0.05 in the S3 family) **in each of 2023,
2024 and 2025 separately**. The knows-ball score needs the dL test to replicate in all
three seasons; if the OI test also replicates, the score is shown, otherwise only the dL
ranking is. An OI-only ranking can be shown if only OI replicates, labeled secondary. A
ranking also has to be stable: a tier assignment is published only for fanbases in the
same tier in all three seasons. If neither rejects, the publishable statement is "no
detectable difference between fanbases", with the posterior of omega as an upper bound, and
per-fanbase numbers appear only as a pooled-shrinkage table labeled as indistinguishable.
2026-only fanbase rankings are never published as findings.

### S4: homers vs neutrals

Q: Do a team's own fans react differently from league-wide fans to the same player-week?

Pairs: player-weeks with >= 3 kept mentions both in the player's own team subreddit that
week and in r/baseball. Differencing within the pair removes the player-week's news.

- Level gap (homer bias): mean(team - r/baseball) sentiment, disattenuated per source.
- Reaction gap: difference in S1b weekly slope (interaction with source, pair FE).
- Overreaction gap: OI(team subs pooled) - OI(r/baseball).
- Luck gap: dL(team) - dL(r/baseball).

### S5: team context

Q: Does reaction depend on the team's situation?

Interactions with the S1a lag-0..1 value response: team won the game; team season win%
(standardized, as of that day); player new to the team this season (not on its week-1
roster); rookie (MLB debut this season, people endpoint). One family of four interactions.
Games back from a playoff spot was in the first draft and is dropped: it needs daily
standings by division and is not approximated.

### S6: trades

Q: How do the old and new fanbases treat a traded player?

Events: mid-season team changes in `player_teams.csv` (excluding week-1 changes).
Event-time weeks -4..+8. Outcomes: new-team fanbase sentiment level vs the player's league
baseline (r/baseball), new-fanbase reaction slope in weeks +1..+4 vs later, old-fanbase
mentions after the trade (needs the `--all-teams` view). Estimands: honeymoon gap (weeks
+1..+4 level minus r/baseball, performance-adjusted) and the early-slope ratio. Expected
sample: a few dozen trades per season, so 2026 alone is descriptive. Under per-season
replication S6 is almost certainly underpowered as a confirmatory test (see power notes);
it stays in the pre-registration but its expected outcome is "not confirmed".

## Exploratory analysis list (2026), by family

All on 2026 only. BH within family; Holm reported.

| Family | ID | Analysis |
|---|---|---|
| F1 reaction | A1 | S1a daily DL, hitters: b_0, B_7 |
| F1 | A2 | S1a daily DL, pitchers: b_0, B_7 |
| F1 | A3 | S1b weekly DL, hitters and pitchers: B_3 |
| F1 | A4 | S1c OI, team subs pooled (own-team units) |
| F1 | A5 | S1c OI, r/baseball |
| F1s secondary | A4u, A5u | S1c OI without the team-won control |
| F2 skill/luck | A6 | S2 dL hitters |
| F2 | A7 | S2 dL pitchers |
| F2c components | A8 | each component slope b_c and w_c (descriptive, BH within) |
| F3 fanbases | A9 | heterogeneity test OI_k |
| F3 | A10 | heterogeneity test dL_k |
| F3d | A11 | shrunk per-fanbase indices, combined score, tiers (descriptive only) |
| F4 homers | A12-A15 | S4 level, reaction, OI and luck gaps |
| F5 context | A16-A19 | S5 four interactions (A20 games-back dropped) |
| F6 trades | A21-A22 | S6 honeymoon gap, early-slope ratio |
| Alignment check (added 2026-11-23 after the first S1 smoke run; diagnostic, no test) | R6 | lead placebo: value of the next fan-day's game added to S1a; must be ~0 |
| Robustness (no tests) | R7 | drop comments before the team's first pitch that fan-day |
| Robustness (no tests) | R1-R5 | fan-day cutoff 04:00 / 06:00; explicit mentions only (no reply_inherited); VADER instead of classifier; xwRAA instead of wRAA as v; min mentions 2 / 5 |

## Power under per-season replication

Computed 2026-11-23 from the 2026 exploratory estimates
(`data/league_v2/runs/2026-11-23-power-per-season.csv`, local). Method: each test's 2026 z
taken as the expected z in a backfill season of the same size, two-sided alpha 0.05 per
season (BH ignored, which flatters power slightly), power in one season `p1 = Phi(|z| -
1.96)`, and power to pass in all three seasons `p1^3`. Shown at the 2026 effect size and at
0.75x and 0.5x, because first-look estimates tend to be too large. Backfill seasons may
have more or fewer mentions than 2026; the table is redone from QA mention counts before the
pre-registration (counts only, allowed under item 2).

| Test | 2026 z | all 3 seasons, 1.0x | 0.75x | 0.5x |
|---|---|---|---|---|
| S1a/S1b reaction (A1-A3) | 17-37 | 1.00 | 1.00 | 1.00 |
| S1c OI team subs, hitters (A4) | 5.4 | 1.00 | 0.95 | 0.47 |
| S1c OI team subs, pitchers (A4) | 3.5 | 0.83 | 0.42 | 0.07 |
| S1c OI r/baseball (A5) | 2.6-2.8 | 0.38-0.51 | 0.11-0.17 | ~0.02 |
| S2 dL hitters (A6) | 0.6 | ~0 | ~0 | ~0 |
| S2 dL pitchers (A7) | 3.5 | 0.82 | 0.41 | 0.07 |
| S4 level and reaction gaps (A12, A13) | 4.0-6.3 | 0.94-1.00 | 0.62-0.99 | 0.14-0.69 |
| S4 OI and luck gaps (A14, A15) | 0.3-1.0 | ~0 | ~0 | ~0 |
| S5 interactions (A16-A19) | 0.1-8.9 | 0-1.00 | 0-1.00 | 0-0.98 |
| S6 honeymoon (A21) | 2.7 | 0.45 | 0.14 | 0.02 |
| S6 early slope (A22) | 5.1 | 1.00 | 0.91 | 0.37 |

Reading: the basic reaction results and the team-sub overreaction for hitters will very
likely replicate if real. Anything near z = 3 in 2026 has roughly even or worse odds of
passing three times even if it is true, and the luck-overweight test for hitters, the S4
OI/luck gaps and the weaker S5/S6 effects are effectively untestable under this rule.
That is the price of the stricter rule and it is accepted: a "not confirmed" there means
"not shown", not "absent".

**S3 heterogeneity tests (the fanbase ranking).** From the 2026 S3 run
(`data/league_v2/runs/2026-11-23-fanbases/`, local; 30 fanbases, player-block bootstrap SEs,
parametric approximation of the permutation null via `fanbase_rank.het_power`). Power
depends on the true between-fanbase SD omega relative to the per-fanbase SE:

| omega (x median per-fanbase SE) | dL: 1 season | dL: all 3 | OI: 1 season | OI: all 3 |
|---|---|---|---|---|
| 0.5x | 0.21 | 0.01 | 0.10 | 0.00 |
| 1.0x (dL 0.076, OI 0.50) | 0.89 | 0.70 | 0.41 | 0.07 |
| 1.5x | 1.00 | 1.00 | 0.94 | 0.82 |

In 2026 neither test comes near rejecting (dL p = 0.84, OI p = 0.95). The spread of the raw
per-fanbase dL (SD 0.065) is smaller than a typical per-fanbase SE (0.076), and the pooled
model puts omega for dL at about 0.03 (95% interval 0.004-0.06), roughly 0.4x the SE. If
that is near the truth, the chance that the dL test passes in all three backfill seasons is
about 1%, and the OI test's is lower. Add the rule that a tier must be the same in all three
seasons (in 2026, 29 of 30 fanbases are "middle") and **S3 fanbase rankings are effectively
unpublishable under per-season replication**. The expected publishable S3 outcome is "no
detectable difference between fanbases", with omega's posterior as the upper bound. The
rule stays as decided; this is written down now so that outcome is not a surprise later.

## Code plan

Public repo (`feat/reverse-study`):

- `src/fansent/team_timezones.csv`, `src/fansent/daypanel.py`: game logs -> per-game value
  table; mentions -> fan-day / fan-week sentiment per unit; the joined day panel (all days
  in the season window per unit, with game flags), weekly aggregation.
- `src/fansent/reaction.py`: within-unit WLS + CR1, distributed lags, EB talent/surprise,
  S1c OI with block bootstrap, the season guard; CLI `fansent react`.
- `src/fansent/schedule.py`: `fansent schedule` (first-pitch times).
- `src/fansent/skill_luck.py` (S2), `fanbase_rank.py` (S3 second stage + heterogeneity
  test), then S4-S6 modules.
- Tests on the core logic: time-zone/cutoff assignment (DST edges, doubleheaders, r/baseball
  zone), linear weights, EB moments recovery, within estimator vs a dense dummy regression,
  OI = 0 on a synthetic rational fan and ~1 on a synthetic recency fan, the season guard.

Private collector (`feat/backfill-seasons`): `xstats-download`, `alias-sample` (item 3a);
run scripts per season.

## Decisions recorded (owner, 2026-11-23)

- Scope: S1-S6, all of them.
- Confirmation: backfill past seasons; 2026 exploratory, backfills confirmatory.
- New data: Statcast expected stats via the private collector; no IL transactions.
- S3 = discrimination + least overreaction combined, hierarchical partial pooling,
  heterogeneity test + tiers/credible intervals, classifier-reliability disattenuation.
- Q1: every confirmatory claim must replicate in 2023, 2024 and 2025 individually; no
  pooled test.
- Q2: skill vs luck (dL) is primary in the knows-ball score, overreaction secondary; not
  equal weight. Follow-up 2026-11-23: owner confirmed the 2:1 weighting,
  `C_k = -(2 z(dL_k) + z(OI_k)) / 3`.
- Q3: a small random sample of backfilled raw comment text may be read for alias
  discovery, contained and logged as in item 3a; disclosed deviation.
- Q4: OI controls for the team result; uncontrolled OI kept as secondary (F1s).
- Q5: fetch schedule start times (done for 2023-2026); used in robustness R7.
- Q6 (push the pre-registration for a public timestamp): held; owner decides later.
- Assumptions accepted: backfill seasons 2023-2025, 05:00 cutoff, runs above average as
  the performance scale, tier rule P >= 0.8, reliability floor sb_rel < 0.3.

## Open questions

None at this gate.
