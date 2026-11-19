# Results: league-wide fielding test and follow-ups (2026 season)

Recorded 2026-11-19. Aggregate numbers only. The Reddit text behind them is private.

## Pre-registered confirmatory test: not confirmed

The test in [preregistration.md](preregistration.md) was run once, on 2026-11-18 05:55 UTC, with
`fansent confirm` at commit 51a94d2. Data: 2026 regular season, player-weeks off the Cubs and Red
Sox, in the 28 other team subreddits plus r/baseball, news excluded.

| | |
|---|---|
| Outcome | `FLD_Z` (next-week fielding index) |
| Pooled effect mu | -0.0023 sd per sd of sentiment |
| 95% credible interval | [-0.022, +0.017] |
| One-sided permutation p (H1: mu < 0) | 0.183 (B = 1,000) |
| Units / player-weeks | 651 / 11,818 |
| Max R-hat | 1.005 |
| **Decision** | **Not confirmed** |

The interval rules out effects of the size seen in the Cubs/Red Sox demo (-0.13 to -0.20 on the
per-fanbase scale). Across the rest of the league, positive sentiment does not predict worse
fielding the next week.

## Bias found in the permutation null (after the test)

The original null shuffled sentiment within each player. That assumes sentiment is unrelated to the
player's recent performance. It isn't: fans react to how a player has been doing. Shuffling breaks
that link, and the model's per-player intercepts are partly shrunk, so they don't absorb it. As a
result the null for AVG, OPS, ERA and COMPOSITE was centered near +0.03, about 3 null sd away from 0.
A synthetic panel where sentiment only reacts to past performance, with no real effect, was rejected
in 10 of 10 replicates.

`correlate` now defaults to a Freedman-Lane null: sentiment is first regressed on the performance
controls, and only the residual part is shuffled. Two-sided p-values are now centered on the null
median. See CHANGELOG 2026-11-18.

**This bias does not change the registered result.** A shifted null makes a one-sided "less than"
result easier to reach, so the registered test was if anything tilted toward confirmation.
`confirm` still reports the registered statistic. The corrected nulls give:

| `FLD_Z` one-sided p | |
|---|---|
| As registered (shuffle null) | 0.183 |
| Shuffle null re-centered on 0 | 0.388 |
| Freedman-Lane null | 0.369 |

## Within-player estimator

The pooled estimator mixes two things: whether well-liked players differ from others (between
players), and whether a player's own good weeks of sentiment come before his worse weeks (within a
player). About 26% of the variance in sentiment is between players. `--estimator mundlak` separates
the two parts and estimates the effect within players only. That changes what is being estimated,
so any future confirmatory use of it needs a new pre-registration.

## Exploratory results (not pre-registered)

All of the runs below use the Freedman-Lane null.

**Whole league, pooled, Cubs and Red Sox players excluded.** Nothing is significant under either
estimator. The smallest q is 0.19 to 0.35. OPS and AVG lean slightly negative (within-player mu
-0.016, one-sided p 0.05 to 0.06), which does not survive correction across the family.

**Per fanbase, whole league.** No cell has q < 0.10, out of about 250. Effects point both ways.

**Cubs and Red Sox demo, re-estimated within players.** The fielding cells that were significant in
the demo shrink, and none survive correction:

| Cell | Demo mu (pooled) | Demo q | Within-player mu | 95% CrI | q |
|---|---|---|---|---|---|
| ERRORS, r/baseball | -0.201 | 0.018 | -0.125 | [-0.266, +0.005] | 0.33 |
| FLD_PCT, r/baseball | -0.127 | 0.018 | -0.071 | [-0.192, +0.036] | 0.33 |
| FLD_PCT, r/CHICubs | -0.152 | 0.018 | -0.117 | [-0.269, +0.025] | 0.32 |
| ERRORS, r/CHICubs | -0.193 | 0.063 | -0.140 | [-0.323, +0.032] | 0.32 |

The demo q-values used the old shuffle null. The within-player rows use Freedman-Lane, so these
rows change the estimator and the null together.

## Conclusion

The demo's sentiment-predicts-worse-fielding result does not hold up. It did not replicate in the
pre-registered league test, and it fades within the demo's own data once each player is compared
with his own usual level. Across the 2026 season we find no evidence that fan sentiment about a
player predicts his next-week performance beyond what his recent performance already predicts.

## Other changes in this round

- Relevance filter: the own-team features had been stuck at 0 since rosters switched to team
  abbreviations (commit a29c147). They now use the player's team in the week of the comment, and
  in a trade week both his old and new team count. Retrained models are slightly better
  (out-of-fold AP 0.829 to 0.831), but nothing has been re-scored yet. The confirmatory run used
  the filter as it was.
- Classifier generalization: the frozen classifier's agreement with Sonnet labels on 1,883
  comments from other fanbases (macro-F1 0.530, Spearman 0.356) was close to its in-domain
  level (0.548, 0.405).
