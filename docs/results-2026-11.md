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

All of the runs below use the Freedman-Lane null. They were re-run on 2026-11-21 after re-scoring
with the fixed relevance filter (see the end of this note); the numbers are from that re-run.

**Whole league, pooled, Cubs and Red Sox players excluded.** Nothing is significant under either
estimator. The smallest q is 0.14 to 0.15. OPS and AVG lean slightly negative (within-player mu
-0.018 and -0.016, one-sided p about 0.04, q 0.27), which does not survive correction across the
family. The ERA fits did not converge (R-hat 1.064, above the 1.05 limit), so the ERA rows are not
usable. `FLD_Z`, re-checked on the new scores as a sensitivity check only, is mu +0.0014, 95% CrI
[-0.018, +0.021].

**Per fanbase, whole league.** No cell has q < 0.10, out of 218. The smallest q is 0.42. Effects
point both ways.

**Cubs and Red Sox demo, re-estimated within players.** The fielding cells weaken under the
corrected null but stay borderline. They do not disappear:

| Cell | Pooled mu | Pooled q | Within-player mu | 95% CrI | Within-player q |
|---|---|---|---|---|---|
| ERRORS, r/baseball | -0.182 | 0.072 | -0.158 | [-0.281, -0.047] | 0.049 |
| FLD_PCT, r/baseball | -0.118 | 0.085 | -0.095 | [-0.201, -0.003] | 0.049 |
| FLD_PCT, r/CHICubs | -0.152 | 0.085 | -0.130 | [-0.267, +0.004] | 0.049 |
| ERRORS, r/CHICubs | -0.193 | 0.085 | -0.177 | [-0.361, -0.001] | 0.049 |

The demo's original q-values, under the old shuffle null, were 0.018 to 0.063. The r/redsox
fielding cells are not significant under either estimator (q 0.28 or higher).

**Correction (2026-11-21).** The first version of this note reported within-player q of 0.32 to
0.33 for these cells and said the effect fades. Those rows had been fitted on an older, smaller
collection (486,942 comments) rather than the corrected demo data (913,344) behind the pooled
numbers. On the same data as the pooled rows, with the old scores, the within-player q is 0.018
to 0.045. The re-score moves it only to 0.049.

## Conclusion

The demo's sentiment-predicts-worse-fielding result is weaker than first reported, but it does
not vanish. Under the corrected null it is borderline within the Cubs and Red Sox data (q about
0.05 within players, 0.07 to 0.09 pooled). It did not replicate in the pre-registered league
test, and nothing comparable appears in any other fanbase. Across the 2026 season we find no
evidence that fan sentiment about a player predicts his next-week performance beyond what his
recent performance already predicts. The Cubs and Red Sox fielding signal is best read as
unconfirmed and specific to those fanbases.

## Other changes in this round

- Relevance filter: the own-team features had been stuck at 0 since rosters switched to team
  abbreviations (commit a29c147). They now use the player's team in the week of the comment, and
  in a trade week both his old and new team count. Retrained models are slightly better
  (out-of-fold AP 0.829 to 0.831). Both the league and demo data were re-scored on 2026-11-21.
  The share of comments dropped as off-topic went from 5.0% to 5.3% in the league data and from
  16.8% to 14.6% in the demo data. Sentiment scores of comments kept in both runs are unchanged.
  The confirmatory run used the filter as it was, and was not re-run.
- Classifier generalization: the frozen classifier's agreement with Sonnet labels on 1,883
  comments from other fanbases (macro-F1 0.530, Spearman 0.356) was close to its in-domain
  level (0.548, 0.405).
