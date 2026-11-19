# Pre-registration: fielding confirmatory test

Written 2026-11-17 (04:27 UTC), before any league-wide (non-Cubs/Red Sox) data was matched, scored or analyzed.

## Why this exists

The corrected Cubs + Red Sox run (2026 season) found an effect nobody planned to test: more positive fan
sentiment about a player in week *t* went with worse fielding in week *t+1*. It survived the within-family
multiple-comparison correction and a battery of robustness checks (leave-one-player-out, count and logit
models, timing placebos, a second sentiment scorer, playing-time / position / team-mood controls). It was
fragile in r/CHICubs (driven by two catchers) and robust in r/baseball. Those r/baseball and r/CHICubs
results draw on the same players and the same fielding outcomes, so they are one observation, not a
replication. An unplanned result found by searching many tests needs an independent, pre-specified test
before it is reported as a finding. This file fixes that test in advance.

## Hypothesis

More positive fan sentiment about a player in week *t* predicts worse fielding by that player in week *t+1*,
beyond what his week-*t* and season-to-date fielding already predict.

## Data

- Season 2026 regular season, Mon–Sun weeks.
- Player-weeks for players who were **not** on the Chicago Cubs (112) or Boston Red Sox (111) that week
  (`player_teams.csv`, weekly 40-man rosters). Cubs/Red Sox player-weeks are excluded entirely: they generated
  the hypothesis.
- Fanbases: the 28 other team-subreddit fanbases (own-team rule by week) and r/baseball. News excluded.
- Sentiment: the trained classifier (`models/clf_big.npz`) with the relevance filter, as in the corrected run.
  The relevance filter may be retrained on league data before this test is run; the sentiment classifier is
  frozen as of this date.

## Outcome (one test)

`FLD_Z`: the chances-weighted mean of pooled z-scores of fielding percentage and of negated errors, oriented
so higher = better fielding. Week-*t* `FLD_Z` and season-to-date `FLD_Z` are the controls.

## Model and inference

- The existing hierarchical model (per-unit sentiment slopes around fanbase means), 4 chains × 6,000
  iterations, 3,000 burn-in, max R-hat < 1.05 required.
- Primary estimate: the **pooled** effect across all included fanbases (all units treated as one group).
- Within-unit permutation null, B = 1,000.
- Existing minimum-sample thresholds (chances, mentions per week, weeks per unit) unchanged.

## Decision rule

- **Confirmed** if the pooled effect is negative with one-sided permutation p < 0.05.
- **Not confirmed** otherwise. The pooled estimate and 95% credible interval are reported either way.
- Per-fanbase estimates are descriptive only and carry no confirmatory weight.
- The result is reported whichever way it comes out.

## Anything not specified here

Any analysis not listed above (other metrics, other fanbases, alternative outcomes, subgroups) is
exploratory and will be labeled as such.
