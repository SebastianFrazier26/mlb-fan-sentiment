# Portfolio demo notes

Written 2026-11-22, when the 2026 analysis closed. What the portfolio entry for this project
needs, and what exists already.

## Hosted URL

- Live: an animated post at <https://sebastianfrazier.com/blog/fan-sentiment/>, built in the
  portfolio repository from an aggregate-only snapshot of these results.
- The GitHub Pages page is retired; the static results are the figures in the README.
- Repository: <https://github.com/SebastianFrazier26/mlb-fan-sentiment>.

## Screenshots to take

1. The opener and the "not confirmed" reveal. This is the card image.
2. The whole-league chart: every interval sitting on zero.
3. The Cubs and Red Sox chart: the fielding rows to the left of zero, the signal that prompted
   the test.

No screenshot may show comment text, usernames or per-player rows; the post is built from
aggregate numbers only.

## Short write-up (draft)

> **Are the fans always right?** When a fanbase is upbeat about a player one week, does he
> play better the next? I collected a season of Reddit comments from all 30 team subreddits and
> r/baseball, built a target-aware sentiment classifier and a hierarchical Bayesian model from
> scratch in numpy, and tested it against permutation nulls. A first pass on two teams
> suggested that optimism came before worse fielding. I pre-registered one test of that on the
> other 28 teams and it failed: the effect was zero to two decimal places. Along the way I found
> and fixed a bias in my own permutation test and published a correction to my own write-up.
> The answer is no, and the project shows how I got to a no I trust.

## Points worth making in the portfolio entry

- Built from scratch: Gibbs sampler, classifier (TF-IDF, fastText reader, logistic regression),
  relevance filter, permutation nulls, power analysis. No modeling libraries beyond numpy/scipy.
- A pre-registered test with a negative result, reported as such.
- A found-and-fixed bias (the permutation null) and a dated correction in the results note.
- Privacy: only aggregate numbers are published; the snapshot format rejects free text.

## Not part of the demo

- Collectors and all comment text are private and stay out of this repository and the page.
- `data/` is gitignored; only aggregate snapshots under `results/published/` are public.
