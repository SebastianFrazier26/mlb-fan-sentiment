# fansent — are the fans always right?

Personal research CLI correlating fan sentiment (Reddit + news) with MLB player
performance, and comparing fanbases on how "right" they are about their players.

**Question:** does fan sentiment in week *t* predict a player's performance in
week *t+1*, beyond what week *t* performance already predicts? For each
(fanbase, player) we compute a partial correlation of weekly sentiment vs.
next-week performance, controlling for current-week performance (this control
is what separates real predictive signal from sentiment merely *reacting* to
stats). Fanbases are then ranked by their players' average correlation.

- `mean_r > 0` — that fanbase's mood carries predictive signal ("fans are right")
- `mean_r < 0` — fans overreact; performance mean-reverts against their mood

Fanbases compared: r/CHICubs, r/redsox, r/baseball (neutral baseline), and
`news` (GDELT media tone) — each judged only on the tracked players it talks about.

## Setup

```sh
make setup          # uv sync (or: uv sync --extra live for the Reddit API collector)
make check          # ruff + pytest
```

Docker alternative: `make docker-build`, then `make docker-run ARGS="rosters"`.

## Pipeline

```sh
uv run fansent rosters --season 2025          # Cubs + Red Sox -> data/players.csv
uv run fansent stats --season 2025            # weekly OPS/ERA -> data/stats.csv
uv run fansent reddit-dump path/to/*.zst \
    --since 2025-03-01 --until 2025-11-15     # dump mentions -> data/texts.csv
uv run fansent news --start 2025-03-01 --end 2025-11-15   # GDELT tone -> data/news_tone.csv
uv run fansent score                          # VADER -> data/sentiment.csv
uv run fansent correlate                      # -> data/analysis_{players,fanbases}.csv
```

`fansent reddit-live` (optional, `--extra live` + `REDDIT_CLIENT_ID`/
`REDDIT_CLIENT_SECRET` env vars) collects new comments going forward via the
free Reddit API tier.

## Getting historical Reddit data

The Reddit API can't reach old content. Use the community Pushshift-derived
dumps on Academic Torrents ("Subreddit comments/submissions 2005-06 to
2025-12", maintained by u/Watchful1 — <https://academictorrents.com/details/3e3f64dee22dc304cdd2546254ca1f8e8ae542b4>):
in your torrent client, select **only** the files for `baseball`, `CHICubs`,
and `redsox` (comments + submissions), then run `fansent reddit-dump` on the
downloaded `.zst` files. Personal research use only.

## Data sources

- **MLB Stats API** (statsapi.mlb.com) — rosters + game logs; free, no key, read-only.
- **Reddit dumps / Reddit API** — fan comments; free tier, non-commercial.
- **GDELT DOC 2.0** — daily news tone per player; free, no key; coverage from 2017.

## Notes & caveats

- Name matching: full name always; bare last name only when unique among
  tracked players (≥4 chars). Common-word last names can still false-positive.
- Weekly windows are Mon–Sun. Player-weeks below 10 PA / 3 IP are dropped.
- Weekly OPS/ERA is extremely noisy — expect small correlations; the
  cross-fanbase *comparison* is the robust readout, not any single player's r.
