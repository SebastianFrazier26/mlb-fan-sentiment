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
make setup          # uv sync
make check          # ruff + pytest
```

Docker alternative: `make docker-build`, then `make docker-run ARGS="rosters"`.

## Pipeline

```sh
uv run fansent rosters --season 2025          # Cubs + Red Sox -> data/players.csv
uv run fansent stats --season 2025            # weekly OPS/ERA -> data/stats.csv
# texts.csv + news_tone.csv come from the collectors (see below)
uv run fansent score                          # VADER -> data/sentiment.csv
uv run fansent correlate                      # -> data/analysis_{players,fanbases}.csv
```

## Where the text data comes from

Reddit and news collection lives in a separate, private companion repo
(`fansent-collect`), kept out of this public repo by design. It reads
`data/players.csv` and writes into `data/`:

- `texts.csv` — `text_id, source, subreddit, date, created_utc, player_id, player_name, text`
  (one row per text × mentioned player; `date` is the UTC day)
- `news_tone.csv` — `player_id, player_name, team, date, tone`

Anything that produces CSVs with those columns can feed `score` / `correlate`.

## Data sources

- **MLB Stats API** (statsapi.mlb.com) — rosters + game logs; free, no key, read-only.
- **Reddit** (via `fansent-collect`) — fan comments; personal, non-commercial research.
- **GDELT DOC 2.0** (via `fansent-collect`) — news per player; free, no key.

## Notes & caveats

- Name matching: full name always; bare last name only when unique among
  tracked players (≥4 chars). Common-word last names can still false-positive.
- Weekly windows are Mon–Sun. Player-weeks below 10 PA / 3 IP are dropped.
- Weekly OPS/ERA is extremely noisy — expect small correlations; the
  cross-fanbase *comparison* is the robust readout, not any single player's r.

## License

Apache-2.0 — see [LICENSE](LICENSE).
