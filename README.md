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
uv run fansent rosters --season 2026          # Cubs + Red Sox -> data/players.csv
uv run fansent stats --season 2026            # weekly stats -> data/stats.csv
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
- Weekly windows are Mon–Sun (regular season only). `analyze.py` still uses OPS/ERA and drops
  player-weeks below 10 PA / 3 IP.
- Weekly OPS/ERA is extremely noisy — expect small correlations; the
  cross-fanbase *comparison* is the robust readout, not any single player's r.

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
