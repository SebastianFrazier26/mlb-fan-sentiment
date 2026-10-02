# CONTINUITY — fansent

## [PLANS]
- 2026-08-18T00:00Z [USER] Goal: correlate fan sentiment (Reddit + news) with MLB weekly player stats; compare fanbases (r/CHICubs vs r/redsox vs r/baseball vs news) on predictive accuracy ("are the fans always right"). Personal research, never commercial.
- 2026-08-18T00:00Z [ASSUMPTION] MVP scope: Cubs + Red Sox rosters only; top-100 MLB players deferred.
- 2026-08-18T00:00Z [USER] Historical Reddit via Academic Torrents subreddit dumps (user downloads .zst files themselves); forward collection via free Reddit API (praw, optional extra).

## [DECISIONS]
- 2026-08-18T00:00Z [USER] Measure FAN sentiment about players (not player-expressed mood).
- 2026-08-18T00:00Z [ASSUMPTION] Core stat design: partial correlation of sentiment_t vs perf_{t+1} controlling perf_t, to separate prediction from reaction/mean-reversion. Weekly Mon–Sun windows; OPS (hitters) / ERA (pitchers, sign-flipped); min 10 PA / 3 IP per week; min 3 mentions/week; min 8 weeks per (fanbase, player); fanbase aggregate via Fisher-z mean.
- 2026-08-18T00:00Z [ASSUMPTION] News sentiment via GDELT timelinetone (free, pre-scored, 2017+) instead of scraping articles.
- 2026-08-18T00:00Z [TOOL] Docker daemon not running on host -> dev via uv venv; Dockerfile + Make targets provided for the container path.
- 2026-08-18T00:00Z [ASSUMPTION] Default season 2025 (matches dump coverage through 2025-12; 2026 season still in progress).

## [PROGRESS]
- 2026-08-18T00:00Z [CODE] Scaffolded ~/Developer/mlb-fan-sentiment: fansent CLI (rosters, stats, reddit-dump, reddit-live, news, score, correlate), tests, Dockerfile, Makefile, README.
- 2026-08-18T00:00Z [TOOL] Verified: ruff clean, 9/9 pytest pass; live smokes OK (rosters=108 players 2025 fullSeason, stats gameLog->weekly OPS, GDELT timelinetone real data); synthetic end-to-end recovered planted lagged signal (r_partial=0.99), control pitcher correctly excluded.
- 2026-08-18T00:00Z [TOOL] Full 2025 pulls launched: stats (all 108) and GDELT news tone 2025-03-01..2025-11-15. Docker image NOT built (daemon down) — `make docker-build` untested.

## [DISCOVERIES]
- 2026-08-18T00:00Z [TOOL] GDELT rate-limits ~1 req/5s per IP (429 otherwise) — collector uses 5.5s pause + backoff retry.
- 2026-08-18T00:00Z [TOOL] Academic Torrents subreddit dumps cover 2005-06..2025-12, per-subreddit files (top 40k subs) — solves Reddit historical backfill.
- 2026-08-18T00:00Z [TOOL] Reddit API free tier (100 QPM/OAuth client, non-commercial) confirmed still current as of 2026-08.

## [OUTCOMES]
- 2026-08-18T00:00Z [CODE] MVP pipeline built and verified (unit + synthetic e2e + live smokes). Remaining: user torrents Reddit .zst dumps (baseball/CHICubs/redsox) and runs reddit-dump -> score -> correlate; optional reddit-live creds; top-100 player list; Docker build unverified.
