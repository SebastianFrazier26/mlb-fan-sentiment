# Repo guidance (fansent)

- Python 3.12+; deps managed by `uv` (`make setup`); package lives in `src/fansent/`.
- Container workflow: `Dockerfile` at repo root; `make docker-build` /
  `make docker-run ARGS="..."` (mounts `./data`). Local `uv` venv is the
  day-to-day dev path; Docker is the reproducible path.
- Checks: `make check` (ruff + pytest). Run before considering a change done.
- All generated CSVs live in `data/` (gitignored). Never commit dump files.
- Remote APIs used here (MLB Stats API) are read-only GETs; keep it that way.
- Reddit/GDELT collectors live in the PRIVATE `fansent-collect` repo
  (`../fansent-collect`). Never add scrapers/collectors to this public repo; it
  only consumes their CSVs (`texts.csv`, `news_tone.csv`).
- Continuity briefing: `.agent/CONTINUITY.md` — read at the start of a session,
  update on meaningful deltas.
