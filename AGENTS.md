# Repo guidance (fansent)

- Python 3.12+; deps managed by `uv` (`make setup`); package lives in `src/fansent/`.
- Container workflow: `Dockerfile` at repo root; `make docker-build` /
  `make docker-run ARGS="..."` (mounts `./data`). Local `uv` venv is the
  day-to-day dev path; Docker is the reproducible path.
- Checks: `make check` (ruff + pytest). Run before considering a change done.
- All generated CSVs live in `data/` (gitignored). Never commit dump files.
- Remote APIs used (MLB Stats API, GDELT, Reddit) are read-only GETs; keep it
  that way. Reddit credentials come from env vars, never files in the repo.
- Continuity briefing: `.agent/CONTINUITY.md` — read at the start of a session,
  update on meaningful deltas.
