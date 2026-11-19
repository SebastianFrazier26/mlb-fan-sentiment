"""`fansent confirm`: the pre-registered league-wide fielding test (docs/preregistration.md).

The specification is constants here, not flags, so the run cannot drift from what was
registered. Everything else (thresholds, scorer outputs) is the pipeline's existing defaults.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .composite import FLD_Z
from .correlate import Scope, analyze, load_inputs, read_player_teams, read_weekly_teams
from .inference import InferenceConfig
from .model_data import POOLED
from .panel import NEWS_FANBASE, PanelConfig
from .util import PLAYER_TEAMS_FILE, PLAYERS_FILE

RESULT_FILE = "confirm_result.json"
EXCLUDED_TEAM_IDS = frozenset({111, 112})  # Red Sox, Cubs: they generated the hypothesis
ALPHA = 0.05
RHAT_MAX = 1.05

CONFIRMED = "CONFIRMED"
NOT_CONFIRMED = "NOT CONFIRMED"
INVALID = "INVALID (R-hat gate failed)"


@dataclass(frozen=True)
class Verdict:
    decision: str
    mu_mean: float
    ci_low: float
    ci_high: float
    perm_p_less: float
    rhat_max: float
    rhat_ok: bool


def decide(mu_mean: float, ci_low: float, ci_high: float, perm_p_less: float, rhat_max: float):
    """Confirmed iff the pooled effect is negative and the one-sided permutation p < 0.05.

    A fit whose max R-hat is not below 1.05 (or is NaN) has not converged, so the estimate is
    not a test result: the verdict is INVALID rather than either outcome."""
    rhat_ok = bool(rhat_max < RHAT_MAX)
    if not rhat_ok:
        decision = INVALID
    elif mu_mean < 0 and perm_p_less < ALPHA:
        decision = CONFIRMED
    else:
        decision = NOT_CONFIRMED
    return Verdict(
        decision, float(mu_mean), float(ci_low), float(ci_high), float(perm_p_less),
        float(rhat_max), rhat_ok,
    )  # fmt: skip


def run(data_dir: Path, workers: int = 1, seed: int = 0) -> Verdict:
    stats, scores, sidecars = load_inputs(data_dir)
    weekly_teams = read_weekly_teams(data_dir / PLAYER_TEAMS_FILE)
    if weekly_teams is None:
        raise SystemExit("confirm needs player_teams.csv (run `fansent rosters`).")
    icfg = InferenceConfig(
        n_chains=4,
        n_iter=6000,
        burn=3000,
        perm_b=1000,
        pool_fanbases=True,
        alternative="less",
        workers=workers,
        seed=seed,
    )
    out = analyze(
        stats,
        scores,
        sidecars,
        None,
        PanelConfig(),
        icfg,
        [FLD_Z],
        False,
        read_player_teams(data_dir / PLAYERS_FILE),
        weekly_teams,
        Scope(exclude_teams=EXCLUDED_TEAM_IDS, exclude_fanbases=frozenset({NEWS_FANBASE})),
    )
    row = out.fanbases[(out.fanbases["metric"] == FLD_Z) & (out.fanbases["fanbase"] == POOLED)]
    if len(row) != 1:
        raise SystemExit(f"expected one pooled {FLD_Z} row, got {len(row)}; skipped: {out.skipped}")
    r = row.iloc[0]
    verdict = decide(r.mu_mean, r.ci_low, r.ci_high, r.perm_p_less, r.rhat_max)
    report(verdict, r, icfg, data_dir)
    return verdict


def report(v: Verdict, row: pd.Series, icfg: InferenceConfig, data_dir: Path) -> None:
    print("\n=== Confirmatory test (docs/preregistration.md): pooled FLD_Z effect ===")
    print(f"pooled mu     {v.mu_mean:+.4f} sd(next-week FLD_Z) per sd(sentiment)")
    print(f"95% CrI       [{v.ci_low:+.4f}, {v.ci_high:+.4f}]")
    print(f"one-sided p   {v.perm_p_less:.4f}  (H1: mu < 0; two-sided {row.perm_p:.4f}, B=1000)")
    print(f"units / obs   {int(row.n_units)} / {int(row.n_obs)}")
    flag = "" if v.rhat_ok else f"   ** FAILS the R-hat < {RHAT_MAX} requirement **"
    print(f"max R-hat     {v.rhat_max:.4f}{flag}")
    print(f"\nDECISION: {v.decision}")
    result = {
        "preregistration": "docs/preregistration.md",
        "run_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "metric": FLD_Z,
        "excluded_team_ids": sorted(EXCLUDED_TEAM_IDS),
        "excluded_fanbases": [NEWS_FANBASE],
        "alpha": ALPHA,
        "rhat_threshold": RHAT_MAX,
        "chains": icfg.n_chains,
        "iterations": icfg.n_iter,
        "burn": icfg.burn,
        "perm_b": icfg.perm_b,
        "seed": icfg.seed,
        "n_units": int(row.n_units),
        "n_obs": int(row.n_obs),
        "perm_p_two_sided": float(row.perm_p),
        **asdict(v),
    }
    path = data_dir / RESULT_FILE
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path)
