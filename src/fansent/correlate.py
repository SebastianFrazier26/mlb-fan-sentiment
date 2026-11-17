"""`fansent correlate` (hierarchical method): panel -> per-metric model + permutation null ->
multiple-comparison correction -> analysis_{fanbases,units,oaa}.csv."""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import InferenceConfig, MetricResult, analyze_metric
from .multitest import benjamini_hochberg, holm
from .panel import (
    PanelConfig,
    _day,
    build_oaa_panel,
    build_weekly_panel,
    fanbase_of,
    filter_own_team,
    sentiment_periods,
)
from .util import (
    DEFAULT_DATA_DIR,
    OAA_FILE,
    PLAYER_TEAMS_FILE,
    PLAYERS_FILE,
    REDDIT_ATTRIBUTION_FILE,
    SENTIMENT_FILE,
    STATS_FILE,
)

FANBASES_OUT = "analysis_fanbases.csv"
UNITS_OUT = "analysis_units.csv"
OAA_OUT = "analysis_oaa.csv"
OAA_UNITS_OUT = "analysis_oaa_units.csv"
NEWS_HEADLINES_FILE = "news_headlines.csv"

FAMILY_METRICS = "metrics"
FAMILY_COMPOSITE = "COMPOSITE"
FAMILY_OAA = "OAA"
FAMILY_EXPLORATORY = "exploratory"
FAMILY_ORDER = [FAMILY_COMPOSITE, FAMILY_METRICS, FAMILY_OAA, FAMILY_EXPLORATORY]
# CS_PCT only has catchers, so it is too thin to correct alongside the rest.
EXPLORATORY_METRICS = frozenset({"CS_PCT"})

FANBASE_COLS = [
    "family",
    "exploratory",
    "metric",
    "fanbase",
    "mu_mean",
    "ci_low",
    "ci_high",
    "p_pos",
    "perm_p",
    "perm_p_less",
    "perm_p_greater",
    "q_bh",
    "p_holm",
    "n_units",
    "n_obs",
    "rhat_max",
]


POOL_LEAGUE = "league"
POOL_TEAMS = "teams"


@dataclass(frozen=True)
class Scope:
    """Restricts which sentiment rows are analysed (defaults = whole league, every fanbase).

    teams: keep rows whose (player, Monday-week of the text date) is on one of these team ids.
    exclude_teams: drop those rows instead (the inverse); a player-week absent from
    player_teams.csv is on no team and is kept.
    fanbases / exclude_fanbases: case-insensitive fanbase names, include-only / drop.
    pool_stats: COMPOSITE z-score pool; "teams" z-scores over only the players left after
    the teams and fanbase filters (before the own-team filter), "league" over every player."""

    teams: frozenset[int] | None = None
    fanbases: frozenset[str] | None = None
    exclude_fanbases: frozenset[str] = frozenset()
    exclude_teams: frozenset[int] = frozenset()
    pool_stats: str = POOL_LEAGUE


WHOLE_LEAGUE = Scope()


def resolve_teams(tokens: list[str], teams_path: Path) -> frozenset[int]:
    """Team ids or abbreviations (teams.csv `team`, case-insensitive) -> team ids."""
    abbr: dict[str, int] = {}
    if teams_path.exists():
        df = pd.read_csv(teams_path, dtype=str)
        abbr = {a.upper(): int(i) for i, a in zip(df["team_id"], df["team"], strict=True)}
    out = set()
    for tok in tokens:
        if tok.isdigit():
            out.add(int(tok))
        elif tok.upper() in abbr:
            out.add(abbr[tok.upper()])
        else:
            raise SystemExit(
                f"unknown team {tok!r}: not a team id or an abbreviation in {teams_path}"
            )
    return frozenset(out)


def apply_scope(
    scores: pd.DataFrame,
    scope: Scope,
    pcfg: PanelConfig,
    weekly_teams: frozenset[tuple[str, str, int]] | None,
) -> pd.DataFrame:
    if scope.fanbases is not None or scope.exclude_fanbases:
        fb = fanbase_of(scores["source"], scores["subreddit"], pcfg.subreddit_fanbase).str.lower()
        keep = pd.Series(True, index=scores.index)
        if scope.fanbases is not None:
            keep &= fb.isin({f.lower() for f in scope.fanbases})
        keep &= ~fb.isin({f.lower() for f in scope.exclude_fanbases})
        scores = scores[keep]
    if scope.teams is not None or scope.exclude_teams:
        if weekly_teams is None:
            raise SystemExit(
                "--teams/--exclude-teams need player_teams.csv (run `fansent rosters`)."
            )
        day = _day(scores["date"])
        week = (day - pd.to_timedelta(day.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
        pairs = list(zip(scores["player_id"].astype(str), week, strict=True))
        if scope.teams is not None:
            on_team = {(p, w) for p, w, t in weekly_teams if t in scope.teams}
            scores = scores[[pw in on_team for pw in pairs]]
            pairs = [pw for pw in pairs if pw in on_team]
        if scope.exclude_teams:
            off = {(p, w) for p, w, t in weekly_teams if t in scope.exclude_teams}
            scores = scores[[pw not in off for pw in pairs]]
    if scores.empty:
        raise SystemExit(
            "--teams/--exclude-teams/--fanbases/--exclude-fanbases left no sentiment rows."
        )
    return scores


@dataclass
class Outputs:
    fanbases: pd.DataFrame
    units: pd.DataFrame
    oaa_fanbases: pd.DataFrame
    oaa_units: pd.DataFrame
    skipped: list[str]
    dropped: dict[str, int]
    seconds: float


def _read_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, dtype=str) if path.exists() else None


def read_player_teams(path: Path) -> dict[str, int]:
    df = pd.read_csv(path, dtype={"player_id": str})
    return dict(zip(df["player_id"], df["team_id"].astype(int), strict=True))


def read_weekly_teams(path: Path) -> frozenset[tuple[str, str, int]] | None:
    """player_teams.csv -> {(player_id, week_start, team_id)}; None (with a warning) if absent,
    so the filter falls back to the single team in players.csv."""
    if not path.exists():
        warnings.warn(
            f"{path} not found: own-team filter falls back to players.csv team_id "
            "(a traded player is attributed to his last team for the whole season)",
            stacklevel=2,
        )
        return None
    df = pd.read_csv(path, dtype={"player_id": str, "week_start": str})
    return frozenset(zip(df["player_id"], df["week_start"], df["team_id"].astype(int), strict=True))


def family_of(metric: str) -> str:
    if metric in EXPLORATORY_METRICS:
        return FAMILY_EXPLORATORY
    if metric in (FAMILY_COMPOSITE, FAMILY_OAA):
        return metric
    return FAMILY_METRICS


ALTERNATIVE_COLUMN = {
    "two-sided": "perm_p",
    "less": "perm_p_less",
    "greater": "perm_p_greater",
}


def correct(df: pd.DataFrame, alternative: str = "two-sided") -> pd.DataFrame:
    """Assign `family`/`exploratory` and add q_bh and p_holm within each family only, from the
    permutation p of the chosen alternative. Exploratory rows get NA corrections."""
    pcol = ALTERNATIVE_COLUMN[alternative]
    out = df.copy()
    out["family"] = out["metric"].map(family_of)
    out["exploratory"] = out["family"] == FAMILY_EXPLORATORY
    out["q_bh"] = np.nan
    out["p_holm"] = np.nan
    for idx in out[~out["exploratory"]].groupby("family").groups.values():
        p = out.loc[idx, pcol].to_numpy()
        out.loc[idx, "q_bh"] = benjamini_hochberg(p)
        out.loc[idx, "p_holm"] = holm(p)
    return out


def analyze(
    stats: pd.DataFrame,
    scores: pd.DataFrame,
    sidecars: list[pd.DataFrame],
    oaa: pd.DataFrame | None,
    pcfg: PanelConfig,
    icfg: InferenceConfig,
    metrics: list[str] | None = None,
    include_composite: bool = True,
    player_teams: Mapping[str, int] | None = None,
    weekly_teams: frozenset[tuple[str, str, int]] | None = None,
    scope: Scope = WHOLE_LEAGUE,
) -> Outputs:
    t0 = time.perf_counter()
    scores = apply_scope(scores, scope, pcfg, weekly_teams)
    if scope.pool_stats == POOL_TEAMS:
        stats = stats[stats["player_id"].astype(str).isin(set(scores["player_id"].astype(str)))]
    scores, dropped = filter_own_team(scores, pcfg, player_teams, weekly_teams)
    weekly_sent, monthly_sent = sentiment_periods(scores, sidecars, pcfg)
    panel = build_weekly_panel(stats, weekly_sent, pcfg, metrics, include_composite)
    results: list[MetricResult] = []
    skipped: list[str] = []
    for metric in sorted(panel["metric"].unique()):
        try:
            results.append(analyze_metric(panel, metric, icfg))
        except ValueError as e:
            skipped.append(f"{metric}: {e}")
    if oaa is not None:
        oaa_panel = build_oaa_panel(oaa, monthly_sent, pcfg)
        if len(oaa_panel):
            try:
                results.append(analyze_metric(oaa_panel, "OAA", icfg))
            except ValueError as e:
                skipped.append(f"OAA: {e}")
    if not results:
        raise SystemExit(
            "No (metric, fanbase) had enough data; lower --min-weeks/--min-mentions or collect more."
        )
    fb = correct(pd.concat([r.fanbases for r in results], ignore_index=True), icfg.alternative)[
        FANBASE_COLS
    ]
    units = pd.concat([r.units for r in results], ignore_index=True)
    is_oaa = fb["metric"] == "OAA"
    return Outputs(
        fanbases=fb[~is_oaa].reset_index(drop=True),
        units=units[units["metric"] != "OAA"].reset_index(drop=True),
        oaa_fanbases=fb[is_oaa].reset_index(drop=True),
        oaa_units=units[units["metric"] == "OAA"].reset_index(drop=True),
        skipped=skipped,
        dropped=dropped,
        seconds=time.perf_counter() - t0,
    )


def format_table(fb: pd.DataFrame, correction: str, alpha: float) -> str:
    col = "q_bh" if correction == "bh" else "p_holm"
    t = fb.copy()
    t["sig"] = (t[col] < alpha).map({True: "*", False: ""})
    for c in ("mu_mean", "ci_low", "ci_high"):
        t[c] = t[c].map("{:+.3f}".format)
    for c in ("p_pos", "perm_p", "perm_p_less", "perm_p_greater", "q_bh", "p_holm", "rhat_max"):
        t[c] = t[c].map("{:.3f}".format).replace("nan", "NA")
    blocks = []
    for fam in FAMILY_ORDER:
        part = t[t["family"] == fam].drop(columns=["family", "exploratory"])
        if len(part):
            note = " (raw perm_p, not corrected)" if fam == FAMILY_EXPLORATORY else ""
            blocks.append(f"-- family: {fam}{note}\n{part.to_string(index=False)}")
    return "\n\n".join(blocks)


def load_inputs(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """(stats, sentiment scores, attribution sidecars that exist) from a data directory."""
    stats = pd.read_csv(data_dir / STATS_FILE, dtype={"player_id": str})
    sentiment_csv = data_dir / SENTIMENT_FILE
    if not sentiment_csv.exists():
        raise SystemExit(f"{sentiment_csv} not found — run `fansent score` first.")
    scores = pd.read_csv(sentiment_csv, dtype={"text_id": str, "player_id": str})
    sidecars = [
        _read_optional(data_dir / REDDIT_ATTRIBUTION_FILE),
        _read_optional(data_dir / NEWS_HEADLINES_FILE),
    ]
    return stats, scores, [s for s in sidecars if s is not None]


def needs_weekly_teams(pcfg: PanelConfig, scope: Scope) -> bool:
    return bool(pcfg.fanbase_teams) or scope.teams is not None or bool(scope.exclude_teams)


def run(
    pcfg: PanelConfig,
    icfg: InferenceConfig,
    metrics: list[str] | None,
    include_composite: bool,
    oaa_path: Path | None,
    correction: str,
    alpha: float,
    data_dir: Path = DEFAULT_DATA_DIR,
    scope: Scope = WHOLE_LEAGUE,
) -> None:
    stats, scores, sidecars = load_inputs(data_dir)
    default_oaa = data_dir / OAA_FILE
    path = oaa_path or (default_oaa if default_oaa.exists() else None)
    oaa = pd.read_csv(path, dtype={"player_id": str}) if path else None

    out = analyze(
        stats, scores, sidecars, oaa, pcfg, icfg, metrics,
        include_composite, read_player_teams(data_dir / PLAYERS_FILE),
        read_weekly_teams(data_dir / PLAYER_TEAMS_FILE) if needs_weekly_teams(pcfg, scope) else None,
        scope,
    )  # fmt: skip
    fanbases_out, units_out, oaa_out = (
        data_dir / FANBASES_OUT,
        data_dir / UNITS_OUT,
        data_dir / OAA_OUT,
    )
    out.fanbases.to_csv(fanbases_out, index=False)
    out.units.to_csv(units_out, index=False)
    if len(out.oaa_fanbases):
        out.oaa_units.to_csv(data_dir / OAA_UNITS_OUT, index=False)
        out.oaa_fanbases.to_csv(oaa_out, index=False)

    if out.dropped:
        print(f"own-team filter dropped text-player rows: {out.dropped}")
    print(
        f"\n=== Fanbase effect mu_k: sentiment_t -> better performance t+1 "
        f"({correction} < {alpha}, {icfg.alternative} permutation p) ==="
    )
    print(format_table(pd.concat([out.fanbases, out.oaa_fanbases]), correction, alpha))
    for s in out.skipped:
        print(f"skipped {s}")
    print(f"\n{fanbases_out}, {units_out}" + (f", {oaa_out}" if len(out.oaa_fanbases) else ""))
    print(f"mu_k is in sd(next-period performance) per sd(sentiment); {out.seconds:.0f}s.")
